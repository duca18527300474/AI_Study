"""
web.py — 嵌入式研发 Log/文档智能分析助手（RAG 增强版）
======================================================
基于 Streamlit + DeepSeek API + Chroma 向量数据库 + 本地 Embedding。

核心 RAG 流程：
  上传文件 → 文本分块(500字/块) → 本地 Embedding → 存入 Chroma
  用户提问 → Embedding 问题 → 检索 Top-3 相关块 → 拼入 Prompt → DeepSeek 回答

依赖：
  streamlit / openai / python-dotenv / chromadb / sentence-transformers
"""

import os
import sys
from datetime import datetime

from dotenv import load_dotenv

sys.stdout.reconfigure(encoding="utf-8")
load_dotenv()

import streamlit as st
from openai import OpenAI

# ============================================================
# 页面基础配置
# ============================================================
st.set_page_config(
    page_title="嵌入式 Log 分析助手 (RAG)",
    page_icon="📟",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ============================================================
# 常量
# ============================================================
MAX_FILE_SIZE_BYTES = 2 * 1024 * 1024   # 上传上限 2MB
CHUNK_SIZE = 500                         # 文本分块：每块 500 字
CHUNK_OVERLAP = 50                       # 相邻块重叠 50 字，避免语义截断
RETRIEVAL_TOP_K = 3                      # 检索时取最相关的 3 个块
CHROMA_COLLECTION_NAME = "log_chunks"    # Chroma 集合名
CHROMA_PERSIST_DIR = "./chroma_db"       # Chroma 持久化目录

# ============================================================
# 会话状态初始化
# ============================================================
if "messages" not in st.session_state:
    st.session_state.messages = []           # 对话历史（仅 user + assistant）

if "file_content" not in st.session_state:
    st.session_state.file_content = None     # 上传文件的原始文本

if "file_name" not in st.session_state:
    st.session_state.file_name = None        # 上传文件的文件名

if "indexing_done" not in st.session_state:
    st.session_state.indexing_done = False   # 是否已完成向量化入库

if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0         # 分块数量

# ============================================================
# DeepSeek 客户端（缓存复用）
# ============================================================
@st.cache_resource
def get_deepseek_client():
    """创建并缓存 DeepSeek API 客户端"""
    return OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
    )

client = get_deepseek_client()

# ============================================================
# 本地 Embedding 模型（缓存复用，首次运行自动下载 ~420MB）
# ============================================================
@st.cache_resource
def get_embedding_model():
    """
    加载本地多语言 Embedding 模型。
    使用 paraphrase-multilingual-MiniLM-L12-v2：
      - 支持 50+ 语言（含中文）
      - 输出 384 维向量
      - 首次运行从 HuggingFace 下载，后续从缓存加载
    """
    from sentence_transformers import SentenceTransformer

    print("[RAG] 正在加载 Embedding 模型（首次约需下载 420MB）...")
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    print("[RAG] Embedding 模型加载完成 ✓")
    return model

# ============================================================
# Chroma 向量数据库客户端（缓存复用）
# ============================================================
@st.cache_resource
def get_chroma_client():
    """
    创建 Chroma PersistentClient，数据持久化到 ./chroma_db 目录。
    页面刷新后索引不丢失。
    """
    import chromadb

    print(f"[RAG] 正在初始化 Chroma 客户端（持久化路径: {CHROMA_PERSIST_DIR}）...")
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    print("[RAG] Chroma 客户端初始化完成 ✓")
    return client

# ============================================================
#                           工具函数
# ============================================================

def validate_and_read_file(uploaded_file) -> tuple[bool, str | None, str | None]:
    """
    五层校验：扩展名 → 文件大小 → 空文件 → 编码 → 纯空白。
    返回 (是否通过, 文件内容|None, 错误消息|None)
    """
    raw_bytes = uploaded_file.getvalue()
    file_size_mb = len(raw_bytes) / (1024 * 1024)
    filename = uploaded_file.name

    if not filename.lower().endswith((".txt", ".log")):
        return False, None, (
            f"❌ 不支持的文件格式 **`{filename}`**，仅允许 `.txt` 或 `.log` 文件。"
        )

    if len(raw_bytes) > MAX_FILE_SIZE_BYTES:
        return False, None, (
            f"❌ 文件过大（{file_size_mb:.1f} MB），已超过 **2 MB** 上限。"
            f"请裁剪日志后重新上传。"
        )

    if len(raw_bytes) == 0:
        return False, None, "❌ 文件内容为空，请上传有效的日志文件。"

    try:
        content = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        try:
            content = raw_bytes.decode("gbk", errors="replace")
        except Exception:
            return False, None, "❌ 无法识别文件编码，请确认文件为文本格式（UTF-8 / GBK）。"

    if not content.strip():
        return False, None, "❌ 文件内容仅含空白字符，请上传有效的日志文件。"

    return True, content, None


# ----------------------------------------------------------
# RAG 核心：文本分块
# ----------------------------------------------------------
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    将长文本切分为固定大小的块（重叠滑动窗口）。

    为什么需要重叠？
      如果严格按 500 字切断，可能把一句完整的话（如"PC 寄存器指向
      0xDEADBEEF，这意味着…"）切成两半。重叠 50 字可以保证每个块的
      边界处不会丢失关键语义。

    参数：
      text:       原始文本
      chunk_size: 每块最大字符数（默认 500）
      overlap:    相邻块重叠字符数（默认 50）

    返回：
      字符串列表，每个元素是一个块
    """
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        # 下一块的起点 = 当前终点 - 重叠量
        start = end - overlap

    return chunks


# ----------------------------------------------------------
# RAG 核心：向量化 + 存入 Chroma
# ----------------------------------------------------------
def index_document(content: str) -> int:
    """
    将上传的文档内容：分块 → Embedding → 存入 Chroma 向量数据库。

    步骤详解：
      ① chunk_text()     将长文本切成 500 字的小块
      ② model.encode()   用本地模型把每个块转成 384 维向量
      ③ 删除旧 Collection 并新建（避免新旧文件混在一起）
      ④ collection.add() 将向量 + 原文块 + ID 批量写入 Chroma

    参数：
      content: 上传文件的完整文本

    返回：
      分块数量
    """
    import chromadb

    print(f"\n{'='*60}")
    print(f"[RAG] 开始索引文档，总长度: {len(content)} 字符")
    print(f"{'='*60}")

    # --- 步骤①：分块 ---
    chunks = chunk_text(content)
    print(f"[RAG] 步骤① 文本分块完成 → 共 {len(chunks)} 个块（{CHUNK_SIZE}字/块，重叠{CHUNK_OVERLAP}字）")

    # --- 步骤②：批量 Embedding ---
    model = get_embedding_model()
    print(f"[RAG] 步骤② 正在向量化 {len(chunks)} 个文本块...")
    embeddings = model.encode(chunks, show_progress_bar=False)
    print(f"[RAG] 步骤② 向量化完成 → 形状 {embeddings.shape}（{embeddings.shape[0]}块 × {embeddings.shape[1]}维）")

    # --- 步骤③：重建 Chroma Collection ---
    chroma = get_chroma_client()

    # 如果旧集合存在则删除（避免新旧数据混淆）
    try:
        chroma.delete_collection(CHROMA_COLLECTION_NAME)
        print(f"[RAG] 步骤③ 已删除旧的 Collection: {CHROMA_COLLECTION_NAME}")
    except (ValueError, chromadb.errors.CollectionNotFoundError):
        pass  # 首次运行，没有旧集合

    collection = chroma.create_collection(
        name=CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},  # 用余弦相似度做检索
    )
    print(f"[RAG] 步骤③ 新建 Collection: {CHROMA_COLLECTION_NAME} (距离度量: cosine)")

    # --- 步骤④：批量写入向量 ---
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    collection.add(
        ids=ids,
        embeddings=embeddings.tolist(),   # numpy → list
        documents=chunks,                  # 原文块，检索时可直接返回
    )
    print(f"[RAG] 步骤④ 已写入 {len(chunks)} 条向量到 Chroma")
    print(f"{'='*60}\n")

    return len(chunks)


# ----------------------------------------------------------
# RAG 核心：检索最相关的块
# ----------------------------------------------------------
def retrieve_top_k(query: str, k: int = RETRIEVAL_TOP_K) -> list[str]:
    """
    根据用户问题，从 Chroma 中检索最相关的 k 个文本块。

    步骤详解：
      ① model.encode(query)  将用户问题转为向量
      ② collection.query()   在 Chroma 中做余弦相似度检索
      ③ 返回匹配到的原文块列表

    参数：
      query: 用户输入的提问
      k:     返回几个最相关的结果（默认 3）

    返回：
      文本块列表（按相关度从高到低排列）
    """
    model = get_embedding_model()
    chroma = get_chroma_client()
    collection = chroma.get_collection(CHROMA_COLLECTION_NAME)

    # --- 步骤①：向量化用户问题 ---
    query_embedding = model.encode([query])[0]

    # --- 步骤②：检索 ---
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=k,
    )

    # --- 步骤③：提取文本 ---
    retrieved_docs = results["documents"][0]  # documents[0] 是第一个查询的结果列表
    distances = results.get("distances", [[]])[0]

    # 打印到终端调试
    print(f"\n[RAG 检索] 问题: {query[:100]}...")
    print(f"[RAG 检索] 返回 {len(retrieved_docs)} 个相关块:")
    for i, (doc, dist) in enumerate(zip(retrieved_docs, distances)):
        preview = doc[:120].replace("\n", " ")
        print(f"  [{i+1}] 距离={dist:.4f} | {preview}...")
    print()

    return retrieved_docs


# ----------------------------------------------------------
# RAG 核心：构建带检索上下文的完整 Prompt
# ----------------------------------------------------------
def build_rag_messages(user_query: str) -> list[dict]:
    """
    组装发送给 DeepSeek 的完整消息列表。

    消息结构（四层）：
      ┌──────────────────────────────────────────┐
      │ system(1):  角色设定（嵌入式分析专家）     │
      │ system(2):  检索到的 Top-K 相关文本块     │  ← RAG 注入点
      │ user + assistant:  历史对话               │
      │ user:  本次提问                           │
      └──────────────────────────────────────────┘

    参数：
      user_query: 用户本次输入的提问

    返回：
      完整的 messages 列表
    """
    api_messages = []

    # --- 第 1 层：系统角色 ---
    system_text = (
        "你是一个专业的嵌入式研发 Log 分析助手。"
        "你擅长分析嵌入式系统的死机日志、寄存器 dump、异常堆栈、驱动错误等。"
        "请用中文回答所有问题，对技术术语保留英文原名。"
        "回答要专业、准确、有条理，必要时用分点列出。"
    )
    api_messages.append({"role": "system", "content": system_text})

    # --- 第 2 层：RAG 检索结果（仅在文件已索引时生效） ---
    if st.session_state.indexing_done and st.session_state.file_content:
        relevant_chunks = retrieve_top_k(user_query, k=RETRIEVAL_TOP_K)

        # 将检索到的块拼成一段背景知识文本
        chunks_text = "\n\n---\n\n".join(
            [f"[相关片段 {i+1}]\n{c}" for i, c in enumerate(relevant_chunks)]
        )

        context_text = (
            "以下是从用户上传的日志/文档中检索到的、与当前问题最相关的文本片段。"
            "请优先基于这些内容进行分析和回答：\n\n"
            f"{chunks_text}\n\n"
            "---\n"
            "注意事项：\n"
            "1. 回答时请引用片段中的具体内容来支撑你的判断。\n"
            "2. 如果检索到的片段不足以回答问题，请如实告知，并结合你的专业知识给出推断。\n"
            "3. 如果问题与日志完全无关，请正常回答。"
        )
        api_messages.append({"role": "system", "content": context_text})

    # --- 第 3 层：历史对话 ---
    api_messages.extend(st.session_state.messages)

    return api_messages


def generate_markdown_report() -> str:
    """将当前对话生成一份 Markdown 格式的分析报告"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# 嵌入式 Log 分析报告 (RAG)",
        "",
        f"**生成时间**：{now}  ",
        f"**分析模型**：DeepSeek-Chat + RAG 检索增强  ",
    ]

    if st.session_state.file_name:
        lines.append(f"**分析文件**：`{st.session_state.file_name}`  ")
        lines.append(f"**索引分块**：{st.session_state.chunk_count} 块  ")

    lines.append("")
    lines.append("---")
    lines.append("")

    qa_pairs = []
    current_q = None
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            current_q = msg["content"]
        elif msg["role"] == "assistant" and current_q is not None:
            qa_pairs.append((current_q, msg["content"]))
            current_q = None

    if not qa_pairs:
        lines.append("*（暂无对话内容）*")
    else:
        for i, (q, a) in enumerate(qa_pairs, 1):
            lines.append(f"### Q{i}：{q}")
            lines.append("")
            lines.append(a)
            lines.append("")
            lines.append("---")
            lines.append("")

    return "\n".join(lines)


def print_api_messages(messages: list[dict]):
    """将发送给 API 的消息摘要打印到本地终端，方便调试"""
    print("\n" + "=" * 60)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 📤 发送请求到 DeepSeek（RAG 模式）")
    print("=" * 60)

    for i, msg in enumerate(messages):
        role = msg["role"].upper()
        content = msg["content"]
        display = content[:300].replace("\n", "\\n")
        if len(content) > 300:
            display += f"... [总长 {len(content)} 字符]"
        print(f"\n--- [{i+1}] {role} ---")
        print(display)

    print(f"\n{'=' * 60}\n")

# ============================================================
# 左侧边栏 —— 文件上传 & 管理
# ============================================================
with st.sidebar:
    st.header("📂 文件上传")
    st.caption("仅支持 `.txt` / `.log`，最大 **2 MB**")
    st.caption("上传后自动向量化 → 检索增强分析")

    uploaded_file = st.file_uploader(
        "选择 .txt 或 .log 文件",
        type=["txt", "log"],
        label_visibility="collapsed",
        key="file_uploader",
    )

    # --- 文件校验与读取 ---
    if uploaded_file is not None:
        passed, content, error_msg = validate_and_read_file(uploaded_file)

        if not passed:
            st.error(error_msg)
        else:
            is_same_file = (
                st.session_state.file_name == uploaded_file.name
                and st.session_state.file_content == content
            )
            if not is_same_file:
                # 新文件 → 清空旧对话 + 标记需要重新索引
                st.session_state.messages = []
                st.session_state.file_content = content
                st.session_state.file_name = uploaded_file.name
                st.session_state.indexing_done = False
                st.session_state.chunk_count = 0

    # --- 执行向量化入库 ---
    if st.session_state.file_content and not st.session_state.indexing_done:
        with st.spinner("🔍 正在向量化文档（分块 → Embedding → 入库）..."):
            try:
                chunk_count = index_document(st.session_state.file_content)
                st.session_state.chunk_count = chunk_count
                st.session_state.indexing_done = True
                st.success(f"✅ 索引完成！共 {chunk_count} 个文本块已入库")
            except Exception as e:
                st.error(f"❌ 向量化失败：{e}")
                print(f"[RAG ERROR] {e}")

    # --- 已加载文件信息 ---
    if st.session_state.file_content:
        st.divider()
        st.subheader("📄 已加载文件")
        st.info(f"**{st.session_state.file_name}**")

        lines_list = st.session_state.file_content.splitlines()
        total_lines = len(lines_list)
        total_chars = len(st.session_state.file_content)
        file_size_kb = len(st.session_state.file_content.encode("utf-8")) / 1024
        st.caption(
            f"共 {total_lines} 行 · {total_chars} 字符 · {file_size_kb:.1f} KB"
        )

        # 显示 RAG 索引状态
        if st.session_state.indexing_done:
            st.caption(f"🔍 RAG 就绪 · {st.session_state.chunk_count} 个向量块")
        else:
            st.caption("⏳ 等待索引...")

        with st.expander("📝 文件内容预览（前 15 行）"):
            preview = "\n".join(lines_list[:15])
            st.code(preview, language="text")

        if st.button("🗑️ 清除文件（同时清空对话）", use_container_width=True):
            # 同时删除 Chroma 中的集合
            try:
                chroma = get_chroma_client()
                chroma.delete_collection(CHROMA_COLLECTION_NAME)
            except Exception:
                pass
            st.session_state.file_content = None
            st.session_state.file_name = None
            st.session_state.messages = []
            st.session_state.indexing_done = False
            st.session_state.chunk_count = 0
            st.rerun()

    # --- 快捷操作 ---
    st.divider()
    st.subheader("⚙️ 快捷操作")
    if st.button("🔄 清空对话历史", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    # --- 使用说明 ---
    st.divider()
    st.markdown("""
    **💡 RAG 工作流程**
    1. 上传 `.txt` / `.log`（≤2MB）
    2. 自动分块 → 向量化 → 入库
    3. 提问时自动检索最相关 3 段
    4. AI 结合精准上下文回答

    **🔧 技术栈**
    - 分块：500字/块（重叠50字）
    - 向量模型：multilingual-MiniLM
    - 向量库：Chroma (cosine)
    """)

# ============================================================
# 页面顶部 —— 标签页
# ============================================================
tab_log, tab_about = st.tabs(["📟 日志分析 (RAG)", "ℹ️ 关于工具"])

# ============================================================
# 标签页 1：日志分析
# ============================================================
with tab_log:
    st.title("📟 嵌入式 Log 分析助手 (RAG)")
    st.caption("上传日志 → 自动向量化 → 检索增强分析 → 精准回答")

    if not st.session_state.file_content:
        st.info("👈 **请先在左侧边栏上传 .txt 或 .log 文件**，系统将自动进行 RAG 索引。")
    elif not st.session_state.indexing_done:
        st.info("⏳ 正在等待向量化完成...")

    st.divider()

    # --- 渲染历史对话 ---
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # --- 导出报告按钮 ---
    has_assistant_reply = any(
        m["role"] == "assistant" for m in st.session_state.messages
    )

    if has_assistant_reply:
        st.divider()
        report_md = generate_markdown_report()
        col1, col2, col3 = st.columns([1, 1, 3])
        with col1:
            st.download_button(
                label="📥 导出 Markdown",
                data=report_md,
                file_name=f"rag_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown",
                use_container_width=True,
            )
        with col2:
            st.download_button(
                label="📄 导出纯文本",
                data=report_md,
                file_name=f"rag_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True,
            )

    # --- 用户输入框 ---
    if prompt := st.chat_input(
        "输入你的问题（系统将自动检索最相关的日志片段作为上下文）..."
    ):
        # --- 步骤 1：记录并显示用户消息 ---
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # --- 步骤 2：RAG 检索 + 流式调用 DeepSeek ---
        with st.chat_message("assistant"):
            placeholder = st.empty()
            full_response = ""

            try:
                # 构建带 RAG 检索结果的消息列表
                api_messages = build_rag_messages(prompt)

                # 打印到终端方便调试
                print_api_messages(api_messages)

                stream = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=api_messages,
                    temperature=0.7,
                    max_tokens=2000,
                    stream=True,
                )

                for chunk in stream:
                    if chunk.choices[0].delta.content:
                        full_response += chunk.choices[0].delta.content
                        placeholder.markdown(full_response + "▌")

                placeholder.markdown(full_response)

            except Exception as e:
                placeholder.error(f"❌ API 调用失败：{e}")
                full_response = f"*[错误] 请求失败：{e}*"

        # --- 步骤 3：存入历史 ---
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )
        st.rerun()

# ============================================================
# 标签页 2：关于工具
# ============================================================
with tab_about:
    st.title("ℹ️ 关于工具")

    st.markdown("""
    ## 🧠 嵌入式 Log RAG 智能分析助手

    本工具将 **RAG（检索增强生成）** 引入嵌入式研发流程：

    ```
    上传 Log ──→ 文本分块 ──→ 本地 Embedding ──→ Chroma 向量库
                                                        │
    用户提问 ──→ Embedding 问题 ──→ 检索 Top-3 ──→ 拼入 Prompt ──→ DeepSeek 回答
    ```

    ### 🔍 为什么用 RAG 而不是直接全量塞入？

    | 方式 | 问题 |
    |------|------|
    | 全量塞入 | 大文件超过模型上下文窗口（DeepSeek 支持 64K，但 1MB 日志约 10 万 token） |
    | 全量塞入 | 大量无关内容稀释注意力，回答质量下降 |
    | **RAG 检索** | ✅ 只注入最相关的 3 段，精准 + 高效 |

    ---

    ## 👨‍💻 关于我

    > **5 年嵌入式老兵，正在探索 AI 全栈开发。**

    写过 BSP，调过驱动，追过死机，改过 Linux 内核。
    2026 年开始用 AI 工具武装嵌入式研发流程。

    这个 RAG 版本是我 **AI + 嵌入式** 跨界尝试的重要里程碑。

    ---

    ## 🔧 技术栈

    | 层级 | 技术 |
    |------|------|
    | 前端 | Streamlit (纯 Python) |
    | 大模型 | DeepSeek-Chat (OpenAI 兼容) |
    | 向量库 | Chroma (PersistentClient, cosine) |
    | Embedding | paraphrase-multilingual-MiniLM-L12-v2 |
    | 文本分块 | 500 字/块，50 字重叠滑动窗口 |
    | SDK | OpenAI SDK + sentence-transformers |
    """)

    st.divider()
    st.caption("Built with ❤️ + Streamlit + Chroma + DeepSeek | 2026")
