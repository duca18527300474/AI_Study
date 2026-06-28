"""
web.py — 嵌入式研发 Log/文档智能分析助手
==========================================
基于 Streamlit + DeepSeek API 的聊天网页。
支持上传 .txt / .log 文件，结合文件内容进行智能分析。
具备文件大小/格式校验 + 导出分析报告功能。
"""

import os
import sys
from datetime import datetime

from dotenv import load_dotenv

sys.stdout.reconfigure(encoding="utf-8")

# 从 .env 文件加载环境变量
load_dotenv()

import streamlit as st
from openai import OpenAI

# ============================================================
# 页面基础配置
# ============================================================
st.set_page_config(
    page_title="嵌入式 Log 分析助手",
    page_icon="📟",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ============================================================
# 常量
# ============================================================
MAX_FILE_SIZE_BYTES = 2 * 1024 * 1024  # 2MB

# ============================================================
# DeepSeek 客户端（缓存复用）
# ============================================================
@st.cache_resource
def get_client():
    return OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
    )

client = get_client()

# ============================================================
# 会话状态初始化
# ============================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "file_content" not in st.session_state:
    st.session_state.file_content = None

if "file_name" not in st.session_state:
    st.session_state.file_name = None

# ============================================================
# 工具函数
# ============================================================

def validate_and_read_file(uploaded_file) -> tuple[bool, str | None, str | None]:
    """
    校验上传文件的大小、格式、内容是否合法。
    返回 (是否通过, 文件内容|None, 错误消息|None)
    """
    raw_bytes = uploaded_file.getvalue()
    file_size_mb = len(raw_bytes) / (1024 * 1024)
    filename = uploaded_file.name

    # --- 校验 1：文件扩展名 ---
    if not filename.lower().endswith((".txt", ".log")):
        return False, None, (
            f"❌ 不支持的文件格式 **`{filename}`**，仅允许 `.txt` 或 `.log` 文件。"
        )

    # --- 校验 2：文件大小上限 ---
    if len(raw_bytes) > MAX_FILE_SIZE_BYTES:
        return False, None, (
            f"❌ 文件过大（{file_size_mb:.1f} MB），已超过 **2 MB** 上限。"
            f"请裁剪日志后重新上传。"
        )

    # --- 校验 3：空文件 ---
    if len(raw_bytes) == 0:
        return False, None, "❌ 文件内容为空，请上传有效的日志文件。"

    # --- 校验 4：编码解码 ---
    try:
        content = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        try:
            content = raw_bytes.decode("gbk", errors="replace")
        except Exception:
            return False, None, "❌ 无法识别文件编码，请确认文件为文本格式（UTF-8 / GBK）。"

    # --- 校验 5：内容非空白 ---
    if not content.strip():
        return False, None, "❌ 文件内容仅含空白字符，请上传有效的日志文件。"

    return True, content, None


def build_api_messages():
    """将系统提示词 + 文件背景知识 + 对话历史 组装成 API 所需的消息列表"""
    api_messages = []

    # --- 第 1 层：系统角色定义 ---
    system_text = (
        "你是一个专业的嵌入式研发 Log 分析助手。"
        "你擅长分析嵌入式系统的死机日志、寄存器 dump、异常堆栈、驱动错误等。"
        "请用中文回答所有问题，对技术术语保留英文原名。"
        "回答要专业、准确、有条理，必要时用分点列出。"
    )
    api_messages.append({"role": "system", "content": system_text})

    # --- 第 2 层：上传文件的背景知识 ---
    if st.session_state.file_content:
        context_text = (
            "以下是用户上传的日志/文档文件内容，"
            "请将其作为分析依据，结合这些内容回答用户的后续问题：\n\n"
            "```\n"
            f"{st.session_state.file_content}\n"
            "```\n\n"
            "注意：如果用户的问题与文件内容无关，请正常回答；"
            "如果问题涉及日志分析，必须引用文件中的具体内容来支撑你的判断。"
        )
        api_messages.append({"role": "system", "content": context_text})

    # --- 第 3 层：对话历史 ---
    api_messages.extend(st.session_state.messages)

    return api_messages


def generate_markdown_report() -> str:
    """将当前对话生成一份 Markdown 格式的分析报告"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# 嵌入式 Log 分析报告",
        "",
        f"**生成时间**：{now}  ",
        f"**分析模型**：DeepSeek-Chat  ",
    ]

    if st.session_state.file_name:
        lines.append(f"**分析文件**：`{st.session_state.file_name}`  ")

    lines.append("")
    lines.append("---")
    lines.append("")

    # 按 Q&A 对输出对话
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


def print_messages_to_terminal(messages: list[dict]):
    """将发送给 API 的消息列表打印到本地终端，方便调试观察数据流"""
    print("\n" + "=" * 60)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 📤 发送请求到 DeepSeek")
    print("=" * 60)

    for i, msg in enumerate(messages):
        role = msg["role"].upper()
        content = msg["content"]

        # 文件内容可能很长，截断显示
        display = content
        if role == "SYSTEM" and "以下是用户上传的日志" in content:
            # 只显示提示语 + 前 200 字符
            preview = content[:200].replace("\n", "\\n")
            display = f"{preview}... [总长度: {len(content)} 字符]"

        print(f"\n--- [{i+1}] {role} ---")
        print(display[:500])  # 每条消息最多打印 500 字符

    print(f"\n{'=' * 60}\n")

# ============================================================
# 左侧边栏 —— 文件上传 & 管理
# ============================================================
with st.sidebar:
    st.header("📂 文件上传")
    st.caption("仅支持 `.txt` / `.log`，最大 **2 MB**")

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
                st.session_state.messages = []
                st.session_state.file_content = content
                st.session_state.file_name = uploaded_file.name

    # --- 已加载文件信息 ---
    if st.session_state.file_content:
        st.divider()
        st.subheader("📄 已加载文件")
        st.info(f"**{st.session_state.file_name}**")

        lines = st.session_state.file_content.splitlines()
        total_lines = len(lines)
        total_chars = len(st.session_state.file_content)
        file_size_kb = len(st.session_state.file_content.encode("utf-8")) / 1024
        st.caption(f"共 {total_lines} 行 · {total_chars} 字符 · {file_size_kb:.1f} KB")

        with st.expander("📝 文件内容预览（前 15 行）"):
            preview = "\n".join(lines[:15])
            st.code(preview, language="text")

        if st.button("🗑️ 清除文件（同时清空对话）", use_container_width=True):
            st.session_state.file_content = None
            st.session_state.file_name = None
            st.session_state.messages = []
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
    **💡 使用步骤**
    1. 上传 `.txt` / `.log` 文件（≤2MB）
    2. 在聊天框提问（如"分析死机原因"）
    3. AI 结合日志内容精准回答
    4. 点击下方按钮导出报告

    **⚠️ 注意**
    上传新文件会清空对话历史
    """)

# ============================================================
# 页面顶部 —— 标签页
# ============================================================
tab_log, tab_about = st.tabs(["📟 日志分析", "ℹ️ 关于工具"])

# ============================================================
# 标签页 1：日志分析（核心功能）
# ============================================================
with tab_log:
    st.title("📟 嵌入式 Log 分析助手")
    st.caption("上传日志 → 提问 → AI 结合上下文精准分析 → 导出报告")

    if not st.session_state.file_content:
        st.info("👈 **请先在左侧边栏上传 .txt 或 .log 文件**，然后开始提问。")

    st.divider()

    # --- 渲染历史对话 ---
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # --- 导出报告按钮（有对话内容时才显示） ---
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
                file_name=f"log_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown",
                use_container_width=True,
            )
        with col2:
            st.download_button(
                label="📄 导出纯文本",
                data=report_md,
                file_name=f"log_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True,
            )

    # --- 用户输入框 ---
    if prompt := st.chat_input(
        "输入你的问题（如：分析这段日志里的死机原因）..."
    ):
        # --- 步骤 1：记录并显示用户消息 ---
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # --- 步骤 2：流式调用 DeepSeek ---
        with st.chat_message("assistant"):
            placeholder = st.empty()
            full_response = ""

            try:
                # 🔍 将完整请求消息打印到终端，方便调试
                api_messages = build_api_messages()
                print_messages_to_terminal(api_messages)

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

        # 新对话产生后自动刷新
        st.rerun()

# ============================================================
# 标签页 2：关于工具（个人介绍）
# ============================================================
with tab_about:
    st.title("ℹ️ 关于工具")

    # --- 工具简介 ---
    st.markdown("""
    ## 🛠️ 嵌入式 Log 智能分析助手

    本工具诞生于一个朴素的想法：**嵌入式工程师不应该在茫茫日志里人肉搜索**。

    把死机 dump、寄存器快照、驱动 log 扔进来，大模型帮你：
    - 🔍 快速定位异常点
    - 📖 翻译晦涩的寄存器含义
    - 💡 推断可能的死机原因
    - 📝 一键导出分析报告

    ---

    ## 👨‍💻 关于我

    > **5 年嵌入式老兵，正在探索 AI 全栈开发。**

    写过 BSP，调过驱动，追过死机，改过 Linux 内核。
    2026 年开始用 AI 工具武装嵌入式研发流程 ——
    从命令行脚本到 Streamlit 网页，从单轮问答到上下文分析，
    一步步把想法变成能跑的产品。

    这个项目是我 **AI + 嵌入式** 跨界尝试的起点。

    ---

    ## 🔧 技术栈

    | 层级 | 技术 |
    |------|------|
    | 前端 | Streamlit (纯 Python) |
    | 大模型 | DeepSeek-Chat (OpenAI 兼容) |
    | SDK | OpenAI Python SDK |
    | 部署 | 本地 `streamlit run` |

    ---

    ## 📬 联系与反馈

    如果这个工具帮到了你，或者有 Bug / 新需求，
    欢迎提 Issue 或 PR 👇

    **仓库地址**：`https://github.com/yourname/log-analyzer`
    """)

    st.divider()
    st.caption("Built with ❤️ + Streamlit + DeepSeek | 2026")
