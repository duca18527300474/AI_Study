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
            # 校验失败 → 显示友好错误提示
            st.error(error_msg)
        else:
            # 校验通过 → 检测是否为新文件（避免重复清空对话）
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
# 主页面标题
# ============================================================
st.title("📟 嵌入式 Log 分析助手")
st.caption("上传日志 → 提问 → AI 结合上下文精准分析 → 导出报告")

if not st.session_state.file_content:
    st.info("👈 **请先在左侧边栏上传 .txt 或 .log 文件**，然后开始提问。")

st.divider()

# ============================================================
# 渲染历史对话
# ============================================================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ============================================================
# 导出报告按钮（有对话内容时才显示）
# ============================================================
has_assistant_reply = any(m["role"] == "assistant" for m in st.session_state.messages)

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
        # 也提供一个纯文本版本（去掉 Markdown 标记的简化版）
        report_txt = report_md.replace("# ", "").replace("**", "").replace("`", "").replace("---", "---")
        st.download_button(
            label="📄 导出纯文本",
            data=report_md,
            file_name=f"log_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True,
        )

# ============================================================
# 用户输入框
# ============================================================
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
            stream = client.chat.completions.create(
                model="deepseek-chat",
                messages=build_api_messages(),
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
    st.session_state.messages.append({"role": "assistant", "content": full_response})

    # 新对话产生后自动刷新，让导出按钮出现
    st.rerun()
