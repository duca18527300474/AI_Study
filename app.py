"""
app.py — 使用 OpenAI SDK 调用 DeepSeek API 进行单轮对话
======================================================
DeepSeek API 完全兼容 OpenAI 的接口格式，因此可以直接使用
OpenAI 官方 Python SDK 来调用。

使用前请先获取 API Key：
  1. 访问 https://platform.deepseek.com
  2. 注册/登录后，在"API Keys"页面创建密钥
  3. 在项目根目录创建 .env 文件，写入：
     DEEPSEEK_API_KEY=你的密钥

依赖：
  openai / python-dotenv

运行：
  python app.py
"""

import os
import sys

from dotenv import load_dotenv

# -------------------------------------------------------
# 0. 初始化环境
# -------------------------------------------------------
# 解决 Windows 终端 GBK 编码无法输出中文/emoji 的问题
sys.stdout.reconfigure(encoding="utf-8")

# 从项目根目录的 .env 文件加载环境变量（如 DEEPSEEK_API_KEY）
# 这样密钥不会硬编码在代码中，也不会被 git 提交
load_dotenv()

from openai import OpenAI

# -------------------------------------------------------
# 1. 配置客户端
# -------------------------------------------------------
# DeepSeek 兼容 OpenAI API 格式，只需指定 base_url 和 api_key
# base_url: 指向 DeepSeek 的 API 网关
# api_key:  从环境变量读取（由 .env 文件注入），安全且便于切换
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

# -------------------------------------------------------
# 2. 发送单轮对话请求
# -------------------------------------------------------
# Chat Completions API —— 这是 OpenAI 定义的标准对话接口，
# DeepSeek、通义千问、Moonshot 等国产模型全部兼容此格式。
#
# 参数说明：
#   model:       使用的模型名称，deepseek-chat 是 DeepSeek 的通用对话模型
#   messages:    对话消息列表，必须包含 role 和 content 两个字段
#     - role: system  → 系统提示词，定义 AI 的行为和风格
#     - role: user    → 用户发送的消息
#     - role: assistant → AI 的回复（多轮对话时使用）
#   temperature: 控制输出的随机性，范围 0~2
#     - 0.0 → 确定性输出，适合代码/翻译
#     - 0.7 → 适中创意，适合通用对话（推荐默认值）
#     - 1.5 → 高度随机，适合创意写作
#   max_tokens:  限制 AI 回复的最大 token 数量（1 token ≈ 0.5-1 个汉字）
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {
            "role": "system",
            "content": "你是一个乐于助人的AI助手，请用中文回答所有问题。",
        },
        {
            "role": "user",
            "content": "你好！请用一句话介绍什么是机器学习。",
        },
    ],
    temperature=0.7,
    max_tokens=500,
)

# -------------------------------------------------------
# 3. 打印结果
# -------------------------------------------------------
# response 对象的结构：
#   response.choices[0].message.content  → AI 的回复文本
#   response.usage.prompt_tokens         → 输入消耗的 token 数
#   response.usage.completion_tokens     → 输出消耗的 token 数
#   response.usage.total_tokens          → 总 token 数（计费依据）
print("=" * 50)
print("[DeepSeek 回复]")
print("=" * 50)
print(response.choices[0].message.content)
print("=" * 50)

# 额外信息：查看 Token 用量
# 这对于控制成本和了解 API 消耗非常有帮助
usage = response.usage
print(f"\n[Token 用量]")
print(f"   - 输入 tokens: {usage.prompt_tokens}")
print(f"   - 输出 tokens: {usage.completion_tokens}")
print(f"   - 总计 tokens: {usage.total_tokens}")
