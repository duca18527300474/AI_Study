"""
app.py — 使用 OpenAI SDK 调用 DeepSeek API 进行单轮对话
======================================================
DeepSeek API 完全兼容 OpenAI 的接口格式，因此可以直接使用
OpenAI 官方 Python SDK 来调用。

使用前请先获取 API Key：
  1. 访问 https://platform.deepseek.com
  2. 注册/登录后，在"API Keys"页面创建密钥
  3. 将密钥设为环境变量 DEEPSEEK_API_KEY
     (Windows 终端: set DEEPSEEK_API_KEY=你的密钥)
"""

import os
import sys

from dotenv import load_dotenv

# 解决 Windows 终端 GBK 编码无法输出 emoji 的问题
sys.stdout.reconfigure(encoding="utf-8")

# 从 .env 文件加载环境变量
load_dotenv()

from openai import OpenAI

# -------------------------------------------------------
# 1. 配置客户端
# -------------------------------------------------------
# DeepSeek 兼容 OpenAI API 格式，只需指定 base_url 和 api_key
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",         # DeepSeek API 地址
)

# -------------------------------------------------------
# 2. 发送单轮对话请求
# -------------------------------------------------------
# 使用 Chat Completions API（与 OpenAI 完全一致的调用方式）
response = client.chat.completions.create(
    model="deepseek-chat",                       # DeepSeek 对话模型
    messages=[
        {"role": "system", "content": "你是一个乐于助人的AI助手，请用中文回答所有问题。"},
        {"role": "user",   "content": "你好！请用一句话介绍什么是机器学习。"},
    ],
    temperature=0.7,                             # 控制随机性 (0-2, 越高越有创意)
    max_tokens=500,                              # 限制回复最大长度
)

# -------------------------------------------------------
# 3. 打印结果
# -------------------------------------------------------
# response.choices[0].message.content 包含模型的回复文本
print("=" * 50)
print("[DeepSeek 回复]")
print("=" * 50)
print(response.choices[0].message.content)
print("=" * 50)

# 额外信息：查看 Token 用量（方便了解花费）
usage = response.usage
print(f"\n[Token 用量]")
print(f"   - 输入 tokens: {usage.prompt_tokens}")
print(f"   - 输出 tokens: {usage.completion_tokens}")
print(f"   - 总计 tokens: {usage.total_tokens}")
