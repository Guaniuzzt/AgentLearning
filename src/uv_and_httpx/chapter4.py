"""HelloAgentsLLM —— 兼容 OpenAI 接口的流式客户端，默认指向 DeepSeek。

文档: https://api-docs.deepseek.com/zh-cn/guides/tool_calls/

用法:
    export DEEPSEEK_API_KEY=sk-xxx
    uv run python -m uv_and_httpx.chapter4
"""

import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

# 加载 .env 文件中的环境变量
load_dotenv()

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-flash"


class HelloAgentsLLM:
    """
    为本书 "Hello Agents" 定制的 LLM 客户端。
    它用于调用任何兼容 OpenAI 接口的服务（默认 DeepSeek），并默认使用流式响应。
    """

    model: str
    client: OpenAI
    thinking: bool
    extra_body: dict[str, Any]

    def __init__(
        self,
        model: str | None = None,
        apiKey: str | None = None,
        baseUrl: str | None = None,
        timeout: int | None = None,
        thinking: bool = False,
    ) -> None:
        """
        初始化客户端。优先使用传入参数，其次环境变量，最后回落到 DeepSeek 默认值。
        """
        model = (
            model
            or os.getenv("LLM_MODEL_ID")
            or os.getenv("DEEPSEEK_MODEL")
            or DEFAULT_MODEL
        )
        apiKey = (
            apiKey
            or os.getenv("LLM_API_KEY")
            or os.getenv("DEEPSEEK_API_KEY")
            or os.getenv("OPENAI_API_KEY")
        )
        baseUrl = baseUrl or os.getenv("LLM_BASE_URL") or DEFAULT_BASE_URL
        timeout = timeout or int(os.getenv("LLM_TIMEOUT") or 60)

        # model / baseUrl 都有默认值，只有密钥是硬要求
        if not apiKey:
            raise ValueError("未找到 API 密钥，请设置 DEEPSEEK_API_KEY 或 LLM_API_KEY。")

        # 类型已在类体里声明，这里不要再写注解，否则会触发 reportRedeclaration
        self.model = model
        self.thinking = thinking
        # thinking 不是 OpenAI 标准字段，靠 extra_body 透传给 DeepSeek
        self.extra_body = {} if thinking else {"thinking": {"type": "disabled"}}
        self.client = OpenAI(api_key=apiKey, base_url=baseUrl, timeout=timeout)

    def think(
        self,
        messages: list[ChatCompletionMessageParam],
        temperature: float = 0,
    ) -> str | None:
        """
        调用大语言模型进行思考，并返回其响应。
        """
        print(f"🧠 正在调用 {self.model} 模型...")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                stream=True,
                extra_body=self.extra_body,
            )

            # 处理流式响应
            print("✅ 大语言模型响应成功:")
            collected_content: list[str] = []
            for chunk in response:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                # thinking 模式下这里还会有 reasoning_content，本 demo 只取正文
                content = delta.content or ""
                print(content, end="", flush=True)
                collected_content.append(content)
            print()  # 在流式输出结束后换行
            return "".join(collected_content)

        except Exception as e:
            print(f"❌ 调用LLM API时发生错误: {e}")
            return None


# --- 客户端使用示例 ---
if __name__ == "__main__":
    try:
        llmClient = HelloAgentsLLM()

        exampleMessages: list[ChatCompletionMessageParam] = [
            {
                "role": "system",
                "content": "You are a helpful assistant that writes Python code.",
            },
            {"role": "user", "content": "写一个快速排序算法"},
        ]

        print("--- 调用LLM ---")
        responseText = llmClient.think(exampleMessages)
        if responseText:
            print("\n\n--- 完整模型响应 ---")
            print(responseText)

    except ValueError as e:
        print(e)
