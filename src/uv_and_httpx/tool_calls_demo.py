"""DeepSeek Tool Calls 官方示例（openai SDK 版）。

文档: https://api-docs.deepseek.com/zh-cn/guides/tool_calls/

用法:
    export DEEPSEEK_API_KEY=sk-xxx
    uv run python -m uv_and_httpx.tool_calls_demo
"""

import os
from typing import Any

from openai import OpenAI
from openai.types.chat import ChatCompletionToolParam
from openai.types.chat.completion_create_params import CompletionCreateParamsNonStreaming

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY"),
    base_url="https://api.deepseek.com",
)

tools: list[ChatCompletionToolParam] = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather of a location, the user should supply a location first.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                },
                "required": ["location"],
            },
        },
    },
]


def send_messages(messages: list[Any]):
    # client.chat.completions.create(...) 就是 POST /chat/completions
    params: CompletionCreateParamsNonStreaming = {
        "model": "deepseek-flash",
        "messages": messages,
        "tools": tools,
    }
    # **params 把字典的每个 key 展开成关键字参数，等价于一个个手写
    response = client.chat.completions.create(
        **params,
        # thinking 不是 OpenAI 标准字段，靠 extra_body 透传给 DeepSeek
        extra_body={"thinking": {"type": "disabled"}},
    )

    # response.choices[0].message 的拆解
    choices = response.choices
    first_choice = choices[0]

    # 把服务端返回的完整内容打出来（含 finish_reason / tool_calls / usage）
    print("\n========== raw response ==========")
    print(response.model_dump_json(indent=2))
    print("==================================")

    return first_choice.message


messages: list[Any] = [
    {"role": "user", "content": "How's the weather in Hangzhou, Zhejiang?"}
]

print(f"User>\t {messages[0]['content']}")
message = send_messages(messages)

assert message.tool_calls, "模型这次没有发起工具调用"
print(f"Model>\t {message.content}")
tool = message.tool_calls[0]
messages.append(message)
messages.append({
    "role": "tool",
    "tool_call_id": tool.id,
    "content": "24℃",  # 这里就是本地执行 get_weather 的结果，示例直接写死
})

message = send_messages(messages)
print(f"Model>\t {message.content}")
