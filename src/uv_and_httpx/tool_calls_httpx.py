"""DeepSeek Tool Calls demo —— 用 httpx 直接调 /chat/completions，不经过 openai SDK。

文档: https://api-docs.deepseek.com/zh-cn/guides/tool_calls/

用法:
    export DEEPSEEK_API_KEY=sk-xxx
    uv run python -m uv_and_httpx.tool_calls_httpx
    uv run python -m uv_and_httpx.tool_calls_httpx --strict "北京天气怎么样"
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from typing import Any

import httpx

BASE_URL = "https://api.deepseek.com"
BASE_URL_BETA = "https://api.deepseek.com/beta"  # strict 模式（Beta）必须走这个域名
MODEL = "deepseek-flash"

TOOLS: list[dict[str, Any]] = [
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


def build_tools(strict: bool) -> list[dict[str, Any]]:
    """strict 模式的三条硬性要求：走 beta 域名、每个 function 都设 strict: true、
    object 全部属性 required 且 additionalProperties: false。"""
    tools = copy.deepcopy(TOOLS)
    if strict:
        for tool in tools:
            fn = tool["function"]
            fn["strict"] = True
            fn["parameters"]["additionalProperties"] = False
    return tools


# 本地"执行"工具。真实项目里换成真正的天气 API 调用。
MOCK_WEATHER = {"Hangzhou": "24℃", "杭州": "24℃", "Shanghai": "27℃", "上海": "27℃"}


def get_weather(location: str) -> str:
    city = location.split(",")[0].strip()  # "Hangzhou, Zhejiang" -> "Hangzhou"
    if city in MOCK_WEATHER:
        return MOCK_WEATHER[city]
    return f"没有 {location} 的天气数据（demo 未接入真实天气 API）"


def chat(
    client: httpx.Client,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": MODEL,
        "messages": messages,
        "tools": tools,
        # 关闭思考模式：思考模式下多轮工具调用还要额外回传 reasoning_content
        "thinking": {"type": "disabled"},
    }
    resp = client.post("/chat/completions", json=payload)
    resp.raise_for_status()
    return resp.json()["choices"][0]


def main() -> None:
    parser = argparse.ArgumentParser(description="DeepSeek Tool Calls demo (httpx)")
    parser.add_argument(
        "question",
        nargs="?",
        default="How's the weather in Hangzhou, Zhejiang?",
    )
    parser.add_argument("--strict", action="store_true", help="启用 strict 模式（Beta 端点）")
    args = parser.parse_args()

    api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("请先设置环境变量 DEEPSEEK_API_KEY")

    tools = build_tools(args.strict)
    messages: list[dict[str, Any]] = [{"role": "user", "content": args.question}]
    print(f"User>\t {args.question}")

    with httpx.Client(
        base_url=BASE_URL_BETA if args.strict else BASE_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=httpx.Timeout(60.0),
    ) as client:
        # 第 1 次请求：期望模型返回 tool_calls
        choice = chat(client, messages, tools)
        message = choice["message"]
        print(f"finish_reason>\t {choice['finish_reason']}")

        if choice["finish_reason"] != "tool_calls":
            print(f"Model>\t {message.get('content')}")
            return

        # 关键顺序：先把带 tool_calls 的 assistant 消息放回历史，再回传 tool 结果
        messages.append({
            "role": "assistant",
            "content": message.get("content") or "",
            "tool_calls": message["tool_calls"],
        })

        for call in message["tool_calls"]:
            name = call["function"]["name"]
            call_args = json.loads(call["function"].get("arguments") or "{}")
            result = get_weather(**call_args)  # 模型只给参数，执行在本地
            print(f"tool>\t {name}({call_args}) -> {result}")
            messages.append({
                "role": "tool",
                "tool_call_id": call["id"],
                "content": result,
            })

        # 第 2 次请求：模型拿到工具结果后给出自然语言回答
        choice = chat(client, messages, tools)
        print(f"finish_reason>\t {choice['finish_reason']}")
        print(f"Model>\t {choice['message'].get('content')}")


if __name__ == "__main__":
    main()
