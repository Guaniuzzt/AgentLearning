"""网页搜索工具 —— 基于 DeepSeek Search API。

文档: https://api-docs.deepseek.com/zh-cn/guides/search
Endpoint: POST https://api.deepseek.com/v1/tools/search
鉴权: 与 DeepSeek 模型 API 相同的 Bearer Token（DEEPSEEK_API_KEY）
计费: 5 元 / 1000 次成功请求；结果为空或请求失败不计费。

要点:
- 同步 `search()` 与异步 `asearch()` 双入口，供 ReAct Agent 的线程池路径/协程路径共用。
- 结果带进程内 TTL 缓存，避免并发/多轮搜索把同一 query 反复计费。
- HTTP 状态码翻译成 ToolError：429/5xx 可重试，401/402/4xx 直接降级不重试。

用法:
    export DEEPSEEK_API_KEY=sk-xxx
    uv run python -m uv_and_httpx.search "英伟达最新的GPU型号是什么"
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from typing import Any

import httpx
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field

from uv_and_httpx.ToolExecutor import ToolError

load_dotenv()

BASE_URL = os.getenv("DEEPSEEK_BASE_URL") or "https://api.deepseek.com"
SEARCH_PATH = "/v1/tools/search"
HTTP_TIMEOUT = float(os.getenv("SEARCH_TIMEOUT") or 20)
DEFAULT_COUNT = 5
TOP_N = 3  # 喂给模型的条数
CACHE_TTL_SECONDS = float(os.getenv("SEARCH_CACHE_TTL") or 300)
CACHE_MAX_ENTRIES = 128
GOOD_SCORE = 1.5  # rerank_score >= 1.5 视为优质结果
MAX_BODY_CHARS = 600  # 单条摘要的上下文字数上限，避免挤爆 prompt


# --------------------------------------------------------------------------- #
# Pydantic 模型：请求 / 响应
# --------------------------------------------------------------------------- #
class SearchRequest(BaseModel):
    """Search 接口的请求体（字段上限严格按官方文档声明）。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    query: str = Field(min_length=1, max_length=128, description="搜索词，支持 site:/lang:/after: 等语法")
    count: int = Field(default=DEFAULT_COUNT, ge=1, le=50, description="返回条数")
    ori_question: str | None = Field(default=None, max_length=128, description="触发搜索的用户原始问题")
    need_content: bool = Field(default=False, description="是否返回网页正文（很占上下文，默认关）")

    def payload(self) -> dict[str, Any]:
        return self.model_dump(exclude_none=True)


class SearchResultItem(BaseModel):
    """单条搜索结果。字段缺失时给保守默认值，避免一条脏数据炸掉整次搜索。"""

    model_config = ConfigDict(extra="ignore")

    title: str = ""
    url: str = ""
    snippet: str = ""
    summary: str = ""
    content: str = ""
    site_name: str | None = None
    date: str | None = None
    authority: int = -1
    rerank_score: float = 0.0


class SearchResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    results: list[SearchResultItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# 缓存：同一 query 在 TTL 内只计费一次
# --------------------------------------------------------------------------- #
_cache: dict[str, tuple[float, str]] = {}
_cache_lock = threading.Lock()


def _cache_key(request: SearchRequest) -> str:
    return f"{request.query}|{request.count}|{request.need_content}"


def _cache_get(key: str) -> str | None:
    with _cache_lock:
        hit = _cache.get(key)
        if hit is None:
            return None
        created_at, value = hit
        if time.monotonic() - created_at > CACHE_TTL_SECONDS:
            _cache.pop(key, None)
            return None
        return value


def _cache_put(key: str, value: str) -> None:
    with _cache_lock:
        if len(_cache) >= CACHE_MAX_ENTRIES:
            _cache.clear()  # demo 级别的容量控制：满了一键清空
        _cache[key] = (time.monotonic(), value)


def clear_cache() -> None:
    with _cache_lock:
        _cache.clear()


# --------------------------------------------------------------------------- #
# 公共辅助
# --------------------------------------------------------------------------- #
def _api_key() -> str:
    key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not key:
        raise ToolError("未配置 DEEPSEEK_API_KEY，无法调用 DeepSeek Search。", retryable=False)
    return key


def _headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {_api_key()}", "Content-Type": "application/json"}


def _error_message(body: str) -> str:
    """从 OpenAI 风格的错误体里取 message，取不到就退回原文。"""
    try:
        data = json.loads(body)
    except ValueError:
        return body[:200]
    if isinstance(data, dict):
        error = data.get("error")
        if isinstance(error, dict) and error.get("message"):
            return str(error["message"])
    return body[:200]


def _raise_for_status(status: int, body: str) -> None:
    """HTTP 状态码 -> 可重试 / 不可重试的 ToolError。"""
    message = _error_message(body)
    if status in (408, 429) or status >= 500:
        # 限流、超时、服务端故障：值得重试
        raise ToolError(f"DeepSeek Search 暂时不可用（{status}: {message}）", retryable=True)
    if status == 402:
        raise ToolError(f"DeepSeek Search 余额不足（402: {message}），请充值后再试", retryable=False)
    if status in (401, 403):
        raise ToolError(f"DeepSeek Search 鉴权失败（{status}: {message}）", retryable=False)
    raise ToolError(f"DeepSeek Search 请求不合法（{status}: {message}）", retryable=False)


def _parse_payload(payload: Any) -> SearchResponse:
    try:
        return SearchResponse.model_validate(payload)
    except Exception as e:  # noqa: BLE001 - 结构不对时统一当作可重试的服务端异常
        raise ToolError(f"DeepSeek Search 响应结构异常: {e}", retryable=True) from e


def _format_results(query: str, response: SearchResponse) -> str:
    """把结构化结果压成一段给模型看的文本：优先 summary，退化到 snippet。"""
    if not response.results:
        return f"对不起，没有找到关于 '{query}' 的信息。"

    picked = [item for item in response.results if item.rerank_score >= GOOD_SCORE][:TOP_N]
    if not picked:
        picked = response.results[:TOP_N]

    blocks: list[str] = []
    for index, item in enumerate(picked, 1):
        body = (item.summary or item.snippet or item.content).replace("\n", " ").strip()
        if len(body) > MAX_BODY_CHARS:
            body = body[:MAX_BODY_CHARS] + "…"
        meta = " / ".join(x for x in [item.site_name or "", (item.date or "")[:10]] if x)
        source = f"来源: {item.url}" + (f"（{meta}）" if meta else "")
        blocks.append(f"[{index}] {item.title}\n{body}\n{source}")
    return "\n\n".join(blocks)


# --------------------------------------------------------------------------- #
# 同步 / 异步入口
# --------------------------------------------------------------------------- #
def search(query: str, *, count: int = DEFAULT_COUNT, ori_question: str | None = None) -> str:
    """同步版搜索：供同步工具、或异步链路里丢进线程池使用。"""
    request = SearchRequest(query=query, count=count, ori_question=ori_question)
    key = _cache_key(request)
    cached = _cache_get(key)
    if cached is not None:
        print(f"🔍 [DeepSeek Search] 命中缓存: {query}")
        return cached

    print(f"🔍 正在执行 [DeepSeek Search] 网页搜索: {query}")
    try:
        with httpx.Client(base_url=BASE_URL, timeout=HTTP_TIMEOUT) as client:
            response = client.post(SEARCH_PATH, json=request.payload(), headers=_headers())
    except httpx.HTTPError as e:
        raise ToolError(f"DeepSeek Search 网络错误: {type(e).__name__}: {e}", retryable=True) from e

    if response.status_code != 200:
        _raise_for_status(response.status_code, response.text)

    result = _format_results(query, _parse_payload(response.json()))
    _cache_put(key, result)
    return result


async def asearch(query: str, *, count: int = DEFAULT_COUNT, ori_question: str | None = None) -> str:
    """异步版搜索：直接 await，不阻塞事件循环。"""
    request = SearchRequest(query=query, count=count, ori_question=ori_question)
    key = _cache_key(request)
    cached = _cache_get(key)
    if cached is not None:
        print(f"🔍 [DeepSeek Search] 命中缓存: {query}")
        return cached

    print(f"🔍 正在执行 [DeepSeek Search] 网页搜索: {query}")
    try:
        async with httpx.AsyncClient(base_url=BASE_URL, timeout=HTTP_TIMEOUT) as client:
            response = await client.post(SEARCH_PATH, json=request.payload(), headers=_headers())
    except httpx.HTTPError as e:
        raise ToolError(f"DeepSeek Search 网络错误: {type(e).__name__}: {e}", retryable=True) from e

    if response.status_code != 200:
        _raise_for_status(response.status_code, response.text)

    result = _format_results(query, _parse_payload(response.json()))
    _cache_put(key, result)
    return result


if __name__ == "__main__":
    print(asyncio.run(asearch("英伟达最新的GPU型号是什么")))
