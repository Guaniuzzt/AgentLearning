"""异步版 ReAct Agent。

整体结构（自顶向下看调用关系）：
    arun()                       主循环：拼提示 → 问模型 → 解析 → 并发执行 → 写回历史
    ├─ _build_prompt()           把工具描述 / 问题 / history 填进 REACT_PROMPT_TEMPLATE
    ├─ _think()                  异步调模型，外面套 asyncio.timeout；失败或超时返回 None
    ├─ parse_response()          用 Pydantic 把模型文本解析成 ReActResponse（可含多个 Action）
    ├─ _execute_action()         先抢 Semaphore 名额，再交给 _execute_with_retry()
    │   └─ _execute_with_retry() 参数校验 → 超时包装 → 重试 → 降级成一条 ToolObservation
    │       └─ _call_tool()      协程直接 await；同步函数丢线程池，避免卡住事件循环
    └─ _degrade()                兜底：用已成功的观察拼出"不完整但可用"的答案

相对同步版本的改造点：
1. 工具调用改为异步：协程工具直接 await，同步阻塞工具丢进线程池，不阻塞事件循环。
2. 同一轮里多个相互独立的 Action 用 asyncio.gather 并发执行。
3. 用 asyncio.Semaphore 限制同时在跑的工具数量。
4. 每次模型调用 / 工具调用都套 asyncio.timeout；超时走重试与降级，取消信号（CancelledError）向上传播。
5. 工具参数、模型响应、Agent 配置都用 Pydantic 模型描述与校验。
6. 参数非法时在进入工具执行逻辑之前就被拦截；工具失败会重试，最终降级为一条 Observation 让推理继续。

线程模型：所有代码跑在单个事件循环里，没有共享可变状态的并发写；
self.history / self.observations 只在 arun() 里串行写入（gather 返回后才统一写），
因此不需要锁。需要 asyncio 3.11+（依赖 asyncio.timeout 与 Self 类型）。

用法:
    export DEEPSEEK_API_KEY=sk-xxx
    uv run python -m uv_and_httpx.ReActAgent                    # 默认离线工具，不产生搜索费用
    uv run python -m uv_and_httpx.ReActAgent --online "英伟达最新的GPU型号是什么"
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import re
from typing import Any, Callable, Literal

from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from uv_and_httpx.ToolExecutor import ToolArgs, ToolError, ToolExecutor
from uv_and_httpx.chapter4 import HelloAgentsLLM
from uv_and_httpx.search import asearch

# ReAct 提示词模板
REACT_PROMPT_TEMPLATE = """
请注意，你是一个有能力调用外部工具的智能助手。

可用工具如下:
{tools}

请严格按照以下格式进行回应:

Thought: 你的思考过程，用于分析问题、拆解任务和规划下一步行动。
Action: 你决定采取的行动，必须是以下格式之一:
- `{{tool_name}}[{{tool_input}}]`:调用一个可用工具。
- `Finish[最终答案]`:当你认为已经获得最终答案时。
- 当你收集到足够的信息，能够回答用户的最终问题时，你必须在Action:字段后使用 Finish[最终答案] 来输出最终答案。
- 如果下一步需要多个彼此不依赖的工具调用，可以在同一轮里写多行 Action，它们会被并发执行。

现在，请开始解决以下问题:
Question: {question}
History: {history}
"""

# Finish[答案]：贪婪匹配，答案里可能自带 [1]、markdown 链接等方括号
FINISH_PATTERN = re.compile(r"Finish\[(.*)\]", re.DOTALL)
THOUGHT_PATTERN = re.compile(r"Thought:\s*(.*?)(?=\nAction:|$)", re.DOTALL)
# 一轮里可以有多个 Action；Finish 必须排在最后，用贪婪匹配取到文末
ACTION_PATTERN = re.compile(r"Action:\s*(Finish\[.*\]|\w+\[.*?\])", re.DOTALL)


# --------------------------------------------------------------------------- #
# Pydantic 模型：配置 / 工具参数 / 模型响应 / 执行结果
# --------------------------------------------------------------------------- #
class AgentConfig(BaseModel):
    """Agent 运行配置，带上下界，非法配置在构造期就报错。

    frozen=True：运行期不可变，避免中途被改出半截状态；
    extra="forbid"：拼错字段名（如 max_concurency）会立刻报错，而不是静默失效。

    超时上有三处不同的计时器，别混淆：
    - llm_timeout   一次模型调用（_think）
    - tool_timeout  一次工具调用（_execute_with_retry 的 try 块）
    - 退避等待      0.5 * attempt 秒，另有 +1s 上界兜底
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_steps: int = Field(default=5, ge=1, le=20, description="ReAct 最大轮数")
    max_concurrency: int = Field(default=3, ge=1, le=16, description="同时执行的工具数上限")
    tool_timeout: float = Field(default=20.0, gt=0, le=300, description="单次工具调用超时（秒）")
    llm_timeout: float = Field(default=90.0, gt=0, le=600, description="单次模型调用超时（秒）")
    tool_retries: int = Field(default=1, ge=0, le=3, description="工具失败后的重试次数（0 = 不重试）")


class SearchArgs(ToolArgs):
    """Search 工具的参数模型（上限对齐 DeepSeek Search 的 128 字符限制）。"""

    query: str = Field(min_length=2, max_length=128, description="要搜索的自然语言问题")


class ActionCall(BaseModel):
    """一个通过格式校验的工具调用。

    两层校验中的第一层：只保证「工具名长得像标识符、参数非空」，
    真正的参数校验发生在执行前一刻（ToolExecutor.parseArgs，见 _execute_with_retry 的第 ① 步）。
    拆成两层的原因：这一层要容忍模型即兴发挥（丢掉明显畸形的行），
    那一层才按工具自己的 schema 严格拦截。
    """

    model_config = ConfigDict(frozen=True)

    tool_name: str = Field(min_length=1, max_length=64, pattern=r"^\w+$")
    tool_input: str = Field(min_length=1, max_length=4000)  # 原始文本，可能是一段 query，也可能是 JSON

    @property
    def raw(self) -> str:
        """还原成模型写出的 `Tool[input]` 形式，用于打印和写入 history。"""
        return f"{self.tool_name}[{self.tool_input}]"


class ReActResponse(BaseModel):
    """对模型一次输出的结构化解析结果。

    三种可能的组合：
    - final_answer 非空       → 本轮结束，直接返回答案
    - actions 非空            → 继续执行动作
    - 两者皆空（thought 有值）→ 模型没按格式写，交给上层降级
    """

    thought: str | None = None
    actions: list[ActionCall] = Field(default_factory=list)
    final_answer: str | None = None

    @property
    def is_finished(self) -> bool:
        return self.final_answer is not None


class ToolObservation(BaseModel):
    """一次工具调用的结果，区分成功 / 参数非法 / 未找到 / 超时 / 异常。

    status 的作用是让「失败」也有结构：既能写进 history 提示模型，
    也能在整体降级时只挑 status == "ok" 的结果拼答案。
    """

    tool_name: str
    tool_input: str
    status: Literal["ok", "invalid_args", "not_found", "timeout", "error"]
    output: str
    attempts: int = 0  # 实际尝试次数；invalid_args / not_found 恒为 0，说明工具压根没跑

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def toHistory(self) -> str:
        """转成写进 history 的一行；失败的打上「已降级」标记，让模型知道别再重复踩坑。"""
        if self.ok:
            return f"Observation: {self.output}"
        return f"Observation: （工具失败，已降级）{self.output}"


class ReActAgent:
    """ReAct 循环的编排者。

    它自己不写提示词模板、不发 HTTP 请求、不解析搜索结果，只负责把三件事串成循环：
    「拼提示 → 问模型 → 执行动作」。具体脏活由两个协作者承担：
    - HelloAgentsLLM：模型调用（think 同步 / athink 异步）
    - ToolExecutor：工具注册、参数校验、取函数

    状态全部挂在实例上（history / observations），所以同一个实例不要在多个任务里并发
    arun()；但一个实例可以反复 arun()，每次开头都会重置状态。
    """

    def __init__(
        self,
        llm_client: HelloAgentsLLM,
        tool_executor: ToolExecutor,
        config: AgentConfig | None = None,
        max_steps: int | None = None,
    ):
        """
        Args:
            llm_client: 模型客户端，优先使用其 athink()；没有则把 think() 丢进线程池。
            tool_executor: 工具注册表，负责「工具名 → 函数 + 参数模型」的查找。
            config: 运行配置，缺省用 AgentConfig() 默认值。
            max_steps: 便捷参数，等价于 config.max_steps，保留是为了兼容旧调用方式。
        """
        self.llm_client = llm_client
        self.tool_executor = tool_executor
        # max_steps 保留为便捷参数，兼容旧调用方式（优先级高于 config）
        if max_steps is not None:
            config = (config or AgentConfig()).model_copy(update={"max_steps": max_steps})
        self.config = config or AgentConfig()

        self.history: list[str] = []          # 每轮的 "Action: ..." / "Observation: ..." 纯文本行
        self.observations: list[ToolObservation] = []  # 结构化版本，供整体降级时挑选
        # 每次运行在 arun 里重建：Semaphore 会绑定事件循环，跨 asyncio.run() 复用会报错
        self._semaphore: asyncio.Semaphore | None = None

    # ------------------------------------------------------------------ #
    # 主流程
    # ------------------------------------------------------------------ #
    async def arun(self, question: str) -> str | None:
        """异步运行 ReAct 智能体来回答一个问题。

        一轮循环 = 七步：
            1 拼提示 → 2 问模型 → 3 解析输出 → 4 判断 Finish
            → 5 同轮去重 → 6 并发执行 → 7 写回 history
        退出条件有三：拿到 Finish、模型调用失败、步数耗尽（后两者走 _degrade 兜底）。

        Args:
            question: 用户的自然语言问题。

        Returns:
            最终答案字符串；拿不到答案时返回 _degrade() 拼出的「不完整答案」而不是 None。
        """
        # 每次 run 都重置状态：同一个实例可以被反复调用
        self.history = []
        self.observations = []
        # Semaphore 必须在事件循环内创建/使用，所以放在这里而不是 __init__
        self._semaphore = asyncio.Semaphore(self.config.max_concurrency)

        for step in range(1, self.config.max_steps + 1):
            print(f"--- 第 {step} 步 ---")

            # 1 + 2. 拼提示并异步问模型（_think 内部套了 llm_timeout）
            response_text = await self._think(self._build_prompt(question))
            if not response_text:
                # 模型挂了/超时：这轮彻底没法继续，直接降级
                return self._degrade(question, "模型调用失败或超时")

            # 3. 解析并校验模型输出（畸形的 Action 在这里就被丢掉，不会走到执行）
            parsed = self.parse_response(response_text)
            if parsed.thought:
                print(f"思考: {parsed.thought}")

            # 4. 模型给出了最终答案 → 收工
            if parsed.is_finished:
                final_answer = (parsed.final_answer or "").strip()
                print(f"🎉 最终答案: {final_answer}")
                return final_answer

            # 4'. 既没有 Finish 也没有合法 Action：模型没按格式回答，不再空转，直接降级
            if not parsed.actions:
                print("警告: 未能解析出有效的 Action，流程终止。")
                return self._degrade(question, "模型未给出合法的 Action")

            # 5. 同一轮里重复的调用只跑一次（搜索按次计费，别为相同 query 付两次钱）
            #    用 (工具名, 原始参数) 做键；dict 保序，所以执行顺序仍与模型书写顺序一致
            unique: dict[tuple[str, str], ActionCall] = {}
            for action in parsed.actions:
                unique.setdefault((action.tool_name, action.tool_input), action)
            deduped = len(parsed.actions) - len(unique)

            # 6. 并发执行本轮所有相互独立的 Action
            #    gather 会把每个协程包成 Task 立刻调度；真正的并发上限由 _execute_action 的 Semaphore 控制
            message = f"⚡ 并发执行 {len(unique)} 个 Action（并发上限 {self.config.max_concurrency}）"
            if deduped:
                message += f"，已合并 {deduped} 个重复调用"
            print(message)

            keys = list(unique)
            executed = await asyncio.gather(
                *(self._execute_action(unique[key]) for key in keys)
            )
            # gather 的返回顺序与入参一致；再按去重键映射回每个原始 Action
            by_key = dict(zip(keys, executed))
            results = [by_key[(action.tool_name, action.tool_input)] for action in parsed.actions]

            # 7. 按模型书写顺序写回 history；失败的结果也写，让模型知道并自行规避
            #    注意这里在 gather 之后串行写，所以 self.history 不存在并发写问题
            for action, observation in zip(parsed.actions, results):
                print(f"🎬 行动: {action.raw}")
                print(f"👀 观察: {observation.output}")
                self.history.append(f"Action: {action.raw}")
                self.history.append(observation.toHistory())
                self.observations.append(observation)

        print("已达到最大步数，流程终止。")
        return self._degrade(question, "达到最大步数仍未得到最终答案")

    def run(self, question: str) -> str | None:
        """同步入口（向后兼容）：内部用 asyncio.run 驱动 arun。

        注意：只能在「没有正在运行的事件循环」的线程里调用
        （例如脚本的 __main__、或普通同步函数）。在 async 代码里请直接 await arun()，
        否则 asyncio.run 会抛 RuntimeError。
        """
        return asyncio.run(self.arun(question))

    # ------------------------------------------------------------------ #
    # 模型调用
    # ------------------------------------------------------------------ #
    async def _think(self, prompt: str) -> str | None:
        """问一次模型，返回纯文本；超时或异常统一返回 None（由调用方决定怎么降级）。

        三种情况：
        - 客户端有 athink()：直接 await，最理想
        - 只有同步 think()：丢进线程池，避免阻塞事件循环（超时依然生效，但线程无法被强杀）
        - 抛异常：这里只吞普通异常；CancelledError 属于取消，必须继续上报
        """
        messages: list[ChatCompletionMessageParam] = [{"role": "user", "content": prompt}]
        try:
            # asyncio.timeout 到点会向内部抛 CancelledError，退出时转换成 TimeoutError
            async with asyncio.timeout(self.config.llm_timeout):
                athink = getattr(self.llm_client, "athink", None)
                if athink is None:
                    # 客户端没有异步接口时退回线程池，同样受超时约束
                    return await asyncio.to_thread(self.llm_client.think, messages=messages)
                return await athink(messages=messages)
        except TimeoutError:
            print(f"⏰ 模型调用超时（>{self.config.llm_timeout}s）")
            return None
        except asyncio.CancelledError:
            # 外部取消不是失败，必须让它继续往上传播
            raise
        except Exception as e:
            print(f"❌ 模型调用失败: {e}")
            return None

    # ------------------------------------------------------------------ #
    # 工具执行
    # ------------------------------------------------------------------ #
    async def _execute_action(self, action: ActionCall) -> ToolObservation:
        """单次工具执行的「入口」：只做并发限流，具体逻辑在 _execute_with_retry。

        Semaphore 的临界区故意只包住执行部分（不含调用方拼接提示等步骤），
        这样并发数严格等于「同时在跑的工具个数」。多出来的调用会在 async with 上排队，
        排队期间不占用线程，也不消耗工具的重试次数。
        """
        assert self._semaphore is not None, "请通过 arun() 启动，semaphore 在运行时创建"
        async with self._semaphore:
            return await self._execute_with_retry(action)

    async def _execute_with_retry(self, action: ActionCall) -> ToolObservation:
        """执行单个 Action，分三阶段；无论失败与否都返回 ToolObservation，只有取消会抛出。

        ① 参数校验：ValidationError → status=invalid_args，工具函数一次都不会被调用
        ② 执行：套 tool_timeout，失败按 tool_retries 重试（指数级退避）
        ③ 降级：重试用尽 → 返回带「已降级」说明的 Observation，让 ReAct 循环继续

        失败分类决定「重不重试」：
        - TimeoutError / 普通 Exception / ToolError(retryable=True) → 重试
        - ToolError(retryable=False)（鉴权、余额、4xx）→ 立即降级，不浪费重试
        - CancelledError → 直接抛，由最外层结束整个流程
        """
        name = action.tool_name
        func = self.tool_executor.getTool(name)
        if func is None:
            return ToolObservation(
                tool_name=name,
                tool_input=action.tool_input,
                status="not_found",
                output=f"错误:未找到名为 '{name}' 的工具，已跳过该调用。",
            )

        # ① 参数校验：不通过直接返回，工具函数绝不执行
        try:
            args = self.tool_executor.parseArgs(name, action.tool_input)
        except ValidationError as e:
            first = e.errors()[0]
            detail = f"{'.'.join(str(x) for x in first['loc'])}: {first['msg']}"
            print(f"⛔ 参数校验失败，工具 '{name}' 未执行: {detail}")
            return ToolObservation(
                tool_name=name,
                tool_input=action.tool_input,
                status="invalid_args",
                output=f"⛔ 参数校验失败，工具未执行（{detail}），请修正参数后重试。",
            )

        # ② 执行：超时 + 重试
        #    range 上界 = 1 次首试 + tool_retries 次重试；attempt 从 1 开始，就是对外报告的尝试次数
        status: Literal["timeout", "error"] = "error"
        last_error = "未知错误"
        attempts = 0
        for attempt in range(1, self.config.tool_retries + 2):
            attempts = attempt
            try:
                async with asyncio.timeout(self.config.tool_timeout):
                    output = await self._call_tool(func, args, action.tool_input)
                return ToolObservation(
                    tool_name=name,
                    tool_input=action.tool_input,
                    status="ok",
                    output=output,
                    attempts=attempt,
                )
            except TimeoutError:
                status = "timeout"
                last_error = f"执行超时（>{self.config.tool_timeout}s）"
                print(f"⏰ 工具 '{name}' {last_error}")
            except asyncio.CancelledError:
                # 取消不是工具失败：向上抛出，由调用方结束整个流程
                raise
            except ToolError as e:
                status = "error"
                last_error = str(e)
                print(f"❌ 工具 '{name}' 执行失败: {last_error}")
                if not e.retryable:
                    # 鉴权失败 / 余额不足 / 参数不合法：重试没意义，直接降级
                    return ToolObservation(
                        tool_name=name,
                        tool_input=action.tool_input,
                        status="error",
                        output=f"⚠️ 工具 '{name}' {last_error}；该错误重试无效，已降级：跳过该工具，请用已有信息继续推理。",
                        attempts=attempt,
                    )
            except Exception as e:
                status = "error"
                last_error = f"执行异常: {type(e).__name__}: {e}"
                print(f"❌ 工具 '{name}' {last_error}")

            if attempt <= self.config.tool_retries:
                delay = 0.5 * attempt
                print(f"↻ {delay}s 后重试 '{name}'（第 {attempt}/{self.config.tool_retries} 次）")
                async with asyncio.timeout(delay + 1.0):  # 退避等待也要有上界
                    await asyncio.sleep(delay)

        # ③ 重试耗尽 -> 降级成一条 Observation，让 ReAct 循环继续
        return ToolObservation(
            tool_name=name,
            tool_input=action.tool_input,
            status=status,
            output=f"⚠️ 工具 '{name}' {last_error}，已降级：跳过该工具，请用已有信息继续推理。",
            attempts=attempts,
        )

    async def _call_tool(
        self,
        func: Callable[..., Any],
        args: ToolArgs | None,
        raw_input: str,
    ) -> str:
        """真正调用工具：协程直接 await，同步函数丢进线程池（否则超时/取消无法生效）。

        入参约定：
        - 工具声明了参数模型 → 用 args.model_dump() 作为关键字实参，例如 search(query=...)
        - 工具没声明参数模型 → 退化成把原始字符串当唯一位置参数传进去（兼容旧注册方式）

        注意：asyncio.to_thread 无法真正杀死线程，超时只是「不再等它」，
        后台线程会跑完；对 HTTP 类的阻塞调用通常还能接受，CPU 死循环则不适用。
        """
        kwargs: dict[str, Any] = args.model_dump() if args is not None else {}
        if inspect.iscoroutinefunction(func):
            result = await func(**kwargs) if args is not None else await func(raw_input)
        elif args is not None:
            result = await asyncio.to_thread(func, **kwargs)
        else:
            result = await asyncio.to_thread(func, raw_input)
        # 统一成字符串：Observation 要写进 history，非字符串结果用 str() 兜底
        return result if isinstance(result, str) else str(result)

    # ------------------------------------------------------------------ #
    # 解析与降级
    # ------------------------------------------------------------------ #
    @staticmethod
    def parse_response(text: str) -> ReActResponse:
        """解析并校验模型输出；非法的 Action 会被丢弃，不会进入执行逻辑。

        解析规则（正则都在模块顶部）：
        - Thought 取到下一个 "Action:" 前为止，缺失则为 None
        - Action 用 finditer 找全部匹配，支持一轮多行 Action（并发执行的前提）
        - 遇到 Finish 立即停止收集：其后的内容一律忽略（Finish 必须写在最后一行的原因）
        - 单个 Action 若形态不合法（工具名非标识符、参数为空/超长）只打警告跳过，
          不影响同轮其它合法 Action；这样一次格式失误不会让整轮作废
        """
        thought_match = THOUGHT_PATTERN.search(text)
        thought = thought_match.group(1).strip() if thought_match else None

        actions: list[ActionCall] = []
        final_answer: str | None = None

        for match in ACTION_PATTERN.finditer(text):
            raw = match.group(1).strip()
            if raw.startswith("Finish"):
                finish_match = FINISH_PATTERN.match(raw)
                if finish_match:
                    final_answer = finish_match.group(1).strip()
                else:
                    # 兜底：模型可能写成 Finish[答案（缺右括号）或 Finish 答案]
                    final_answer = raw.removeprefix("Finish").strip().strip("[]`").strip()
                break  # Finish 之后的内容一律忽略

            # 工具调用用非贪婪匹配到第一个 ]，避免把后面的 Observation / Thought 一起吞进来
            action_match = re.match(r"(\w+)\[(.*)\]", raw, re.DOTALL)
            if not action_match:
                print(f"⚠️ 忽略无法解析的 Action: {raw}")
                continue
            try:
                actions.append(
                    ActionCall(
                        tool_name=action_match.group(1),
                        tool_input=action_match.group(2).strip(),
                    )
                )
            except ValidationError as e:
                print(f"⚠️ 忽略非法 Action `{raw}`: {e.errors()[0]['msg']}")

        return ReActResponse(thought=thought, actions=actions, final_answer=final_answer)

    def _build_prompt(self, question: str) -> str:
        """拼本轮提示词：工具清单 + 问题 + 累积的历史（历史只存 Action/Observation 文本）。"""
        return REACT_PROMPT_TEMPLATE.format(
            tools=self.tool_executor.getAvailableTools(),
            question=question,
            history="\n".join(self.history),
        )

    def _degrade(self, question: str, reason: str) -> str:
        """整体降级：把已成功收集到的观察拼成答案，而不是直接返回 None。

        这是「部分失败」的最后一道保险：即使 ReAct 没跑出 Finish，
        用户也能拿到已经搜到的信息 + 失败原因，而不是一个空结果。
        """
        useful = [o for o in self.observations if o.ok]
        # 并发搜索经常拿到重复结果，按 output 文本去重后再拼
        unique: dict[str, ToolObservation] = {}
        for o in useful:
            unique.setdefault(o.output, o)
        if not unique:
            return f"抱歉，暂时无法回答「{question}」：{reason}，且没有成功执行的工具结果可供参考。"
        detail = "\n".join(f"- [{o.tool_name}] {output}" for output, o in unique.items())
        return f"关于「{question}」，因{reason}，以下是已收集到的信息（可能不完整）：\n{detail}"


# --------------------------------------------------------------------------- #
# 离线工具：搜索额度已用完时仍能演示完整 ReAct 流程
# --------------------------------------------------------------------------- #
SEARCH_DESCRIPTION = (
    "一个网页搜索引擎。当你需要回答关于时事、事实以及在你的知识库中找不到的信息时，应使用此工具。"
)

OFFLINE_KNOWLEDGE = {
    "gpu": "NVIDIA 数据中心 GPU 当前主力为 Blackwell 架构（B200 / GB200），上一代为 Hopper（H100 / H200）。",
    "英伟达": "NVIDIA 数据中心 GPU 当前主力为 Blackwell 架构（B200 / GB200），上一代为 Hopper（H100 / H200）。",
    "nvidia": "NVIDIA 数据中心 GPU 当前主力为 Blackwell 架构（B200 / GB200），上一代为 Hopper（H100 / H200）。",
}


async def offline_search(query: str) -> str:
    """离线假搜索：DeepSeek Search 余额不足或断网时仍能跑通整条链路（也用作降级兜底）。"""
    await asyncio.sleep(0.2)
    lowered = query.lower()
    for key, value in OFFLINE_KNOWLEDGE.items():
        if key in lowered:
            return value
    return f"[离线结果] 本地知识库未命中：{query}"


async def _amain() -> None:
    parser = argparse.ArgumentParser(description="异步 ReAct Agent demo")
    parser.add_argument("question", nargs="?", default="英伟达最新的GPU型号是什么")
    parser.add_argument(
        "--online",
        action="store_true",
        help="使用 DeepSeek Search API（按成功请求计费，5 元/千次）",
    )
    parser.add_argument("--max-steps", type=int, default=5)
    args = parser.parse_args()

    # 1. 工具箱：默认离线工具；--online 走 DeepSeek Search（同一个 DEEPSEEK_API_KEY）
    tool_executor = ToolExecutor()
    tool_executor.registerTool(
        "Search",
        SEARCH_DESCRIPTION,
        asearch if args.online else offline_search,
        SearchArgs,
    )

    # 2. LLM 客户端（自动读 .env / 环境变量）
    llm_client = HelloAgentsLLM()

    # 3. 组装智能体并运行
    agent = ReActAgent(
        llm_client,
        tool_executor,
        AgentConfig(max_steps=args.max_steps, max_concurrency=3, tool_timeout=20),
    )
    answer = await agent.arun(args.question)
    print(f"\n最终返回: {answer}")


if __name__ == "__main__":
    asyncio.run(_amain())
