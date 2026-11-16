from typing import Any, Callable, Dict, Self

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict

load_dotenv()


class ToolError(Exception):
    """工具执行失败的统一异常。

    retryable=False 表示重试没有意义（参数不合法、鉴权失败、余额不足等），
    上层应立即降级；retryable=True 表示偶发失败（限流、超时、5xx），值得重试。
    """

    def __init__(self, message: str, *, retryable: bool = True) -> None:
        super().__init__(message)
        self.retryable = retryable


class ToolArgs(BaseModel):
    """工具参数的基类。

    每个工具声明自己的参数模型，ReAct 输出里的原始字符串在这里被校验成结构化参数；
    校验不通过会抛 pydantic.ValidationError，调用方必须拦下来，不要进入工具执行逻辑。
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    @classmethod
    def fromToolInput(cls, raw_input: str) -> Self:
        """把 `Action: Tool[...]` 里的原始文本解析成参数模型。"""
        field_names = list(cls.model_fields)
        if len(field_names) == 1:
            # 单参数工具：Search[英伟达最新GPU] -> {"query": "英伟达最新GPU"}
            return cls.model_validate({field_names[0]: raw_input})
        # 多参数工具：Weather[{"city": "杭州", "unit": "c"}]
        return cls.model_validate_json(raw_input)


class ToolExecutor:
    """
    一个工具执行器，负责管理和执行工具。
    """

    def __init__(self):
        self.tools: Dict[str, Dict[str, Any]] = {}

    def registerTool(
        self,
        name: str,
        description: str,
        func: Callable[..., Any],
        argsModel: type[ToolArgs] | None = None,
    ):
        """
        向工具箱中注册一个新工具。

        argsModel 为该工具的参数模型；声明后，调用前会先做参数校验。
        """
        if name in self.tools:
            print(f"警告:工具 '{name}' 已存在，将被覆盖。")
        self.tools[name] = {
            "description": description,
            "func": func,
            "args_model": argsModel,
        }
        print(f"工具 '{name}' 已注册。")

    def getTool(self, name: str) -> Callable[..., Any] | None:
        """
        根据名称获取一个工具的执行函数。
        """
        return self.tools.get(name, {}).get("func")

    def getArgsModel(self, name: str) -> type[ToolArgs] | None:
        """获取工具声明的参数模型（未声明时返回 None）。"""
        return self.tools.get(name, {}).get("args_model")

    def parseArgs(self, name: str, raw_input: str) -> ToolArgs | None:
        """按工具声明的参数模型校验原始输入，非法参数会抛 ValidationError。"""
        args_model = self.getArgsModel(name)
        if args_model is None:
            return None
        return args_model.fromToolInput(raw_input)

    def getAvailableTools(self) -> str:
        """
        获取所有可用工具的格式化描述字符串。
        """
        return "\n".join([
            f"- {name}: {info['description']}" 
            for name, info in self.tools.items()
        ])

# --- 工具初始化与使用示例 ---
if __name__ == '__main__':
    # 延迟导入：search 依赖本模块的 ToolError，放在模块级别会循环导入
    from uv_and_httpx.search import search

    # 1. 初始化工具执行器
    toolExecutor = ToolExecutor()

    # 2. 注册我们的实战搜索工具
    search_description = "一个网页搜索引擎。当你需要回答关于时事、事实以及在你的知识库中找不到的信息时，应使用此工具。"
    toolExecutor.registerTool("Search", search_description, search)
    
    # 3. 打印可用的工具
    print("\n--- 可用的工具 ---")
    print(toolExecutor.getAvailableTools())

    # 4. 智能体的Action调用，这次我们问一个实时性的问题
    print("\n--- 执行 Action: Search['英伟达最新的GPU型号是什么'] ---")
    tool_name = "Search"
    tool_input = "英伟达最新的GPU型号是什么"

    tool_function = toolExecutor.getTool(tool_name)
    if tool_function:
        observation = tool_function(tool_input)
        print("--- 观察 (Observation) ---")
        print(observation)
    else:
        print(f"错误:未找到名为 '{tool_name}' 的工具。")        
