import inspect
import asyncio
import warnings
from typing import (
    Any,
    Dict,
    Callable,
    List,
    Optional,
    get_type_hints,
    Annotated,
    get_origin,
    get_args,
)
from agents.tools.base import ToolBase, JustInTimeToolingBase


def parse_annotation(ann):
    if get_origin(ann) is Annotated:
        base_type, *metadata = get_args(ann)
        description = ", ".join(str(m) for m in metadata)
        return base_type, description
    return ann, ""


def extract_param_info(func):
    sig = inspect.signature(func)
    hints = get_type_hints(func, include_extras=True)
    result = {}

    for name, param in sig.parameters.items():
        ann = hints.get(name, param.annotation)
        base_type, description = parse_annotation(ann)

        required = param.default is inspect.Parameter.empty
        # default_value = None if required else param.default

        result[name] = {
            "type": getattr(base_type, "__name__", str(base_type)),
            "required": required,
            # "default": default_value,
            "description": description,
        }
    return result


class FunctionTool(ToolBase):
    def __init__(self, func: Callable, keywords: Optional[List[str]] = None):
        self.func = func
        self._keywords = keywords or []
        self.signature = inspect.signature(func)

        # Validate docstring
        if not func.__doc__:
            warnings.warn(
                f"Functiontool '{func.__name__}' has no docstring. "
                "Please add one — the first 2 lines will be used as summary."
            )
        super().__init__(id=func.__name__)

    def get_keywords(self) -> List[str]:
        return self._keywords

    def get_summary(self) -> str:
        """
        Return the first 1–2 lines of the function docstring as the summary.
        """
        if not self.func.__doc__:
            return "(No documentation provided)"
        lines = [line.strip() for line in self.func.__doc__.splitlines() if line.strip()]

        return f"{self.id}: {' '.join(lines[:2])}"

    def get_details(self) -> str:
        """
        Return the docs of function.
        """
        return self.func.__doc__ or ""

    def get_parameter_schema(self) -> Dict[str, Any]:
        return extract_param_info(self.func)

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)


class FunctionToolProvider(JustInTimeToolingBase):
    def __init__(self):
        self._tools: Dict[str, FunctionTool] = {}

    def tool(self, keywords: Optional[List[str]] = None):
        """
        Decorator to register a function as a tool for this provider, with optional keywords.

        Args:
            keywords (List[str], optional): Keywords describing the tool for search/matching.
        """
        def decorator(func: Callable):
            tool_obj = FunctionTool(func, keywords=keywords)
            self._tools[tool_obj.id] = tool_obj
            return tool_obj
        return decorator

    def search(self, query: str, *, top_k: int = 10) -> List[FunctionTool]:
        results = []
        query_lower = query.lower()
        for tool in self._tools.values():
            summary = tool.get_summary().lower()
            if (
                any(word in summary for word in query_lower.split())
                or any(kw in query_lower.split() for kw in tool.get_keywords())
                or tool.id.lower() in query_lower
            ):
                results.append(tool)
        return results[:top_k]

    def load(self, tool: FunctionTool) -> FunctionTool:
        if getattr(tool, 'id', None) not in self._tools:
            raise ValueError(f"function tool dont have  '{tool}' defenition")
        return tool

    async def _execute_async(self, tool: FunctionTool, parameters: Dict[str, Any]) -> Any:
        bound = inspect.signature(tool.func).bind(**parameters)
        return await tool.func(*bound.args, **bound.kwargs)

    def _execute_sync(self, tool: FunctionTool, parameters: Dict[str, Any]) -> Any:
        bound = inspect.signature(tool.func).bind(**parameters)
        return tool.func(*bound.args, **bound.kwargs)

    def execute(self, tool: FunctionTool, parameters: Dict[str, Any]) -> Any:
        """Detect if async or sync function and run appropriately."""
        if inspect.iscoroutinefunction(tool.func):
            return asyncio.run(self._execute_async(tool, parameters))
        else:
            return self._execute_sync(tool, parameters)
