import pytest
from typing import Any, Dict, List

from agents.llm.base_llm import BaseLLM
from agents.tools.base import JustInTimeToolingBase, ToolBase


class DummyLLM(BaseLLM):
    def __init__(self, *, text_queue: List[str] | None = None, json_queue: List[Dict[str, Any]] | None = None):
        # Intentionally do not call super().__init__ to avoid model env requirement
        self.text_queue = list(text_queue or [])
        self.json_queue = list(json_queue or [])

    # Satisfy abstract method from BaseLLM
    def completion(self, messages: List[Dict[str, str]], **kwargs) -> str:  # type: ignore[override]
        if not self.text_queue:
            return ""
        return self.text_queue.pop(0)

    # Override to avoid BaseLLM JSON-mode behavior
    def prompt(self, text: str) -> str:  # type: ignore[override]
        if not self.text_queue:
            return ""
        return self.text_queue.pop(0)

    def prompt_to_json(self, text: str, max_retries: int = 0) -> Dict[str, Any]:  # type: ignore[override]
        if not self.json_queue:
            return {}
        return self.json_queue.pop(0)


class DummyTool(ToolBase):
    def __init__(self, id: str, summary: str, schema: Dict[str, Any] | None = None):
        self.id = id
        self._summary = summary
        self._schema = schema or {}

    def get_summary(self) -> str:
        return self._summary

    def get_parameters(self) -> Dict[str, Any]:
        return self._schema

    def get_details(self) -> Dict[str, Any]:
        return {"id": self.id, "summary": self._summary}


class DummyTools(JustInTimeToolingBase):
    def __init__(self, tools: List[ToolBase] | None = None, failures: Dict[str, Exception] | None = None):
        self._tools = tools or []
        self._failures = failures or {}

    def search(self, query: str, top_k: int = 15) -> List[ToolBase]:
        return self._tools[:top_k]

    def load(self, tool: ToolBase) -> ToolBase:
        return tool

    def execute(self, tool: ToolBase, params: Dict[str, Any]) -> Any:
        err = self._failures.get(getattr(tool, "id", ""))
        if err:
            raise err
        return {"ok": True, "tool": tool.id, "params": params}


class CaptureTools(DummyTools):
    """Test helper: wraps DummyTools but captures last executed params and maps loads.

    - last_params stores the most recent params passed to execute
    - load maps arbitrary tool-like objects by id back to the registered DummyTool
    """
    def __init__(self, tools: List[ToolBase] | None = None, failures: Dict[str, Exception] | None = None):
        super().__init__(tools=tools, failures=failures)
        self.last_params: Dict[str, Any] | None = None

    def load(self, tool: ToolBase) -> ToolBase:  # type: ignore[override]
        target_id = getattr(tool, "id", None)
        mapped = next((t for t in self._tools if getattr(t, "id", None) == target_id), None)
        return mapped or tool

    def execute(self, tool: ToolBase, params: Dict[str, Any]) -> Any:  # type: ignore[override]
        self.last_params = params
        return super().execute(tool, params)


@pytest.fixture
def dummy_llm() -> DummyLLM:
    return DummyLLM()


@pytest.fixture
def dummy_tools() -> DummyTools:
    return DummyTools()
