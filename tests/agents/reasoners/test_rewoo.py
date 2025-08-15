from agents.memory.dict_memory import DictMemory
from agents.reasoner.rewoo import ReWOOReasoner
from typing import Any, Dict, List

from tests.agents.reasoners.conftest import DummyLLM, DummyTools, DummyTool
from agents.tools.exceptions import ToolExecutionError

def test_rewoo_plan_parses_valid_bullets_and_records_successful_tool_call():
    # Plan: one TOOL step producing k1, then one REASONING step using k1
    plan_text = "\n".join([
        "- fetch data (output: k1)",
        "- summarize (input: k1) (output: k2)",
    ])

    llm = DummyLLM(
        text_queue=[
            plan_text,   # plan
            "TOOL",      # classify step 1
            "t1",        # tool selection
            "REASONING", # classify step 2
            "summary",   # reasoning step result
        ],
        json_queue=[
            {},  # param_gen for step 1
        ],
    )

    tools = DummyTools([DummyTool("t1", "Tool One", schema={})])
    memory: Dict[str, Any] = DictMemory()

    reasoner = ReWOOReasoner(llm=llm, tools=tools, memory=memory, max_iterations=5)
    result = reasoner.run("goal")

    # Successful tool call recorded once
    assert result.tool_calls and result.tool_calls[0] == {"tool_id": "t1", "summary": "Tool One"}
    # Transcript contains remembered k1 and executed steps
    assert "remembered k1" in result.transcript
    assert "Executed step:" in result.transcript


def test_rewoo_plan_raises_on_input_before_output():
    # Plan references input that was never produced
    plan_text = "- use prior (input: missing)"
    llm = DummyLLM(text_queue=[plan_text])
    tools = DummyTools([])
    memory: Dict[str, Any] = DictMemory()
    reasoner = ReWOOReasoner(llm=llm, tools=tools, memory=memory)

    try:
        reasoner._plan("goal")
        assert False, "Expected ValueError for input-before-output"
    except ValueError:
        pass


def test_rewoo_param_generation_filters_to_schema_keys():
    # Plan: single TOOL step; param_gen returns extra key 'c' which must be dropped
    plan_text = "- act (output: k1)"
    llm = DummyLLM(
        text_queue=[
            plan_text,  # plan
            "TOOL",     # classify step
            "t1",       # tool selection
        ],
        json_queue=[
            {"a": 1, "b": 2, "c": 3},  # param_gen
        ],
    )

    class CaptureTools(DummyTools):
        def __init__(self, tools: List[DummyTool]):
            super().__init__(tools)
            self.last_params: Dict[str, Any] | None = None

        def execute(self, tool, params):  # type: ignore[override]
            self.last_params = params
            return {"ok": True}

    tools = CaptureTools([DummyTool("t1", "Tool One", schema={"a": {}, "b": {}})])
    memory: Dict[str, Any] = DictMemory()
    reasoner = ReWOOReasoner(llm=llm, tools=tools, memory=memory)
    result = reasoner.run("goal")

    assert result.success in (True, False)
    assert tools.last_params == {"a": 1, "b": 2}
    assert result.tool_calls and result.tool_calls[0]["tool_id"] == "t1"


def test_rewoo_records_tool_call_on_success_only():
    # Plan: two TOOL steps. First succeeds (t1), second fails with execution error (t2)
    plan_text = "\n".join([
        "- step one (output: k1)",
        "- step two (input: k1)",
    ])

    t1 = DummyTool("t1", "Tool One", schema={})
    t2 = DummyTool("t2", "Tool Two", schema={})

    llm = DummyLLM(
        text_queue=[
            plan_text,  # plan
            "TOOL",     # classify step 1
            "t1",       # select tool for step 1
            "TOOL",     # classify step 2
            "t2",       # select tool for step 2 (will fail)
        ],
        json_queue=[
            {},  # param_gen for step 1
            {},  # param_gen for step 2
        ],
    )

    tools = DummyTools(tools=[t1, t2], failures={"t2": ToolExecutionError("boom", t2)})
    memory: Dict[str, Any] = DictMemory()

    # Disable retries so failure does not trigger any additional attempts that could add tool_calls
    reasoner = ReWOOReasoner(llm=llm, tools=tools, memory=memory, max_iterations=5, max_retries=0)
    result = reasoner.run("goal")

    # Exactly one successful tool call (t1) recorded; failing t2 is not recorded
    assert result.tool_calls == [{"tool_id": "t1", "summary": "Tool One"}]


def test_rewoo_does_not_record_tool_call_on_selection_error():
    # Plan: single TOOL step, but selection returns "none" causing a selection error
    plan_text = "- do it (output: k1)"
    t1 = DummyTool("t1", "Tool One", schema={})

    llm = DummyLLM(
        text_queue=[
            plan_text,  # plan
            "TOOL",     # classify step
            "none",     # tool selection → selection error
        ],
        json_queue=[
            {},  # would be param_gen, but selection fails before execute
        ],
    )

    tools = DummyTools(tools=[t1])
    memory: Dict[str, Any] = DictMemory()

    # Disable retries so we don't attempt reflection-based re-runs
    reasoner = ReWOOReasoner(llm=llm, tools=tools, memory=memory, max_iterations=3, max_retries=0)
    result = reasoner.run("goal")

    # No tool call recorded on selection error
    assert result.tool_calls == []


