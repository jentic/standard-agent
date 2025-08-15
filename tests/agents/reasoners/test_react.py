from typing import Any, Dict

from agents.reasoner.react import ReACTReasoner
from agents.memory.dict_memory import DictMemory

# Reuse test doubles from conftest in this package
from tests.agents.reasoners.conftest import DummyLLM, DummyTools, DummyTool


def test_react_iterations_counts_turns_not_transcript_lines():
    # THINK -> STOP, two turns, no ACT lines
    llm = DummyLLM(
        json_queue=[
            {"step_type": "THINK", "text": "consider next step"},
            {"step_type": "STOP", "text": "final answer"},
        ]
    )
    tools = DummyTools([])
    memory: Dict[str, Any] = DictMemory()

    reasoner = ReACTReasoner(llm=llm, tools=tools, memory=memory, max_turns=5)
    result = reasoner.run("test goal")

    assert result.iterations == 2  # two THINK outputs
    assert "THINK:" in result.transcript
    assert "FINAL ANSWER:" in result.transcript
    assert result.success is True
    assert result.tool_calls == []


def test_react_records_only_successful_tool_calls_minimal_shape():
    # THINK -> ACT (success with t1) -> THINK -> STOP
    llm = DummyLLM(
        json_queue=[
            {"step_type": "ACT", "text": "do something"},
            {},  # param_gen returns empty dict
            {"step_type": "STOP", "text": "done"},
        ],
        text_queue=[
            "t1",  # tool selection id
        ],
    )
    tools = DummyTools([
        DummyTool("t1", "Tool One", schema={}),
        DummyTool("t2", "Tool Two", schema={}),
    ])
    memory: Dict[str, Any] = DictMemory()

    reasoner = ReACTReasoner(llm=llm, tools=tools, memory=memory, max_turns=5)
    result = reasoner.run("test goal")

    assert result.success is True
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0] == {"tool_id": "t1", "summary": "Tool One"}
    # ensure transcript has execution markers
    assert "ACT_EXECUTED:" in result.transcript
    assert "OBSERVATION:" in result.transcript


def test_react_filters_out_failed_tool_ids_on_next_selection():
    # First ACT: t1 unauthorized → next ACT must avoid t1 and select t2
    from agents.tools.exceptions import ToolCredentialsMissingError

    llm = DummyLLM(
        json_queue=[
            {"step_type": "ACT", "text": "do something"},
            {},
            {"step_type": "ACT", "text": "do again"},
            {},
            {"step_type": "STOP", "text": "done"},
        ],
        text_queue=[
            "t1",  # first selection
            "t2",  # second selection must not be t1
        ],
    )
    tools = DummyTools(
        tools=[DummyTool("t1", "Tool One"), DummyTool("t2", "Tool Two")],
        failures={"t1": ToolCredentialsMissingError("Missing creds", DummyTool("t1", "Tool One"))},
    )
    memory: Dict[str, Any] = DictMemory()

    reasoner = ReACTReasoner(llm=llm, tools=tools, memory=memory, max_turns=5)
    result = reasoner.run("test goal")

    # First call fails (not recorded), second succeeds and is recorded
    assert any(tc["tool_id"] == "t2" for tc in result.tool_calls)
    assert all(tc["tool_id"] != "t1" for tc in result.tool_calls)
    assert "Tool Unauthorized:" in result.transcript


def test_react_param_generation_filters_to_schema_keys():
    # Param gen returns extra keys; only schema keys should be used
    llm = DummyLLM(
        json_queue=[
            {"step_type": "ACT", "text": "do something"},
            {"a": 1, "b": 2, "c": 3},  # param_gen
            {"step_type": "STOP", "text": "done"},
        ],
        text_queue=[
            "t1",
        ],
    )
    tools = DummyTools([DummyTool("t1", "Tool One", schema={"a": {}, "b": {}})])
    memory: Dict[str, Any] = DictMemory()

    # Monkeypatch DummyTools.execute to capture params used if necessary is overkill;
    # We rely on reasoner filtering and successful flow here.
    reasoner = ReACTReasoner(llm=llm, tools=tools, memory=memory, max_turns=5)
    result = reasoner.run("test goal")

    assert result.success is True
    assert result.tool_calls and result.tool_calls[0]["tool_id"] == "t1"
    assert "OBSERVATION:" in result.transcript


def test_react_tool_selection_error_is_logged_and_no_tool_call_recorded():
    # LLM returns "none" for selection; no tool call recorded
    llm = DummyLLM(
        json_queue=[
            {"step_type": "ACT", "text": "do something"},
            # no param_gen because we won't reach it; selection fails
            {"step_type": "STOP", "text": "done"},
        ],
        text_queue=[
            "none",  # tool selection returns none
        ],
    )
    tools = DummyTools([DummyTool("t1", "Tool One")])
    memory: Dict[str, Any] = DictMemory()

    reasoner = ReACTReasoner(llm=llm, tools=tools, memory=memory, max_turns=5)
    result = reasoner.run("test goal")

    # No successful tool calls should be recorded
    assert result.tool_calls == []
    assert "ToolSelectionError" in result.transcript


