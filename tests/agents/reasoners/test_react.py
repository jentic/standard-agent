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


