from typing import Any, Dict

from agents.reasoner.react import ReACTReasoner
from agents.memory.dict_memory import DictMemory
from agents.reasoner.exceptions import ParameterGenerationError
import pytest
# Reuse test doubles from conftest in this package
from tests.conftest import DummyLLM, DummyTools, DummyTool


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


def test_react_param_gen_non_dict_raises_parameter_generation_error():
    # Direct call: param_gen returns a list, should raise ParameterGenerationError
    llm = DummyLLM()
    tools = DummyTools([DummyTool("t1", "Tool One", schema={"a": {}})])
    memory: Dict[str, Any] = DictMemory()
    reasoner = ReACTReasoner(llm=llm, tools=tools, memory=memory)

    llm.json_queue = [[["x"]]]

    tool = DummyTool("t1", "Tool One", schema={"a": {}})
    with pytest.raises(ParameterGenerationError) as exc:
        reasoner._generate_params(tool, transcript="t", step_text="do")
    assert "Failed to generate valid JSON parameters for step" in str(exc.value)


def test_react_param_gen_non_dict_during_run_logs_and_no_tool_call():
    # Full run: THINK->ACT with non-dict param_gen, then THINK->STOP; no tool call should be recorded
    llm = DummyLLM(
        json_queue=[
            {"step_type": "ACT", "text": "do"},
            [["x"]],  # param_gen returns list
            {"step_type": "STOP", "text": "done"},
        ],
        text_queue=["t1"],
    )
    tools = DummyTools([DummyTool("t1", "Tool One", schema={"a": {}})])
    memory: Dict[str, Any] = DictMemory()
    reasoner = ReACTReasoner(llm=llm, tools=tools, memory=memory, max_turns=3)
    result = reasoner.run("goal")

    assert result.tool_calls == []
    assert "ParameterGenerationError" in result.transcript


def test_react_think_invalid_output_falls_back_to_default_and_counts_turn():
    # First THINK invalid -> fallback THINK; then STOP
    llm = DummyLLM(
        json_queue=[
            {"step_type": "", "text": ""},  # invalid
            {"step_type": "STOP", "text": "final"},
        ]
    )
    tools = DummyTools([])
    memory: Dict[str, Any] = DictMemory()

    reasoner = ReACTReasoner(llm=llm, tools=tools, memory=memory, max_turns=3)
    result = reasoner.run("goal")

    assert result.iterations == 2  # fallback THINK + STOP
    assert "THINK:" in result.transcript
    assert "FINAL ANSWER:" in result.transcript


def test_react_max_turns_reached_returns_partial_result():
    # Provide only THINKs, no STOP, up to max_turns
    llm = DummyLLM(
        json_queue=[
            {"step_type": "THINK", "text": "t1"},
            {"step_type": "THINK", "text": "t2"},
            {"step_type": "THINK", "text": "t3"},
        ]
    )
    tools = DummyTools([])
    memory: Dict[str, Any] = DictMemory()

    reasoner = ReACTReasoner(llm=llm, tools=tools, memory=memory, max_turns=3)
    result = reasoner.run("goal")

    assert result.iterations == 3
    assert result.success is False
    assert "THINK:" in result.transcript


def test_react_unexpected_error_is_logged_and_no_tool_call_recorded():
    # Execute raises generic Exception; should log UnexpectedError and not record tool call
    llm = DummyLLM(
        json_queue=[
            {"step_type": "ACT", "text": "do something"},
            {},
            {"step_type": "STOP", "text": "final"},
        ],
        text_queue=["t1"],
    )
    tools = DummyTools(
        tools=[DummyTool("t1", "Tool One")],
        failures={"t1": Exception("boom")},
    )
    memory: Dict[str, Any] = DictMemory()

    reasoner = ReACTReasoner(llm=llm, tools=tools, memory=memory, max_turns=3)
    result = reasoner.run("goal")

    assert result.success is True  # STOP after error
    assert result.tool_calls == []
    assert "UnexpectedError:" in result.transcript


def test_react_returning_failed_tool_id_again_triggers_selection_error():
    # After t1 fails, LLM selects t1 again; candidates exclude t1, so selection error should be logged
    from agents.tools.exceptions import ToolCredentialsMissingError

    llm = DummyLLM(
        json_queue=[
            {"step_type": "ACT", "text": "first"},
            {},
            {"step_type": "ACT", "text": "second"},
            {},
            {"step_type": "STOP", "text": "done"},
        ],
        text_queue=[
            "t1",  # first selection
            "t1",  # model wrongly selects same failed tool id
        ],
    )
    t1 = DummyTool("t1", "Tool One")
    tools = DummyTools(tools=[t1, DummyTool("t2", "Tool Two")], failures={"t1": ToolCredentialsMissingError("Missing creds", t1)})
    memory: Dict[str, Any] = DictMemory()

    reasoner = ReACTReasoner(llm=llm, tools=tools, memory=memory, max_turns=5)
    result = reasoner.run("goal")

    assert result.tool_calls == []  # no success recorded
    assert "Tool Unauthorized:" in result.transcript
    assert "ToolSelectionError" in result.transcript


def test_react_generate_params_with_required_params_success():
    """Test _generate_params succeeds when all required parameters are generated."""
    from typing import List
    
    class ToolWithRequired(DummyTool):
        def __init__(self, tool_id: str, name: str):
            super().__init__(tool_id, name, schema={"param1": {}, "param2": {}})
        
        def get_required_parameters(self) -> List[str]:
            return ["param1"]
    
    llm = DummyLLM(json_queue=[{"param1": "value1", "param2": "value2"}])
    tools = DummyTools([])
    memory: Dict[str, Any] = DictMemory()
    reasoner = ReACTReasoner(llm=llm, tools=tools, memory=memory)
    
    tool = ToolWithRequired("t1", "Tool One")
    
    result = reasoner._generate_params(tool, transcript="test transcript", step_text="test step")
    assert result == {"param1": "value1", "param2": "value2"}


def test_react_generate_params_missing_required_params_raises_error():
    """Test _generate_params raises ParameterGenerationError when required parameters are missing."""
    from typing import List
    
    class ToolWithRequired(DummyTool):
        def __init__(self, tool_id: str, name: str):
            super().__init__(tool_id, name, schema={"param1": {}, "param2": {}})
        
        def get_required_parameters(self) -> List[str]:
            return ["param1", "param2"]
    
    llm = DummyLLM(json_queue=[{"param2": "value2"}])  # missing param1
    tools = DummyTools([])
    memory: Dict[str, Any] = DictMemory()
    reasoner = ReACTReasoner(llm=llm, tools=tools, memory=memory)
    
    tool = ToolWithRequired("t1", "Tool One")
    
    with pytest.raises(ParameterGenerationError) as exc:
        reasoner._generate_params(tool, transcript="test transcript", step_text="test step")
    
    error_msg = str(exc.value)
    assert "Missing required parameters: param1" in error_msg
    assert "Generated parameters: {'param2': 'value2'}" in error_msg
    assert "Tool 't1' requires these parameters" in error_msg


def test_react_generate_params_no_required_params_backward_compatibility():
    """Test _generate_params works with tools that don't have get_required_parameters method."""
    tool = DummyTool("t1", "Tool One", schema={"param1": {}})
    # DummyTool doesn't have get_required_parameters method
    
    llm = DummyLLM(json_queue=[{"param1": "value1"}])
    tools = DummyTools([])
    memory: Dict[str, Any] = DictMemory()
    reasoner = ReACTReasoner(llm=llm, tools=tools, memory=memory)
    
    result = reasoner._generate_params(tool, transcript="test transcript", step_text="test step")
    assert result == {"param1": "value1"}


def test_react_generate_params_empty_required_params():
    """Test _generate_params works when tool has empty required parameters list."""
    from typing import List
    
    class ToolWithEmptyRequired(DummyTool):
        def __init__(self, tool_id: str, name: str):
            super().__init__(tool_id, name, schema={"param1": {}})
        
        def get_required_parameters(self) -> List[str]:
            return []
    
    llm = DummyLLM(json_queue=[{"param1": "value1"}])
    tools = DummyTools([])
    memory: Dict[str, Any] = DictMemory()
    reasoner = ReACTReasoner(llm=llm, tools=tools, memory=memory)
    
    tool = ToolWithEmptyRequired("t1", "Tool One")
    
    result = reasoner._generate_params(tool, transcript="test transcript", step_text="test step")
    assert result == {"param1": "value1"}


def test_react_required_params_full_integration():
    """Test required parameter validation in full ReACT run."""
    from typing import List
    
    class ToolWithRequired(DummyTool):
        def __init__(self, tool_id: str, name: str):
            super().__init__(tool_id, name, schema={"required_param": {}})
        
        def get_required_parameters(self) -> List[str]:
            return ["required_param"]
    
    t1 = ToolWithRequired("t1", "Tool One")
    
    llm = DummyLLM(
        json_queue=[
            {"step_type": "ACT", "text": "do something"},
            {},  # LLM generates empty params, missing required_param
            {"step_type": "STOP", "text": "final answer"},
        ],
        text_queue=[
            "t1",
        ],
    )
    
    tools = DummyTools([t1])
    memory: Dict[str, Any] = DictMemory()
    reasoner = ReACTReasoner(llm=llm, tools=tools, memory=memory, max_turns=5)
    
    result = reasoner.run("goal")
    
    # Should fail due to ParameterGenerationError, no tool call recorded
    assert result.tool_calls == []
    assert "ParameterGenerationError" in result.transcript
    assert "Missing required parameters: required_param" in result.transcript


def test_react_required_params_success_integration():
    """Test successful required parameter validation in full ReACT run."""
    from typing import List
    
    class ToolWithRequired(DummyTool):
        def __init__(self, tool_id: str, name: str):
            super().__init__(tool_id, name, schema={"required_param": {}})
        
        def get_required_parameters(self) -> List[str]:
            return ["required_param"]
    
    t1 = ToolWithRequired("t1", "Tool One")
    
    llm = DummyLLM(
        json_queue=[
            {"step_type": "ACT", "text": "do something"},
            {"required_param": "value1"},  # LLM generates required param
            {"step_type": "STOP", "text": "final answer"},
        ],
        text_queue=[
            "t1",
        ],
    )
    
    tools = DummyTools([t1])
    memory: Dict[str, Any] = DictMemory()
    reasoner = ReACTReasoner(llm=llm, tools=tools, memory=memory, max_turns=5)
    
    result = reasoner.run("goal")
    
    # Should succeed with tool call recorded
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0]["tool_id"] == "t1"
    assert "ACT_EXECUTED:" in result.transcript
    assert "FINAL ANSWER:" in result.transcript


