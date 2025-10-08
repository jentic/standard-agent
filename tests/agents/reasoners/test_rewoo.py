from agents.memory.dict_memory import DictMemory
from agents.reasoner.rewoo import ReWOOReasoner, Step
from agents.reasoner.exceptions import ParameterGenerationError
from typing import Any, Dict, List
import pytest

from tests.conftest import DummyLLM, DummyTools, DummyTool, CaptureTools
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


def test_rewoo_param_gen_non_dict_raises_parameter_generation_error():
    # Directly exercise _generate_params: LLM returns a list -> should raise ParameterGenerationError
    llm = DummyLLM(
        text_queue=[],
        json_queue=[[["x"]]],  # non-dict JSON output from LLM
    )
    tools = DummyTools([DummyTool("t1", "Tool One", schema={"a": {}})])
    memory: Dict[str, Any] = DictMemory()
    reasoner = ReWOOReasoner(llm=llm, tools=tools, memory=memory)

    step = Step(text="act")
    tool = DummyTool("t1", "Tool One", schema={"a": {}})

    with pytest.raises(ParameterGenerationError) as exc:
        reasoner._generate_params(step, tool, inputs={})
    msg = str(exc.value)
    assert "Failed to generate valid JSON parameters for step" in msg
    assert "list' object has no attribute 'items" in msg


def test_rewoo_param_gen_non_dict_triggers_reflection():
    # Full run: param_gen returns list, which should be caught and trigger reflection (no crash)
    plan_text = "- act (output: k1)"
    llm = DummyLLM(
        text_queue=[
            plan_text,  # plan
            "TOOL",    # classify
            "t1",      # select
        ],
        json_queue=[
            [["x"]],                     # param_gen returns non-dict → error
            {"action": "give_up"},      # reflection decision
        ],
    )
    tools = DummyTools([DummyTool("t1", "Tool One", schema={"a": {}})])
    memory: Dict[str, Any] = DictMemory()
    reasoner = ReWOOReasoner(llm=llm, tools=tools, memory=memory, max_retries=2)
    result = reasoner.run("goal")

    assert result.tool_calls == []
    assert "Reflection decision" in result.transcript


def test_rewoo_reflector_suggested_params_non_dict_triggers_param_error():
    # First execution fails → reflection suggests retry_params with non-dict params (list)
    # Next attempt should raise ParameterGenerationError from suggestion branch and reflect again.
    plan_text = "- act (output: k1)"
    t1 = DummyTool("t1", "Tool One", schema={"a": {}})
    llm = DummyLLM(
        text_queue=[
            plan_text,  # plan
            "TOOL",    # classify step 1
            "t1",      # select
            "TOOL",    # classify retried step
        ],
        json_queue=[
            {},  # initial param_gen (dict) so we reach execute and fail
            {"action": "retry_params", "params": [["x"]]},  # reflection suggests non-dict params
            {"action": "give_up"},  # second reflection after ParameterGenerationError
        ],
    )
    tools = DummyTools([t1], failures={"t1": ToolExecutionError("boom", t1)})
    memory: Dict[str, Any] = DictMemory()
    reasoner = ReWOOReasoner(llm=llm, tools=tools, memory=memory, max_retries=2)

    result = reasoner.run("goal")

    # First attempt failed to execute; second attempt failed at param gen from suggestion; no successful tool calls
    assert result.tool_calls == []
    # Two reflections should have been recorded in the transcript (initial failure, then param error)
    assert result.transcript.count("Reflection decision") >= 2


def test_rewoo_plan_raises_on_duplicate_output_key():
    plan_text = "\n".join([
        "- first (output: k1)",
        "- second (output: k1)",
    ])
    llm = DummyLLM(text_queue=[plan_text])
    tools = DummyTools([])
    memory: Dict[str, Any] = DictMemory()
    reasoner = ReWOOReasoner(llm=llm, tools=tools, memory=memory)
    try:
        reasoner._plan("goal")
        assert False, "Expected ValueError for duplicate output key"
    except ValueError:
        pass


def test_rewoo_empty_plan_falls_back_to_goal_step():
    # Plan contains no recognizable bullet lines → fallback to a single goal step
    llm = DummyLLM(text_queue=["this is not a bullet list"])
    tools = DummyTools([])
    memory: Dict[str, Any] = DictMemory()
    reasoner = ReWOOReasoner(llm=llm, tools=tools, memory=memory)
    steps = reasoner._plan("my goal")
    assert len(steps) == 1
    assert steps[0].text == "my goal"


def test_rewoo_classify_reasoning_path_skips_tool_execution():
    # Plan: REASONING step should not execute any tool, but should update memory if output_key present
    plan_text = "- think (output: kx)"
    llm = DummyLLM(
        text_queue=[
            plan_text,   # plan
            "REASONING", # classify step
            "some reasoning result",  # reason result
        ],
    )
    tools = DummyTools([DummyTool("t1", "Tool One")])
    memory: Dict[str, Any] = DictMemory()
    reasoner = ReWOOReasoner(llm=llm, tools=tools, memory=memory)
    result = reasoner.run("goal")
    assert memory.get("kx") == "some reasoning result"
    assert result.tool_calls == []


def test_rewoo_selection_invalid_id_records_no_tool_call():
    # Selection returns an ID not in candidates
    plan_text = "- act (output: k1)"
    t1 = DummyTool("t1", "Tool One")
    llm = DummyLLM(
        text_queue=[
            plan_text,  # plan
            "TOOL",     # classify
            "t999",     # invalid selection
        ],
        json_queue=[{}],
    )
    tools = DummyTools([t1])
    memory: Dict[str, Any] = DictMemory()
    reasoner = ReWOOReasoner(llm=llm, tools=tools, memory=memory, max_retries=0)
    result = reasoner.run("goal")
    assert result.tool_calls == []


def test_rewoo_param_gen_error_triggers_reflection_and_no_tool_call():
    # Override LLM to raise a ValueError during param generation
    class FailParamLLM(DummyLLM):
        def __init__(self, *, text_queue: List[str] | None = None, json_queue: List[Dict[str, Any]] | None = None):
            super().__init__(text_queue=text_queue, json_queue=json_queue)
            self._raised_once = False

        def prompt_to_json(self, text: str, max_retries: int = 0):  # type: ignore[override]
            # Raise only on first call (param gen), allow subsequent reflection JSON to pass
            if not self._raised_once:
                self._raised_once = True
                raise ValueError("bad json")
            return super().prompt_to_json(text, max_retries=max_retries)

    plan_text = "- act (output: k1)"
    t1 = DummyTool("t1", "Tool One", schema={"a": {}})
    llm = FailParamLLM(
        text_queue=[
            plan_text,  # plan
            "TOOL",     # classify
            "t1",       # select
        ],
        json_queue=[
            {"action": "give_up"},  # reflection decision
        ],
    )
    tools = DummyTools([t1])
    memory: Dict[str, Any] = DictMemory()
    reasoner = ReWOOReasoner(llm=llm, tools=tools, memory=memory, max_retries=1)
    result = reasoner.run("goal")
    assert result.tool_calls == []


def test_rewoo_reflection_change_tool_is_honored_next_select():
    plan_text = "- do (output: k1)"
    t1 = DummyTool("t1", "Bad Tool", schema={})
    t2 = DummyTool("t2", "Good Tool", schema={})
    llm = DummyLLM(
        text_queue=[
            plan_text,  # plan
            "TOOL",     # classify
            "t1",       # initial selection (will fail)
            "TOOL",     # classify retried step
        ],
        json_queue=[
            {},  # param gen for first try
            {"action": "change_tool", "tool_id": "t2"},  # reflection
            {},  # param gen for second try
        ],
    )
    tools = DummyTools([t1, t2], failures={"t1": ToolExecutionError("boom", t1)})
    memory: Dict[str, Any] = DictMemory()
    reasoner = ReWOOReasoner(llm=llm, tools=tools, memory=memory, max_retries=1)
    result = reasoner.run("goal")
    # Success via t2 should be recorded (summary comes from JenticTool when using suggestion path)
    assert any(entry.get("tool_id") == "t2" and isinstance(entry.get("summary"), str) for entry in result.tool_calls)


def test_rewoo_reflection_retry_params_is_honored_next_param_gen():
    # First execution fails; reflection suggests retry with params, which must be filtered to schema keys
    plan_text = "- act (output: k1)"
    t1 = DummyTool("t1", "Tool One", schema={"a": {}, "b": {}})
    # First run fails; reflection asks to retry same tool with params including an extra key 'c'
    llm = DummyLLM(
        text_queue=[
            plan_text,  # plan
            "TOOL",     # classify
            "t1",       # select
            "TOOL",     # classify retried step
        ],
        json_queue=[
            {},  # initial param gen
            {"action": "retry_params", "params": {"a": 1, "b": 2, "c": 3}},  # reflection
        ],
    )
    tools = CaptureTools([t1])
    # Make first execution fail to trigger reflection, second succeed
    memory: Dict[str, Any] = DictMemory()
    reasoner = ReWOOReasoner(llm=llm, tools=tools, memory=memory, max_retries=1)

    # Monkey-patch execute to fail the first time then succeed
    original_execute = tools.execute
    call_count = {"n": 0}
    def flaky_execute(tool, params):  # type: ignore[override]
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise ToolExecutionError("first fail", t1)
        return original_execute(tool, params)
    tools.execute = flaky_execute  # type: ignore[assignment]

    result = reasoner.run("goal")
    assert tools.last_params == {"a": 1, "b": 2}
    assert {"tool_id": "t1", "summary": "Tool One"} in result.tool_calls


def test_rewoo_reflection_rephrase_step_updates_step_text():
    # After selection error, reflection rephrases the step; next classify handles the new text
    plan_text = "- do something (output: k1)"
    t1 = DummyTool("t1", "Tool One")
    llm = DummyLLM(
        text_queue=[
            plan_text,           # plan
            "TOOL",              # classify
            "none",              # selection error
            "REASONING",         # classify retried step (now reasoning)
            "rephrased success", # reasoning result
        ],
        json_queue=[
            {"action": "rephrase_step", "step": "rephrased step"},  # reflection
        ],
    )
    tools = DummyTools([t1])
    memory: Dict[str, Any] = DictMemory()
    reasoner = ReWOOReasoner(llm=llm, tools=tools, memory=memory, max_retries=1)
    result = reasoner.run("goal")
    assert "rephrased success" in result.transcript
    assert result.tool_calls == []


def test_rewoo_reflection_give_up_stops_retrying_step():
    # Selection error then give_up → step not requeued; no tool_calls
    plan_text = "- act (output: k1)"
    t1 = DummyTool("t1", "Tool One")
    llm = DummyLLM(
        text_queue=[
            plan_text,
            "TOOL",
            "none",
        ],
        json_queue=[
            {"action": "give_up"},
        ],
    )
    tools = DummyTools([t1])
    memory: Dict[str, Any] = DictMemory()
    reasoner = ReWOOReasoner(llm=llm, tools=tools, memory=memory, max_retries=1)
    result = reasoner.run("goal")
    assert result.tool_calls == []


def test_rewoo_missing_input_error_stops_processing_and_returns_partial():
    # Step1 fails and we give_up; Step2 expects k1 → MissingInputError handled and run stops
    plan_text = "\n".join([
        "- first (output: k1)",
        "- second (input: k1)",
    ])
    t1 = DummyTool("t1", "Tool One")
    llm = DummyLLM(
        text_queue=[
            plan_text,
            "TOOL",  # classify step1
            "t1",    # select
            "TOOL",  # classify step2 (won't reach execution due to missing k1)
        ],
        json_queue=[
            {},  # params for step1
            {"action": "give_up"},  # reflection after step1 failure
        ],
    )
    tools = DummyTools([t1], failures={"t1": ToolExecutionError("boom", t1)})
    memory: Dict[str, Any] = DictMemory()
    reasoner = ReWOOReasoner(llm=llm, tools=tools, memory=memory, max_retries=0)
    result = reasoner.run("goal")
    assert "Stopping: missing dependency" in result.transcript


def test_rewoo_tool_credentials_missing_logs_unauthorized_and_continues():
    from agents.tools.exceptions import ToolCredentialsMissingError
    plan_text = "- do (output: k1)"
    t1 = DummyTool("t1", "Tool One")
    llm = DummyLLM(
        text_queue=[
            plan_text,
            "TOOL",
            "t1",
        ],
        json_queue=[
            {"action": "give_up"},
        ],
    )
    tools = DummyTools([t1], failures={"t1": ToolCredentialsMissingError("Missing creds", t1)})
    memory: Dict[str, Any] = DictMemory()
    reasoner = ReWOOReasoner(llm=llm, tools=tools, memory=memory, max_retries=0)
    result = reasoner.run("goal")
    assert "Tool Unauthorized:" in result.transcript
    assert result.tool_calls == []


def test_rewoo_generate_params_with_required_params_success():
    """Test _generate_params succeeds when all required parameters are generated."""
    class ToolWithRequired(DummyTool):
        def __init__(self, tool_id: str, name: str):
            super().__init__(tool_id, name, schema={"param1": {}, "param2": {}})
        
        def get_required_parameter_keys(self) -> List[str]:
            return ["param1"]
    
    llm = DummyLLM(json_queue=[{"param1": "value1", "param2": "value2"}])
    tools = DummyTools([])
    memory: Dict[str, Any] = DictMemory()
    reasoner = ReWOOReasoner(llm=llm, tools=tools, memory=memory)
    
    step = Step(text="test step")
    tool = ToolWithRequired("t1", "Tool One")
    
    result = reasoner._generate_params(step, tool, inputs={})
    assert result == {"param1": "value1", "param2": "value2"}


def test_rewoo_generate_params_missing_required_params_raises_error():
    """Test _generate_params raises ParameterGenerationError when required parameters are missing."""
    class ToolWithRequired(DummyTool):
        def __init__(self, tool_id: str, name: str):
            super().__init__(tool_id, name, schema={"param1": {}, "param2": {}})
        
        def get_required_parameter_keys(self) -> List[str]:
            return ["param1", "param2"]
    
    llm = DummyLLM(json_queue=[{"param2": "value2"}])  # missing param1
    tools = DummyTools([])
    memory: Dict[str, Any] = DictMemory()
    reasoner = ReWOOReasoner(llm=llm, tools=tools, memory=memory)
    
    step = Step(text="test step")
    tool = ToolWithRequired("t1", "Tool One")
    
    with pytest.raises(ParameterGenerationError) as exc:
        reasoner._generate_params(step, tool, inputs={})
    
    error_msg = str(exc.value)
    assert "Missing required parameters: param1" in error_msg
    assert "Generated parameters: {'param2': 'value2'}" in error_msg
    assert "Tool 't1' requires these parameters" in error_msg


def test_rewoo_generate_params_no_required_params_backward_compatibility():
    """Test _generate_params works with tools that don't have get_required_parameter_keys method."""
    tool = DummyTool("t1", "Tool One", schema={"param1": {}})
    # DummyTool doesn't have get_required_parameter_keys method
    
    llm = DummyLLM(json_queue=[{"param1": "value1"}])
    tools = DummyTools([])
    memory: Dict[str, Any] = DictMemory()
    reasoner = ReWOOReasoner(llm=llm, tools=tools, memory=memory)
    
    step = Step(text="test step")
    
    result = reasoner._generate_params(step, tool, inputs={})
    assert result == {"param1": "value1"}


def test_rewoo_generate_params_empty_required_params():
    """Test _generate_params works when tool has empty required parameters list."""
    class ToolWithEmptyRequired(DummyTool):
        def __init__(self, tool_id: str, name: str):
            super().__init__(tool_id, name, schema={"param1": {}})
        
        def get_required_parameter_keys(self) -> List[str]:
            return []
    
    llm = DummyLLM(json_queue=[{"param1": "value1"}])
    tools = DummyTools([])
    memory: Dict[str, Any] = DictMemory()
    reasoner = ReWOOReasoner(llm=llm, tools=tools, memory=memory)
    
    step = Step(text="test step")
    tool = ToolWithEmptyRequired("t1", "Tool One")
    
    result = reasoner._generate_params(step, tool, inputs={})
    assert result == {"param1": "value1"}


def test_rewoo_generate_params_reflector_suggested_params_missing_required():
    """Test _generate_params validates reflector-suggested params for required fields."""
    class ToolWithRequired(DummyTool):
        def __init__(self, tool_id: str, name: str):
            super().__init__(tool_id, name, schema={"param1": {}, "param2": {}})
        
        def get_required_parameter_keys(self) -> List[str]:
            return ["param1", "param2"]
    
    llm = DummyLLM(json_queue=[])  # No LLM calls expected since using reflector params
    tools = DummyTools([])
    memory: Dict[str, Any] = DictMemory()
    reasoner = ReWOOReasoner(llm=llm, tools=tools, memory=memory)
    
    # Set up reflector suggestion with missing required param
    step = Step(text="test step")
    memory[f"rewoo_reflector_suggestion:{step.text}"] = {
        "action": "retry_params",
        "params": {"param2": "value2"}  # missing param1
    }
    
    tool = ToolWithRequired("t1", "Tool One")
    
    with pytest.raises(ParameterGenerationError) as exc:
        reasoner._generate_params(step, tool, inputs={})
    
    error_msg = str(exc.value)
    assert "Missing required parameters: param1" in error_msg
    assert "Generated parameters: {'param2': 'value2'}" in error_msg


def test_rewoo_generate_params_reflector_suggested_params_success():
    """Test _generate_params succeeds with valid reflector-suggested params."""
    class ToolWithRequired(DummyTool):
        def __init__(self, tool_id: str, name: str):
            super().__init__(tool_id, name, schema={"param1": {}, "param2": {}})
        
        def get_required_parameter_keys(self) -> List[str]:
            return ["param1"]
    
    llm = DummyLLM(json_queue=[])  # No LLM calls expected
    tools = DummyTools([])
    memory: Dict[str, Any] = DictMemory()
    reasoner = ReWOOReasoner(llm=llm, tools=tools, memory=memory)
    
    # Set up reflector suggestion with all required params
    step = Step(text="test step")
    memory[f"rewoo_reflector_suggestion:{step.text}"] = {
        "action": "retry_params", 
        "params": {"param1": "value1", "param2": "value2"}
    }
    
    tool = ToolWithRequired("t1", "Tool One")
    
    result = reasoner._generate_params(step, tool, inputs={})
    assert result == {"param1": "value1", "param2": "value2"}


def test_rewoo_generate_params_unknown_required_params_raises_error():
    """Test _generate_params raises ParameterGenerationError when LLM returns <UNKNOWN> for required params."""
    class ToolWithRequired(DummyTool):
        def __init__(self, tool_id: str, name: str):
            # Include both param1 (required) and param2 (optional) in schema so both pass filtering
            super().__init__(tool_id, name, schema={"param1": {}, "param2": {}})
        def get_required_parameters(self) -> List[str]:
            return ["param1"]
    
    llm = DummyLLM(json_queue=[{"param1": "<UNKNOWN>", "param2": "value2"}])
    tool = ToolWithRequired("t1", "Tool One")
    reasoner = ReWOOReasoner(llm=llm, tools=[tool], memory=DictMemory())
    step = Step(text="test step")
    
    with pytest.raises(ParameterGenerationError) as exc:
        reasoner._generate_params(step, tool, inputs={})
    
    error_msg = str(exc.value)
    assert "LLM indicated missing data using <UNKNOWN> for parameters: param1" in error_msg
    assert "Generated parameters: {'param1': '<UNKNOWN>', 'param2': 'value2'}" in error_msg
    assert "Tool 't1' requires these parameters" in error_msg


def test_rewoo_generate_params_combined_unknown_and_missing_params_raises_error():
    """Test _generate_params raises ParameterGenerationError with both unknown and missing required params."""
    class ToolWithRequired(DummyTool):
        def __init__(self, tool_id: str, name: str):
            # Only required params in schema; param4 will be filtered out
            super().__init__(tool_id, name, schema={"param1": {}, "param2": {}, "param3": {}})
        def get_required_parameter_keys(self) -> List[str]:
            return ["param1", "param2", "param3"]
    
    llm = DummyLLM(json_queue=[{"param1": "<UNKNOWN>", "param4": "value4"}])  # param1 is unknown, param2&3 missing
    tool = ToolWithRequired("t1", "Tool One")
    reasoner = ReWOOReasoner(llm=llm, tools=[tool], memory=DictMemory())
    step = Step(text="test step")
    
    with pytest.raises(ParameterGenerationError) as exc:
        reasoner._generate_params(step, tool, inputs={})
    
    error_msg = str(exc.value)
    assert "LLM indicated missing data using <UNKNOWN> for parameters: param1" in error_msg
    assert "Missing required parameters: param2, param3" in error_msg
    assert "Generated parameters: {'param1': '<UNKNOWN>'}" in error_msg  # param4 filtered out
    assert "Tool 't1' requires these parameters" in error_msg


def test_rewoo_required_params_full_integration():
    """Test required parameter validation in full ReWOO run."""
    class ToolWithRequired(DummyTool):
        def __init__(self, tool_id: str, name: str):
            super().__init__(tool_id, name, schema={"required_param": {}})
        
        def get_required_parameter_keys(self) -> List[str]:
            return ["required_param"]
    
    plan_text = "- do something (output: k1)"
    t1 = ToolWithRequired("t1", "Tool One")
    
    llm = DummyLLM(
        text_queue=[
            plan_text,
            "TOOL",
            "t1",
        ],
        json_queue=[
            {},  # LLM generates empty params, missing required_param
            {"action": "give_up"},  # reflection gives up
        ],
    )
    
    tools = DummyTools([t1])
    memory: Dict[str, Any] = DictMemory()
    reasoner = ReWOOReasoner(llm=llm, tools=tools, memory=memory, max_retries=1)
    
    result = reasoner.run("goal")
    
    # Should trigger reflection due to ParameterGenerationError
    assert result.tool_calls == []
    assert "Reflection decision" in result.transcript


def test_rewoo_generate_params_multiple_schemas():
    """Test _generate_params with tool that has multiple schemas."""
    class ToolWithMultipleSchemas(DummyTool):
        def __init__(self, tool_id: str, name: str):
            super().__init__(tool_id, name, schema=None)
        
        def get_parameter_schema(self):
            return [
                {"param1": {"type": "string"}, "param2": {"type": "integer"}},
                {"param3": {"type": "boolean"}, "param4": {"type": "number"}}
            ]
        
        def get_parameter_keys(self):
            return ["param1", "param2", "param3", "param4"]
    
    llm = DummyLLM(json_queue=[{"param1": "value1", "param3": True, "extra": "ignored"}])
    tools = DummyTools([])
    memory: Dict[str, Any] = DictMemory()
    reasoner = ReWOOReasoner(llm=llm, tools=tools, memory=memory)
    
    tool = ToolWithMultipleSchemas("t1", "Tool One")
    inputs = {"some_input": "data"}
    
    result = reasoner._generate_params(Step("test step", "k1", []), tool, inputs)
    # Should filter out "extra" key since it's not in allowed_keys
    assert result == {"param1": "value1", "param3": True}


def test_rewoo_generate_params_empty_schema():
    """Test _generate_params with tool that has empty schema."""
    class ToolWithEmptySchema(DummyTool):
        def __init__(self, tool_id: str, name: str):
            super().__init__(tool_id, name, schema=None)
        
        def get_parameter_schema(self):
            return None
        
        def get_parameter_keys(self):
            return []
    
    llm = DummyLLM(json_queue=[{"param1": "value1", "param2": "value2"}])
    tools = DummyTools([])
    memory: Dict[str, Any] = DictMemory()
    reasoner = ReWOOReasoner(llm=llm, tools=tools, memory=memory)
    
    tool = ToolWithEmptySchema("t1", "Tool One")
    inputs = {"some_input": "data"}
    
    result = reasoner._generate_params(Step("test step", "k1", []), tool, inputs)
    # Should return empty dict since no allowed keys
    assert result == {}


def test_rewoo_generate_params_fallback_to_schema_keys():
    """Test _generate_params falls back to schema.keys() when get_parameter_keys is not available."""
    class ToolWithoutAllowedKeysMethod(DummyTool):
        def __init__(self, tool_id: str, name: str):
            super().__init__(tool_id, name, schema={"param1": {"type": "string"}, "param2": {"type": "integer"}})
        
        def get_parameter_schema(self):
            return {"param1": {"type": "string"}, "param2": {"type": "integer"}}
        
        # Note: no get_parameter_keys method - should fall back to schema.keys()
    
    llm = DummyLLM(json_queue=[{"param1": "value1", "param2": 42, "extra": "ignored"}])
    tools = DummyTools([])
    memory: Dict[str, Any] = DictMemory()
    reasoner = ReWOOReasoner(llm=llm, tools=tools, memory=memory)
    
    tool = ToolWithoutAllowedKeysMethod("t1", "Tool One")
    inputs = {"some_input": "data"}
    
    result = reasoner._generate_params(Step("test step", "k1", []), tool, inputs)
    # Should filter out "extra" key since it's not in schema.keys()
    assert result == {"param1": "value1", "param2": 42}


def test_rewoo_generate_params_multiple_schemas_union_keys():
    """Test _generate_params with multiple schemas returns union of all keys."""
    class ToolWithOverlappingSchemas(DummyTool):
        def __init__(self, tool_id: str, name: str):
            super().__init__(tool_id, name, schema=None)
        
        def get_parameter_schema(self):
            return [
                {"param1": {"type": "string"}, "param2": {"type": "integer"}},
                {"param2": {"type": "string"}, "param3": {"type": "boolean"}}  # param2 overlaps
            ]
        
        def get_parameter_keys(self):
            return ["param1", "param2", "param3"]  # Union of all keys
    
    llm = DummyLLM(json_queue=[{"param1": "value1", "param2": 42, "param3": True}])
    tools = DummyTools([])
    memory: Dict[str, Any] = DictMemory()
    reasoner = ReWOOReasoner(llm=llm, tools=tools, memory=memory)
    
    tool = ToolWithOverlappingSchemas("t1", "Tool One")
    inputs = {"some_input": "data"}
    
    result = reasoner._generate_params(Step("test step", "k1", []), tool, inputs)
    assert result == {"param1": "value1", "param2": 42, "param3": True}


def test_rewoo_generate_params_reflector_suggestion_with_multiple_schemas():
    """Test _generate_params with reflector suggestion and multiple schemas filtering."""
    class ToolWithMultipleSchemas(DummyTool):
        def __init__(self, tool_id: str, name: str):
            super().__init__(tool_id, name, schema=None)
        
        def get_parameter_schema(self):
            return [
                {"param1": {"type": "string"}, "param2": {"type": "integer"}},
                {"param3": {"type": "boolean"}, "param4": {"type": "number"}}
            ]
        
        def get_parameter_keys(self):
            return ["param1", "param2", "param3", "param4"]
    
    llm = DummyLLM(json_queue=[])  # No LLM calls needed since we're using reflector suggestion
    tools = DummyTools([])
    memory: Dict[str, Any] = DictMemory()
    reasoner = ReWOOReasoner(llm=llm, tools=tools, memory=memory)
    
    # Set up reflector suggestion with extra keys
    memory["rewoo_reflector_suggestion:test step"] = {
        "action": "retry_params",
        "params": {"param1": "value1", "param3": True, "extra": "ignored", "invalid": "also_ignored"}
    }
    
    tool = ToolWithMultipleSchemas("t1", "Tool One")
    inputs = {"some_input": "data"}
    
    result = reasoner._generate_params(Step("test step", "k1", []), tool, inputs)
    # Should filter out "extra" and "invalid" keys since they're not in allowed_keys
    assert result == {"param1": "value1", "param3": True}


