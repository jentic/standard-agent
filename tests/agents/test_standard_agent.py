from typing import Any, Deque, Dict, List, Optional, Tuple

import pytest
import json

from agents.standard_agent import StandardAgent, AgentState
from agents.reasoner.base import BaseReasoner, ReasoningResult
from agents.goal_preprocessor.base import BaseGoalPreprocessor
from agents.memory.dict_memory import DictMemory

from tests.conftest import DummyLLM, DummyTools


class DummyReasoner(BaseReasoner):
    def __init__(self):
        # type: ignore[call-arg]
        pass

    def run(self, goal: str) -> ReasoningResult:  # type: ignore[override]
        return ReasoningResult(transcript=f"Goal was: {goal}", success=True, iterations=1)


class CapturingReasoner(BaseReasoner):
    def __init__(self):
        self.last_goal: Optional[str] = None

    def run(self, goal: str) -> ReasoningResult:  # type: ignore[override]
        self.last_goal = goal
        return ReasoningResult(transcript="trace", success=True)


class FailingReasoner(BaseReasoner):
    def __init__(self):
        # type: ignore[call-arg]
        pass

    def run(self, goal: str) -> ReasoningResult:  # type: ignore[override]
        raise RuntimeError("boom")


class PassThroughPreprocessor(BaseGoalPreprocessor):
    def process(self, goal: str, history):  # type: ignore[override]
        return goal, None


class RevisingPreprocessor(BaseGoalPreprocessor):
    def __init__(self, revised: str):
        self.revised = revised

    def process(self, goal: str, history):  # type: ignore[override]
        return self.revised, None


class InterventionPreprocessor(BaseGoalPreprocessor):
    def __init__(self, message: str):
        self.message = message

    def process(self, goal: str, history):  # type: ignore[override]
        return goal, self.message


def _fixed_uuid4(monkeypatch, value: str) -> None:
    class _U:
        def __init__(self, hx: str):
            self.hex = hx

    import agents.standard_agent as standard_agent

    monkeypatch.setattr(standard_agent, "uuid4", lambda: _U(value))


def test_agent_solve_sets_final_answer_from_summarizer_and_records_history(monkeypatch):
    _fixed_uuid4(monkeypatch, "RUN123")
    llm = DummyLLM(text_queue=["SUMMARIZED"])
    tools = DummyTools()
    memory: Dict[str, Any] = DictMemory()
    reasoner = DummyReasoner()

    agent = StandardAgent(llm=llm, tools=tools, memory=memory, reasoner=reasoner)

    result = agent.solve("find answer")

    assert result.final_answer == "SUMMARIZED"
    assert agent.state == AgentState.READY

    # Conversation history updated
    hist = memory.get("conversation_history")
    assert hist and len(hist) == 1
    assert hist[-1]["goal"] == "find answer"
    assert hist[-1]["result"] == "SUMMARIZED"


def test_agent_uses_goal_preprocessor_and_returns_intervention_message(monkeypatch):
    _fixed_uuid4(monkeypatch, "RUNINT")
    llm = DummyLLM(text_queue=["UNUSED"])
    tools = DummyTools()
    memory: Dict[str, Any] = DictMemory()
    reasoner = DummyReasoner()
    pre = InterventionPreprocessor("Please clarify your request.")

    agent = StandardAgent(llm=llm, tools=tools, memory=memory, reasoner=reasoner, goal_preprocessor=pre)
    result = agent.solve("ambiguous goal")

    assert result.success is False
    assert result.final_answer == "Please clarify your request."
    # Early return path: no run id keys set
    assert memory.get("goal:RUNINT") is None
    assert memory.get("result:RUNINT") is None
    # Conversation history contains intervention note; state remains READY
    hist = memory.get("conversation_history")
    assert hist and "user intervention message" in hist[-1]["result"]
    assert agent.state == AgentState.READY


def test_agent_passes_revised_goal_to_reasoner(monkeypatch):
    _fixed_uuid4(monkeypatch, "RUNREV")
    llm = DummyLLM(text_queue=["OK"])
    tools = DummyTools()
    memory: Dict[str, Any] = DictMemory()
    reasoner = CapturingReasoner()
    pre = RevisingPreprocessor("revised goal")

    agent = StandardAgent(llm=llm, tools=tools, memory=memory, reasoner=reasoner, goal_preprocessor=pre)
    agent.solve("original goal")

    assert reasoner.last_goal == "revised goal"


def test_agent_preserves_reasoner_fields(monkeypatch):
    class FixedReasoner(BaseReasoner):
        def __init__(self):
            # type: ignore[call-arg]
            pass

        def run(self, goal: str) -> ReasoningResult:  # type: ignore[override]
            return ReasoningResult(
                transcript="t",
                iterations=7,
                tool_calls=[{"tool_id": "x", "summary": "X"}],
                success=False,
            )

    _fixed_uuid4(monkeypatch, "RUNPRESERVE")
    llm = DummyLLM(text_queue=["S"])  # summarizer output
    tools = DummyTools()
    memory: Dict[str, Any] = DictMemory()
    reasoner = FixedReasoner()

    agent = StandardAgent(llm=llm, tools=tools, memory=memory, reasoner=reasoner)
    result = agent.solve("g")

    assert result.iterations == 7
    assert result.tool_calls == [{"tool_id": "x", "summary": "X"}]
    assert result.success is False
    assert result.final_answer == "S"


def test_agent_conversation_history_respects_window(monkeypatch):
    class SmallReasoner(BaseReasoner):
        def __init__(self):
            # type: ignore[call-arg]
            pass

        def run(self, goal: str) -> ReasoningResult:  # type: ignore[override]
            return ReasoningResult(transcript="T", success=True)

    _fixed_uuid4(monkeypatch, "RUNWIN1")
    llm = DummyLLM(text_queue=["A", "B"])  # two summaries
    tools = DummyTools()
    memory: Dict[str, Any] = DictMemory()
    reasoner = SmallReasoner()

    agent = StandardAgent(llm=llm, tools=tools, memory=memory, reasoner=reasoner, conversation_history_window=1)
    agent.solve("first")
    agent.solve("second")

    hist = memory.get("conversation_history")
    assert hist and len(hist) == 1
    assert hist[-1]["goal"] == "second"


def test_agent_sets_needs_attention_and_reraises_on_error(monkeypatch):
    _fixed_uuid4(monkeypatch, "RUNERR")
    llm = DummyLLM(text_queue=["UNUSED"])
    tools = DummyTools()
    memory: Dict[str, Any] = DictMemory()
    reasoner = FailingReasoner()

    agent = StandardAgent(llm=llm, tools=tools, memory=memory, reasoner=reasoner)

    assert agent.state == AgentState.READY
    with pytest.raises(RuntimeError):
        agent.solve("goal")
    assert agent.state == AgentState.NEEDS_ATTENTION
    # No result persisted on failure
    assert not any(k.startswith("result:") for k in memory.keys())


def test_agent_initial_state_is_ready():
    llm = DummyLLM()
    tools = DummyTools()
    memory: Dict[str, Any] = DictMemory()
    reasoner = DummyReasoner()

    agent = StandardAgent(llm=llm, tools=tools, memory=memory, reasoner=reasoner)

    assert agent.state == AgentState.READY



def test_agent_memory_is_json_serializable(monkeypatch):
    llm = DummyLLM(text_queue=["S1", "S2"])
    tools = DummyTools()
    memory: Dict[str, Any] = DictMemory()
    agent = StandardAgent(llm=llm, tools=tools, memory=memory, reasoner=DummyReasoner(), conversation_history_window=2)

    agent.solve("g1")
    agent.solve("g2")

    dumped = json.dumps(agent.memory)
    assert isinstance(dumped, str)
    assert isinstance(agent.memory.get("conversation_history"), list)
    assert len(agent.memory["conversation_history"]) == 2


def test_agent_conversation_history_disabled_window():
    llm = DummyLLM(text_queue=["S"])  # summarizer output
    tools = DummyTools()
    memory: Dict[str, Any] = DictMemory()
    agent = StandardAgent(llm=llm, tools=tools, memory=memory, reasoner=DummyReasoner(), conversation_history_window=0)

    agent.solve("g1")

    assert memory.get("conversation_history") == []
    
def test_solve_raises_error_for_none_goal():
    """Test that solve raises ValueError when goal is None"""
    agent = StandardAgent(
        llm=DummyLLM(),
        tools=DummyTools(),
        memory=DictMemory(),
        reasoner=DummyReasoner()
    )
    
    with pytest.raises(ValueError, match="Goal cannot be None"):
        agent.solve(None)


def test_solve_raises_error_for_empty_goal():
    """Test that solve raises ValueError when goal is empty string"""
    agent = StandardAgent(
        llm=DummyLLM(),
        tools=DummyTools(), 
        memory=DictMemory(),
        reasoner=DummyReasoner()
    )
    
    with pytest.raises(ValueError, match="Goal cannot be empty or whitespace"):
        agent.solve("")


def test_solve_raises_error_for_whitespace_goal():
    """Test that solve raises ValueError when goal is only whitespace"""
    agent = StandardAgent(
        llm=DummyLLM(),
        tools=DummyTools(),
        memory=DictMemory(), 
        reasoner=DummyReasoner()
    )
    
    with pytest.raises(ValueError, match="Goal cannot be empty or whitespace"):
        agent.solve("   ")

