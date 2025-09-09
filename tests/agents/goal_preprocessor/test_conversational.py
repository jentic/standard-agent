from typing import Any, Dict, List

from agents.goal_preprocessor.conversational import ConversationalGoalPreprocessor
from datetime import datetime
from zoneinfo import ZoneInfo

from tests.conftest import DummyLLM


def _history() -> List[Dict[str, Any]]:
    return [
        {"goal": "g1", "result": "r1"},
        {"goal": "g2", "result": "r2"},
    ]


def test_conversational_returns_revised_goal_when_present():
    llm = DummyLLM(json_queue=[{"revised_goal": "revised"}])
    pre = ConversationalGoalPreprocessor(llm=llm)

    revised, question = pre.process("original", _history())
    assert revised == "revised"
    assert question is None


def test_conversational_returns_clarification_when_present():
    llm = DummyLLM(json_queue=[{"clarification_question": "what do you mean?"}])
    pre = ConversationalGoalPreprocessor(llm=llm)

    revised, question = pre.process("original", _history())
    assert revised == "original"  # unchanged
    assert question == "what do you mean?"


def test_conversational_falls_back_to_original_when_empty_json():
    llm = DummyLLM(json_queue=[{}])
    tz = ZoneInfo("UTC")
    pre = ConversationalGoalPreprocessor(llm=llm, now_fn=lambda: datetime(2025, 1, 15, 12, 0, tzinfo=tz))

    revised, question = pre.process("original", _history())
    assert revised == "original"
    assert question is None

    # No normalization performed now; goal remains unchanged when no revision


