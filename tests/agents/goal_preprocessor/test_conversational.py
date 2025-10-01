from typing import Any, Dict, List
import pytest

from agents.goal_preprocessor.conversational import ConversationalGoalPreprocessor

from tests.conftest import DummyLLM


def _history() -> List[Dict[str, Any]]:
    return [
        {"goal": "g1", "result": "r1"},
        {"goal": "g2", "result": "r2"},
    ]


def test_conversational_with_empty_history():
    llm = DummyLLM(json_queue=[{"revised_goal": "revised"}])
    pre = ConversationalGoalPreprocessor(llm=llm)

    revised, question = pre.process("original", [])
    assert revised == "revised"
    assert question is None


def test_conversational_returns_revised_goal_when_present():
    llm = DummyLLM(json_queue=[{"revised_goal": "revised"}])
    pre = ConversationalGoalPreprocessor(llm=llm)

    revised, question = pre.process("original", _history())
    assert revised == "revised"
    assert question is None


def test_conversational_with_none_values_in_response():
    llm = DummyLLM(json_queue=[{"revised_goal": None, "clarification_question": None}])
    pre = ConversationalGoalPreprocessor(llm=llm)

    revised, question = pre.process("original", _history())
    assert revised == "original"
    assert question is None


def test_conversational_with_empty_string_values():
    llm = DummyLLM(json_queue=[{"revised_goal": "", "clarification_question": ""}])
    pre = ConversationalGoalPreprocessor(llm=llm)

    revised, question = pre.process("original", _history())
    assert revised == "original"
    assert question is None


def test_conversational_returns_clarification_when_present():
    llm = DummyLLM(json_queue=[{"clarification_question": "what do you mean?"}])
    pre = ConversationalGoalPreprocessor(llm=llm)

    revised, question = pre.process("original", _history())
    assert revised == "original"  # unchanged
    assert question == "what do you mean?"


def test_conversational_with_only_revised_goal_empty():
    llm = DummyLLM(json_queue=[{"revised_goal": "", "clarification_question": "what do you mean?"}])
    pre = ConversationalGoalPreprocessor(llm=llm)

    revised, question = pre.process("original", _history())
    assert revised == "original"
    assert question == "what do you mean?"


def test_conversational_with_only_clarification_empty():
    llm = DummyLLM(json_queue=[{"revised_goal": "revised", "clarification_question": ""}])
    pre = ConversationalGoalPreprocessor(llm=llm)

    revised, question = pre.process("original", _history())
    assert revised == "revised"
    assert question is None


def test_conversational_with_both_fields_present():
    llm = DummyLLM(json_queue=[{"revised_goal": "revised", "clarification_question": "what do you mean?"}])
    pre = ConversationalGoalPreprocessor(llm=llm)

    revised, question = pre.process("original", _history())
    assert revised == "revised"
    assert question is None


def test_conversational_falls_back_to_original_when_empty_json():
    llm = DummyLLM(json_queue=[{}])
    pre = ConversationalGoalPreprocessor(llm=llm)

    revised, question = pre.process("original", _history())
    assert revised == "original"
    assert question is None


def test_conversational_with_extra_fields_in_response():
    llm = DummyLLM(json_queue=[{
        "revised_goal": "revised",
        "clarification_question": "what do you mean?",
        "extra_field": "ignored",
        "another_field": 123
    }])
    pre = ConversationalGoalPreprocessor(llm=llm)

    revised, question = pre.process("original", _history())
    assert revised == "revised"
    assert question is None


def test_conversational_handles_missing_goal_or_result_in_history():
    history_missing_fields = [
        {"goal": "g1"},  # missing result
        {"result": "r2"},  # missing goal
        {"goal": "g3", "result": "r3"},  # complete
    ]

    llm = DummyLLM(json_queue=[{"revised_goal": "revised"}])
    pre = ConversationalGoalPreprocessor(llm=llm)

    with pytest.raises(KeyError):
        pre.process("original", history_missing_fields)


def test_conversational_with_complex_history_structure():
    complex_history = [
        {"goal": "complex goal with\nnewlines", "result": "result with \"quotes\""},
        {"goal": "", "result": "empty goal"},
    ]

    llm = DummyLLM(json_queue=[{"clarification_question": "clarify this"}])
    pre = ConversationalGoalPreprocessor(llm=llm)

    revised, question = pre.process("original", complex_history)
    assert revised == "original"
    assert question == "clarify this"


def test_conversational_with_malformed_history_entries():
    malformed_history = [
        {},  # completely empty entry
        {"wrong_key": "value"},  # wrong key
        {"goal": "g1", "result": "r1", "extra": "ignored"},  # extra field ignored
    ]

    llm = DummyLLM(json_queue=[{"revised_goal": "revised"}])
    pre = ConversationalGoalPreprocessor(llm=llm)

    with pytest.raises(KeyError):
        pre.process("original", malformed_history)


def test_conversational_with_non_string_goal_result_in_history():
    history_with_non_strings = [
        {"goal": 123, "result": 456},  # numbers
        {"goal": None, "result": None},  # None values
        {"goal": ["list"], "result": {"key": "value"}},  # complex types
    ]

    llm = DummyLLM(json_queue=[{"clarification_question": "what do you mean?"}])
    pre = ConversationalGoalPreprocessor(llm=llm)

    revised, question = pre.process("original", history_with_non_strings)
    assert revised == "original"
    assert question == "what do you mean?"


def test_conversational_with_very_long_history():
    long_history = [{"goal": f"goal_{i}", "result": f"result_{i}"} for i in range(100)]

    llm = DummyLLM(json_queue=[{"revised_goal": "what do you mean?"}])
    pre = ConversationalGoalPreprocessor(llm=llm)

    revised, question = pre.process("original", long_history)
    assert revised == "what do you mean?"
    assert question is None
