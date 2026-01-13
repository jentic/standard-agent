from typing import Any, Dict, List
import pytest
import re

from agents.goal_preprocessor.conversational import ConversationalGoalPreprocessor
from agents.memory.dict_memory import DictMemory
from datetime import datetime
from zoneinfo import ZoneInfo

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
    tz = ZoneInfo("UTC")
    pre = ConversationalGoalPreprocessor(llm=llm)

    revised, question = pre.process("original", _history())
    assert revised == "original"
    assert question is None

    # No normalization performed now; goal remains unchanged when no revision


def test_conversational_returns_utc_offset_label_when_no_iana():
    """Test that preprocessor returns UTC offset label when no IANA in memory."""
    llm = DummyLLM(json_queue=[{}])
    memory = DictMemory()
    # No timezone in context
    pre = ConversationalGoalPreprocessor(llm=llm, memory=memory)

    now, label = pre._current_time_and_timezone()

    # Label should be in UTC offset format
    assert label.startswith("UTC")
    assert ":" in label  # Should have format UTC+HH:MM or UTC-HH:MM
    # Validate format matches UTC±HH:MM
    assert re.match(r'^UTC[+-]\d{2}:\d{2}$', label), f"Expected UTC±HH:MM format, got {label}"
    # Time should be timezone-aware
    assert now.tzinfo is not None


def test_conversational_with_iana_returns_utc_offset_label():
    """Test that preprocessor returns UTC offset label even when IANA is provided."""
    llm = DummyLLM(json_queue=[{}])
    memory = DictMemory()
    memory["context"] = {"timezone": "Europe/Dublin"}  # UTC+05:30
    pre = ConversationalGoalPreprocessor(llm=llm, memory=memory)

    now, label = pre._current_time_and_timezone()

    # Should return UTC offset label, not the IANA name
    assert label == "UTC+01:00"
    assert now.tzinfo is not None
    # Verify time is computed with correct timezone
    assert now.tzinfo.key == "Europe/Dublin"  # type: ignore


def test_conversational_invalid_iana_falls_back(caplog):
    """Test that invalid IANA in memory logs warning and falls back to system tz."""
    import logging
    llm = DummyLLM(json_queue=[{}])
    memory = DictMemory()
    memory["context"] = {"timezone": "Invalid/Timezone"}
    pre = ConversationalGoalPreprocessor(llm=llm, memory=memory)

    with caplog.at_level(logging.WARNING):
        now, label = pre._current_time_and_timezone()

    # Should fall back to system timezone and return UTC offset label
    assert label.startswith("UTC")
    assert re.match(r'^UTC[+-]\d{2}:\d{2}$', label)
    assert now.tzinfo is not None


def test_conversational_utc_offset_label_helper():
    """Test the _utc_offset_label static method formats correctly."""
    # UTC
    dt_utc = datetime(2025, 1, 15, 12, 0, tzinfo=ZoneInfo("UTC"))
    assert ConversationalGoalPreprocessor._utc_offset_label(dt_utc) == "UTC+00:00"

    # Positive offset with minutes
    dt_kolkata = datetime(2025, 1, 15, 12, 0, tzinfo=ZoneInfo("Asia/Kolkata"))
    assert ConversationalGoalPreprocessor._utc_offset_label(dt_kolkata) == "UTC+05:30"

    # Negative offset
    dt_ny = datetime(2025, 1, 15, 12, 0, tzinfo=ZoneInfo("America/New_York"))
    assert ConversationalGoalPreprocessor._utc_offset_label(dt_ny) == "UTC-05:00"

    # DST affects offset (summer in NY is UTC-04:00)
    dt_ny_summer = datetime(2025, 7, 15, 12, 0, tzinfo=ZoneInfo("America/New_York"))
    assert ConversationalGoalPreprocessor._utc_offset_label(dt_ny_summer) == "UTC-04:00"


def test_conversational_passes_correct_timezone_info_to_prompt():
    """Test that process() passes now_iso, timezone_name (as UTC offset), and weekday to the prompt."""
    class CapturingLLM:
        def __init__(self):
            self.last_prompt = None

        def prompt_to_json(self, prompt: str, **kwargs):
            self.last_prompt = prompt
            return {}  # Return empty to avoid revision

    llm = CapturingLLM()
    memory = DictMemory()
    memory["context"] = {"timezone": "America/New_York"}
    pre = ConversationalGoalPreprocessor(llm=llm, memory=memory)

    revised, question = pre.process("what is the time?", _history())

    # Verify prompt was called
    assert llm.last_prompt is not None

    # Check that timezone_name is UTC offset format, not IANA
    assert "UTC-0" in llm.last_prompt  # Either UTC-04:00 or UTC-05:00 depending on DST
    assert "America/New_York" not in llm.last_prompt

    # Check that now_iso (ISO format datetime) is present
    # ISO format includes 'T' and timezone offset
    assert "T" in llm.last_prompt  # ISO datetime has T separator

    # Check that weekday is present (should be a day name)
    weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    assert any(day in llm.last_prompt for day in weekdays)


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
