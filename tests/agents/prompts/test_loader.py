import pytest

from agents.prompts import load_prompts


def test_load_prompts_succeeds_for_agent_profile_when_required_key_present():
    prompts = load_prompts("agent", required_prompts=["summarize"])
    assert isinstance(prompts, dict)
    assert isinstance(prompts["summarize"], str)
    assert prompts["summarize"].strip() != ""


def test_load_prompts_succeeds_for_nested_profile_paths_react():
    required = [
        "think",
        "tool_select",
        "param_gen",
    ]
    prompts = load_prompts("reasoners/react", required_prompts=required)
    assert isinstance(prompts, dict)
    assert set(required).issubset(prompts.keys())
    for k in required:
        assert isinstance(prompts[k], str)
        assert prompts[k].strip() != ""


def test_load_prompts_succeeds_for_nested_profile_paths_rewoo():
    required = [
        "plan",
        "classify_step",
        "reason",
        "tool_select",
        "param_gen",
        "reflect",
        "reflect_alternatives",
    ]
    prompts = load_prompts("reasoners/rewoo", required_prompts=required)
    assert isinstance(prompts, dict)
    assert set(required).issubset(prompts.keys())
    for k in required:
        assert isinstance(prompts[k], str)
        assert prompts[k].strip() != ""


def test_load_prompts_raises_file_not_found_for_missing_profile_yaml():
    with pytest.raises(FileNotFoundError):
        load_prompts("reasoners/does_not_exist", required_prompts=["x"])  # type: ignore[arg-type]


def test_load_prompts_raises_key_error_when_required_key_missing():
    with pytest.raises(KeyError):
        load_prompts("agent", required_prompts=["does_not_exist"])  # type: ignore[arg-type]


def test_load_prompts_raises_type_error_when_yaml_root_is_not_mapping():
    with pytest.raises(TypeError):
        # Path is relative to agents/prompts package directory
        load_prompts("../../tests/agents/prompts/testdata/bad_root", required_prompts=["x"])  # type: ignore[arg-type]


def test_load_prompts_raises_key_error_when_required_value_is_empty_string():
    with pytest.raises(KeyError):
        # Path is relative to agents/prompts package directory
        load_prompts("../../tests/agents/prompts/testdata/empty_value", required_prompts=["x"])  # type: ignore[arg-type]


