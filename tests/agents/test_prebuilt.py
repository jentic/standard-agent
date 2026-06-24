from typing import Any, Dict

import pytest

import agents.prebuilt as prebuilt


@pytest.fixture(autouse=True)
def clear_relevant_env(monkeypatch):
    for name in ("LLM_MODEL", "JENTIC_AGENT_API_KEY", "AWS_BEARER_TOKEN_BEDROCK"):
        monkeypatch.delenv(name, raising=False)


def _patch_prebuilt_dependencies(monkeypatch) -> Dict[str, Any]:
    captured: Dict[str, Any] = {}

    def fake_standard_agent_init(self, **kwargs):  # type: ignore[no-untyped-def]
        captured["standard_agent_kwargs"] = kwargs

    def fake_litellm(*, model: str | None = None):  # type: ignore[no-untyped-def]
        captured["lite_model"] = model
        return {"kind": "lite", "model": model}

    def fake_bedrock_llm(*, model: str | None = None):  # type: ignore[no-untyped-def]
        captured["bedrock_model"] = model
        return {"kind": "bedrock", "model": model}

    def fake_jentic_client():  # type: ignore[no-untyped-def]
        captured["jentic_created"] = True
        return {"kind": "tools"}

    def fake_memory():  # type: ignore[no-untyped-def]
        captured["memory_created"] = True
        return {"conversation_history": []}

    def fake_rewoo_reasoner(**kwargs):  # type: ignore[no-untyped-def]
        captured["rewoo_reasoner_kwargs"] = kwargs
        return {"kind": "rewoo_reasoner"}

    def fake_react_reasoner(**kwargs):  # type: ignore[no-untyped-def]
        captured["react_reasoner_kwargs"] = kwargs
        return {"kind": "react_reasoner"}

    def fake_goal_preprocessor(**kwargs):  # type: ignore[no-untyped-def]
        captured.setdefault("goal_preprocessor_kwargs", []).append(kwargs)
        return {"kind": "goal_preprocessor"}

    monkeypatch.setattr(prebuilt.StandardAgent, "__init__", fake_standard_agent_init)
    monkeypatch.setattr(prebuilt, "LiteLLM", fake_litellm)
    monkeypatch.setattr(prebuilt, "BedrockLLM", fake_bedrock_llm)
    monkeypatch.setattr(prebuilt, "JenticClient", fake_jentic_client)
    monkeypatch.setattr(prebuilt, "DictMemory", fake_memory)
    monkeypatch.setattr(prebuilt, "ReWOOReasoner", fake_rewoo_reasoner)
    monkeypatch.setattr(prebuilt, "ReACTReasoner", fake_react_reasoner)
    monkeypatch.setattr(prebuilt, "ConversationalGoalPreprocessor", fake_goal_preprocessor)

    return captured


def test_rewoo_agent_requires_jentic_api_key(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "gpt-4o")

    with pytest.raises(ValueError, match="JENTIC_AGENT_API_KEY"):
        prebuilt.ReWOOAgent()


def test_react_agent_requires_model_when_not_passed_or_configured(monkeypatch):
    monkeypatch.setenv("JENTIC_AGENT_API_KEY", "test-key")

    with pytest.raises(ValueError, match="LLM_MODEL"):
        prebuilt.ReACTAgent()


def test_react_agent_rejects_blank_model_without_llm_model_env(monkeypatch):
    monkeypatch.setenv("JENTIC_AGENT_API_KEY", "test-key")

    with pytest.raises(ValueError, match="LLM_MODEL"):
        prebuilt.ReACTAgent(model="   ")


def test_rewoo_agent_accepts_explicit_model_without_llm_model_env(monkeypatch):
    monkeypatch.setenv("JENTIC_AGENT_API_KEY", "test-key")
    captured = _patch_prebuilt_dependencies(monkeypatch)

    prebuilt.ReWOOAgent(model="gpt-4o-mini")

    assert captured["lite_model"] == "gpt-4o-mini"
    assert captured["jentic_created"] is True
    assert captured["memory_created"] is True
    assert captured["standard_agent_kwargs"]["llm"] == {"kind": "lite", "model": "gpt-4o-mini"}


def test_bedrock_agent_requires_bedrock_token(monkeypatch):
    monkeypatch.setenv("JENTIC_AGENT_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "us.anthropic.claude-3-5-haiku-20241022-v1:0")

    with pytest.raises(ValueError, match="AWS_BEARER_TOKEN_BEDROCK"):
        prebuilt.ReWOOAgentBedrock()


def test_bedrock_agent_accepts_explicit_model_with_required_env(monkeypatch):
    monkeypatch.setenv("JENTIC_AGENT_API_KEY", "test-key")
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "bedrock-token")
    captured = _patch_prebuilt_dependencies(monkeypatch)

    prebuilt.ReACTAgentBedrock(model="us.anthropic.claude-3-5-haiku-20241022-v1:0")

    assert captured["bedrock_model"] == "us.anthropic.claude-3-5-haiku-20241022-v1:0"
    assert captured["jentic_created"] is True
    assert captured["standard_agent_kwargs"]["llm"] == {
        "kind": "bedrock",
        "model": "us.anthropic.claude-3-5-haiku-20241022-v1:0",
    }
