"""Tests for prebuilt agent validation logic."""
import os
import pytest
from unittest.mock import patch
from agents.prebuilt import (
    ReWOOAgent,
    ReACTAgent,
    ReWOOAgentBedrock,
    ReACTAgentBedrock,
    _validate_litellm_environment,
    _validate_bedrock_environment,
    _validate_jentic_environment,
)


class TestLiteLLMValidation:
    """Test validation for LiteLLM-based agents."""

    def test_validate_litellm_with_openai_model_and_key(self):
        """Test validation passes when OpenAI model and API key are present."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=False):
            # Should not raise
            _validate_litellm_environment("gpt-4")

    def test_validate_litellm_with_anthropic_model_and_key(self):
        """Test validation passes when Anthropic model and API key are present."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}, clear=False):
            # Should not raise
            _validate_litellm_environment("claude-3-opus")

    def test_validate_litellm_with_gemini_model_and_key(self):
        """Test validation passes when Gemini model and API key are present."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}, clear=False):
            # Should not raise
            _validate_litellm_environment("gemini-pro")

    def test_validate_litellm_with_unknown_model_but_common_key(self):
        """Test validation passes when model is unknown but a common API key is present."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=False):
            # Should not raise even with unknown model
            _validate_litellm_environment("some-unknown-model")

    def test_validate_litellm_with_no_model_specified(self):
        """Test validation with no model specified (relies on LLM_MODEL env var)."""
        with patch.dict(os.environ, {"LLM_MODEL": "gpt-4", "OPENAI_API_KEY": "test-key"}, clear=False):
            # Should not raise
            _validate_litellm_environment(None)

    def test_validate_litellm_with_no_model_and_no_env_var(self):
        """Test validation when no model is specified and no LLM_MODEL env var."""
        # Clear environment
        with patch.dict(os.environ, {}, clear=True):
            # Should not raise - BaseLLM will handle this case
            _validate_litellm_environment(None)

    def test_validate_litellm_missing_openai_key(self):
        """Test validation fails when OpenAI model is specified but API key is missing."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="Missing required environment variables for model 'gpt-4'"):
                _validate_litellm_environment("gpt-4")

    def test_validate_litellm_missing_anthropic_key(self):
        """Test validation fails when Anthropic model is specified but API key is missing."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="Missing required environment variables for model 'claude-3-opus'"):
                _validate_litellm_environment("claude-3-opus")

    def test_validate_litellm_unknown_model_no_common_keys(self):
        """Test validation fails when unknown model and no common API keys are present."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="No API key found for model 'unknown-model'"):
                _validate_litellm_environment("unknown-model")

    def test_validate_litellm_multiple_provider_keys_work(self):
        """Test that having multiple provider keys allows any model to work."""
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "openai-key",
            "ANTHROPIC_API_KEY": "anthropic-key"
        }, clear=False):
            # Should work with either provider
            _validate_litellm_environment("gpt-4")
            _validate_litellm_environment("claude-3-opus")
            _validate_litellm_environment("some-unknown-model")


class TestBedrockValidation:
    """Test validation for Bedrock-based agents."""

    def test_validate_bedrock_with_access_keys(self):
        """Test validation passes when AWS access keys are present."""
        with patch.dict(os.environ, {
            "AWS_ACCESS_KEY_ID": "test-access-key",
            "AWS_SECRET_ACCESS_KEY": "test-secret-key"
        }, clear=False):
            # Should not raise
            _validate_bedrock_environment()

    def test_validate_bedrock_with_bearer_token(self):
        """Test validation passes when AWS bearer token is present."""
        with patch.dict(os.environ, {"AWS_BEARER_TOKEN_BEDROCK": "test-bearer-token"}, clear=False):
            # Should not raise
            _validate_bedrock_environment()

    def test_validate_bedrock_with_profile(self):
        """Test validation passes when AWS profile is present."""
        with patch.dict(os.environ, {"AWS_PROFILE": "test-profile"}, clear=False):
            # Should not raise
            _validate_bedrock_environment()

    def test_validate_bedrock_missing_credentials(self):
        """Test validation fails when no AWS credentials are present."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="Missing AWS credentials for Bedrock"):
                _validate_bedrock_environment()

    def test_validate_bedrock_partial_access_keys(self):
        """Test validation fails when only one AWS access key is present."""
        with patch.dict(os.environ, {"AWS_ACCESS_KEY_ID": "test-key"}, clear=True):
            with pytest.raises(ValueError, match="Missing AWS credentials for Bedrock"):
                _validate_bedrock_environment()


class TestJenticValidation:
    """Test validation for Jentic tools."""

    def test_validate_jentic_with_api_key(self):
        """Test validation passes when Jentic API key is present."""
        with patch.dict(os.environ, {"JENTIC_AGENT_API_KEY": "test-jentic-key"}, clear=False):
            # Should not raise
            _validate_jentic_environment()

    def test_validate_jentic_missing_api_key(self):
        """Test validation fails when Jentic API key is missing."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="Missing JENTIC_AGENT_API_KEY environment variable"):
                _validate_jentic_environment()


class TestReWOOAgentValidation:
    """Test validation during ReWOO agent creation."""

    def test_rewoo_agent_creation_with_valid_openai_env(self):
        """Test ReWOO agent can be created with valid OpenAI environment."""
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "test-key",
            "JENTIC_AGENT_API_KEY": "test-jentic-key"
        }, clear=False):
            # Should not raise during initialization
            agent = ReWOOAgent(model="gpt-4")
            assert agent is not None

    def test_rewoo_agent_creation_missing_env_vars(self):
        """Test ReWOO agent creation fails with missing environment variables."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="Missing required environment variables"):
                ReWOOAgent(model="gpt-4")

    def test_rewoo_agent_creation_missing_jentic_key(self):
        """Test ReWOO agent creation fails with missing Jentic API key."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True):
            with pytest.raises(ValueError, match="Missing JENTIC_AGENT_API_KEY"):
                ReWOOAgent(model="gpt-4")

    def test_rewoo_agent_bedrock_creation_with_valid_env(self):
        """Test ReWOO Bedrock agent can be created with valid AWS environment."""
        with patch.dict(os.environ, {
            "AWS_ACCESS_KEY_ID": "test-key",
            "AWS_SECRET_ACCESS_KEY": "test-secret",
            "JENTIC_AGENT_API_KEY": "test-jentic-key"
        }, clear=False):
            agent = ReWOOAgentBedrock(model="us.anthropic.claude-3-5-haiku-20241022-v1:0")
            assert agent is not None

    def test_rewoo_agent_bedrock_creation_missing_env_vars(self):
        """Test ReWOO Bedrock agent creation fails with missing AWS credentials."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="Missing AWS credentials for Bedrock"):
                ReWOOAgentBedrock(model="us.anthropic.claude-3-5-haiku-20241022-v1:0")


class TestReACTAgentValidation:
    """Test validation during ReACT agent creation."""

    def test_react_agent_creation_with_valid_anthropic_env(self):
        """Test ReACT agent can be created with valid Anthropic environment."""
        with patch.dict(os.environ, {
            "ANTHROPIC_API_KEY": "test-key",
            "JENTIC_AGENT_API_KEY": "test-jentic-key"
        }, clear=False):
            agent = ReACTAgent(model="claude-3-opus")
            assert agent is not None

    def test_react_agent_creation_missing_env_vars(self):
        """Test ReACT agent creation fails with missing environment variables."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="Missing required environment variables"):
                ReACTAgent(model="claude-3-opus")

    def test_react_agent_creation_missing_jentic_key(self):
        """Test ReACT agent creation fails with missing Jentic API key."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}, clear=True):
            with pytest.raises(ValueError, match="Missing JENTIC_AGENT_API_KEY"):
                ReACTAgent(model="claude-3-opus")

    def test_react_agent_bedrock_creation_with_valid_env(self):
        """Test ReACT Bedrock agent can be created with valid AWS environment."""
        with patch.dict(os.environ, {
            "AWS_BEARER_TOKEN_BEDROCK": "test-token",
            "JENTIC_AGENT_API_KEY": "test-jentic-key"
        }, clear=False):
            agent = ReACTAgentBedrock(model="us.anthropic.claude-3-5-haiku-20241022-v1:0")
            assert agent is not None

    def test_react_agent_bedrock_creation_missing_env_vars(self):
        """Test ReACT Bedrock agent creation fails with missing AWS credentials."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="Missing AWS credentials for Bedrock"):
                ReACTAgentBedrock(model="us.anthropic.claude-3-5-haiku-20241022-v1:0")


class TestProviderAgnosticBehavior:
    """Test that the validation doesn't break provider agnosticism."""

    def test_supports_all_litellm_providers(self):
        """Test that validation supports various LiteLLM providers."""
        test_cases = [
            ("gpt-4", "OPENAI_API_KEY"),
            ("claude-3-opus", "ANTHROPIC_API_KEY"),
            ("gemini-pro", "GEMINI_API_KEY"),
            ("command-r", "COHERE_API_KEY"),
            ("llama-2", "REPLICATE_API_TOKEN"),
            ("mistral-large", "MISTRAL_API_KEY"),
        ]
        
        for model, env_var in test_cases:
            with patch.dict(os.environ, {env_var: "test-key"}, clear=True):
                # Should not raise for any supported provider
                _validate_litellm_environment(model)

    def test_fallback_to_common_keys_preserves_agnosticism(self):
        """Test that unknown models can still work with common API keys."""
        unknown_models = [
            "some-new-provider-model",
            "experimental-model-v2",
            "custom-fine-tuned-model"
        ]
        
        # Test with each common API key
        common_keys = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY"]
        
        for api_key in common_keys:
            with patch.dict(os.environ, {api_key: "test-key"}, clear=True):
                for model in unknown_models:
                    # Should not raise - preserves ability to use any model
                    _validate_litellm_environment(model)

    def test_validation_provides_helpful_error_messages(self):
        """Test that error messages guide users to provider documentation."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError) as exc_info:
                _validate_litellm_environment("unknown-model")
            
            error_msg = str(exc_info.value)
            assert "https://docs.litellm.ai/docs/providers" in error_msg
            assert "OPENAI_API_KEY" in error_msg
            assert "ANTHROPIC_API_KEY" in error_msg