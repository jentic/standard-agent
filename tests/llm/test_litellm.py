# test_litellm.py

import pytest
import json
from unittest.mock import patch, MagicMock
from agents.llm.litellm import LiteLLM, JSON_CORRECTION_PROMPT

class TestLiteLLM:
    # Tests default initialisation of LLM service with default model from environment variable
    @patch('os.getenv')
    def test_env_model_used(self, mock_getenv):
        mock_getenv.return_value = "claude-sonnet-4"
        svc = LiteLLM()
        assert svc.model == "claude-sonnet-4"

    # Tests default initialisation of LLM service with default model and temperature parameter set from environment variable
    @patch('os.getenv')
    def test_env_temperature_used(self, mock_getenv):
        mock_getenv.return_value = "claude-sonnet-4"
        svc = LiteLLM(temperature=0.7)
        assert svc.model == "claude-sonnet-4"
        assert svc.temperature == pytest.approx(0.7)

    # Tests default initialisation of LLM service with default model and max tokens parameter set from environment variable
    @patch('os.getenv')
    def test_env_max_tokens_used(self, mock_getenv):
        mock_getenv.return_value = "claude-sonnet-4"
        svc = LiteLLM(max_tokens=10000)
        assert svc.model == "claude-sonnet-4"
        assert svc.max_tokens == 10000

    # Tests initialisation of LLM service with Anthropic provider for any model
    def test_anthropic_init_any_model_parameter_override(self):
        svc = LiteLLM(
            model="claude-sonnet-4-20250514",
        )
        assert svc.model == "claude-sonnet-4-20250514"

    # Tests initialisation of LLM service with OpenAI provider for any model
    def test_openai_init_any_model_parameter_override(self):
        svc = LiteLLM(
            model="gpt-4o"
        )
        assert svc.model == "gpt-4o"

    # Tests initialisation of LLM service with Gemini provider for any model
    def test_gemini_init_any_model_parameter_override(self):
        svc = LiteLLM(
            model="gemini/gemini-2.0-flash",
        )
        assert svc.model == "gemini/gemini-2.0-flash"
    
    @patch('agents.llm.litellm.litellm.completion')
    # Tests completion method
    def test_completion(self, mock_litellm_completion):
        # Arrange: Configure the mock to return a predictable response
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "  Mocked response content  "
        mock_litellm_completion.return_value = mock_response

        svc = LiteLLM(model="gemini/gemini-2.0-flash",temperature=0.7)

        # Act: Call the completion method
        messages = [{"role": "user", "content": "Hello"}]
        result = svc.completion(messages)

        # Assert: Check that the result is the stripped content from the mock
        assert result == "Mocked response content"

        # Assert: Check that litellm.completion was called with the correct arguments
        mock_litellm_completion.assert_called_once_with(
            model="gemini/gemini-2.0-flash",
            messages=messages,
            temperature=0.7
        )
    
    @patch('os.getenv')
    @patch('agents.llm.litellm.litellm.completion')
    # Tests completion method
    def test_completion_env_parameters_used(self, mock_litellm_completion, mock_getenv):
        # Arrange: Configure the mock to return a predictable response
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "  Mocked response content  "
        mock_litellm_completion.return_value = mock_response
        mock_getenv.return_value.temperature = 0.7
        mock_getenv.return_value.max_tokens = 10000

        svc = LiteLLM(
            model="gemini/gemini-2.0-flash",
            temperature=mock_getenv.return_value.temperature,
            max_tokens=mock_getenv.return_value.max_tokens,
        )

        assert svc.model == "gemini/gemini-2.0-flash"
        assert svc.temperature == pytest.approx(0.7)
        assert svc.max_tokens == 10000

        # Act: Call the completion method
        messages = [{"role": "user", "content": "Hello"}]
        result = svc.completion(messages)

        # Assert: Check that the result is the stripped content from the mock
        assert result == "Mocked response content"

        # Assert: Check that litellm.completion was called with the correct arguments
        mock_litellm_completion.assert_called_once_with(
            model="gemini/gemini-2.0-flash",
            messages=messages,
            temperature=0.7,
            max_tokens=10000,
        )
    
    @patch('agents.llm.litellm.litellm.completion')
    # Tests completion method
    def test_completion_kwargs_parameters_used(self, mock_litellm_completion):
        # Arrange: Configure the mock to return a predictable response
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "  Mocked response content  "
        mock_litellm_completion.return_value = mock_response

        svc = LiteLLM(
            model="gemini/gemini-2.0-flash",
            temperature=0.7,
            max_tokens=10000,
        )

        assert svc.model == "gemini/gemini-2.0-flash"
        assert svc.temperature == pytest.approx(0.7)
        assert svc.max_tokens == 10000

        # Act: Call the completion method
        messages = [{"role": "user", "content": "Hello"}]
        result = svc.completion(messages)

        # Assert: Check that the result is the stripped content from the mock
        assert result == "Mocked response content"

        # Assert: Check that litellm.completion was called with the correct arguments
        mock_litellm_completion.assert_called_once_with(
            model="gemini/gemini-2.0-flash",
            messages=messages,
            temperature=0.7,
            max_tokens=10000,
        )

    @patch("agents.llm.litellm.LiteLLM.prompt", return_value='{"key": "value"}')
    @patch("agents.llm.base_llm.BaseLLM.prompt_to_json")
    def test_prompt_to_json_success(self, mock_base_prompt_to_json, mock_prompt):
        mock_prompt.return_value = '{"key": "value"}'
        mock_base_prompt_to_json.return_value = {"key": "value"}
        # Provide explicit model to satisfy BaseLLM requirement without relying on env var
        svc = LiteLLM(model="test-model")
        result = svc.prompt_to_json("Give me a JSON object")

        assert result == {"key": "value"}
        mock_base_prompt_to_json.assert_called_once()

    @patch("agents.llm.litellm.LiteLLM.prompt")
    @patch("agents.llm.base_llm.BaseLLM.prompt_to_json")
    def test_prompt_to_json_jsondecode_retry_then_success(self, mock_base_prompt_to_json, mock_prompt):
        mock_prompt.return_value = '{"invalid": "json"}'
        mock_base_prompt_to_json.side_effect = [
            json.JSONDecodeError("bad json", "", 0),
            {"fixed": "json"},
        ]
        svc = LiteLLM(model="test-model")
        result = svc.prompt_to_json("Give me a JSON object", max_retries=1)

        assert result == {"fixed": "json"}
        assert mock_base_prompt_to_json.call_count == 2

    @patch("agents.llm.litellm.LiteLLM.prompt")
    @patch("agents.llm.base_llm.BaseLLM.prompt_to_json")
    def test_prompt_to_json_valueerror_retry_same_prompt(self, mock_base_prompt_to_json, mock_prompt):
        mock_prompt.return_value = '{"some": "content"}'
        mock_base_prompt_to_json.side_effect = [
            ValueError("malformed response"),
            {"ok": True},
        ]
        svc = LiteLLM(model="test-model")
        result = svc.prompt_to_json("Give me a JSON object", max_retries=1)

        assert result == {"ok": True}
        assert mock_base_prompt_to_json.call_count == 2

    @patch("agents.llm.litellm.LiteLLM.prompt")
    @patch("agents.llm.base_llm.BaseLLM.prompt_to_json")
    def test_prompt_to_json_max_retries_exceeded_jsondecode(self, mock_base_prompt_to_json, mock_prompt):
        mock_prompt.return_value = '{"bad": "json"}'
        mock_base_prompt_to_json.side_effect = json.JSONDecodeError("bad json", "", 0)
        svc = LiteLLM(model="test-model")

        # Act & Assert: should raise after retries
        with pytest.raises(json.JSONDecodeError):
            svc.prompt_to_json("Give me a JSON object", max_retries=1)

    @patch("agents.llm.litellm.LiteLLM.prompt")
    @patch("agents.llm.base_llm.BaseLLM.prompt_to_json")
    def test_prompt_to_json_max_retries_exceeded_valueerror(self, mock_base_prompt_to_json, mock_prompt):
        mock_prompt.return_value = ''
        mock_base_prompt_to_json.side_effect = ValueError("empty response")
        svc = LiteLLM(model="test-model")
        with pytest.raises(ValueError):
            svc.prompt_to_json("Give me a JSON object", max_retries=1)

    def test_completion_raises_valueerror_for_empty_content(self):
        svc = LiteLLM(model="test-model")
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = "   "
        with patch("agents.llm.litellm.litellm.completion", return_value=mock_resp):
            with pytest.raises(ValueError, match="empty response"):
                svc.completion([{"role": "user", "content": "test"}])

    def test_completion_raises_valueerror_for_malformed_response(self):
        svc = LiteLLM(model="test-model")
        # Simulate missing choices
        mock_resp = MagicMock()
        mock_resp.choices = []
        with patch("agents.llm.litellm.litellm.completion", return_value=mock_resp):
            with pytest.raises(ValueError, match="malformed response"):
                svc.completion([{"role": "user", "content": "test"}])

    @patch("agents.llm.base_llm.BaseLLM.prompt_to_json")
    def test_prompt_to_json_uses_raw_content_when_available(self, mock_base_prompt_to_json):
        from agents.llm.litellm import LiteLLM, JSON_CORRECTION_PROMPT
        svc = LiteLLM(model="test-model")
        class FakeError(json.JSONDecodeError):
            def __init__(self):
                super().__init__("bad json", "", 0)
                self.raw_content = "RAW_JSON_ERROR_CONTENT"
        # First call raises error with raw_content, second returns success
        mock_base_prompt_to_json.side_effect = [FakeError(), {"fixed": True}]
        result = svc.prompt_to_json("prompt", max_retries=1)
        assert result == {"fixed": True}
        # Check that the correction prompt used raw_content
        correction_prompt = JSON_CORRECTION_PROMPT.format(
            original_prompt="prompt",
            bad_json="RAW_JSON_ERROR_CONTENT"
        )
        called_args = mock_base_prompt_to_json.call_args_list[1][0][0]
        assert called_args == correction_prompt

    @patch("agents.llm.base_llm.BaseLLM.prompt_to_json")
    def test_prompt_to_json_fallback_when_raw_content_missing(self, mock_base_prompt_to_json):
        svc = LiteLLM(model="test-model")
        # Error without raw_content
        err = json.JSONDecodeError("bad json", "", 0)
        mock_base_prompt_to_json.side_effect = [err, {"fixed": True}]
        result = svc.prompt_to_json("prompt", max_retries=1)
        assert result == {"fixed": True}
        correction_prompt = JSON_CORRECTION_PROMPT.format(
            original_prompt="prompt",
            bad_json="The previous response was not valid JSON"
        )
        called_args = mock_base_prompt_to_json.call_args_list[1][0][0]
        assert called_args == correction_prompt

    @patch("agents.llm.base_llm.BaseLLM.prompt_to_json")
    def test_prompt_to_json_correction_prompt_content_differs_by_raw_availability(self, mock_base_prompt_to_json):
        svc = LiteLLM(model="test-model")
        class FakeError(json.JSONDecodeError):
            def __init__(self):
                super().__init__("bad json", "", 0)
                self.raw_content = "RAW_JSON_ERROR_CONTENT"
        # First with raw_content, then without
        mock_base_prompt_to_json.side_effect = [FakeError(), json.JSONDecodeError("bad json", "", 0), {"fixed": True}]
        svc.prompt_to_json("prompt", max_retries=2)
        # Check both correction prompts
        prompt1 = mock_base_prompt_to_json.call_args_list[1][0][0]
        prompt2 = mock_base_prompt_to_json.call_args_list[2][0][0]
        assert "RAW_JSON_ERROR_CONTENT" in prompt1
        assert "The previous response was not valid JSON" in prompt2