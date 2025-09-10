# Test for the litellm.py script

import pytest
import json
import os
from unittest.mock import patch, MagicMock
from agents.llm.litellm import LiteLLM

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
        assert svc.temperature == 0.7

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
        assert svc.temperature == 0.7
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
        assert svc.temperature == 0.7
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
    def test_prompt_to_json(self, mock_base_prompt_to_json, mock_prompt):
        # Arrange
        expected_json = {"key": "value"}
        mock_base_prompt_to_json.return_value = expected_json

        svc = LiteLLM()
        prompt_content = "Give me a JSON object"

        # Act
        result = svc.prompt_to_json(prompt_content)

        # Assert
        assert result == expected_json
        mock_base_prompt_to_json.assert_called_once_with(prompt_content)

    @patch("agents.llm.litellm.LiteLLM.prompt")
    @patch("agents.llm.base_llm.BaseLLM.prompt_to_json")
    def test_prompt_to_json_max_retries_used(self, mock_base_prompt_to_json, mock_prompt):
        # Arrange: fail twice, then succeed
        expected_json = {"key": "value"}
        mock_base_prompt_to_json.side_effect = [
            json.JSONDecodeError("Invalid JSON", "", 0),  # fail
            json.JSONDecodeError("Invalid JSON", "", 0),  # fail again
            expected_json,  # succeed
        ]
        mock_prompt.return_value = '{"invalid": "json"}'

        svc = LiteLLM()
        prompt_content = "Give me a JSON object"

        # Act
        result = svc.prompt_to_json(prompt_content, max_retries=2)

        # Assert
        assert result == expected_json
        assert mock_base_prompt_to_json.call_count == 3

        # Check that retry prompts included the original content
        retry_call_args = [c[0][0] for c in mock_base_prompt_to_json.call_args_list]
        assert any("Give me a JSON object" in arg for arg in retry_call_args)
        # Loosened: just make sure some correction instructions were injected
        assert any("correct" in arg.lower() or "fix" in arg.lower() for arg in retry_call_args)

    @patch("agents.llm.litellm.LiteLLM.prompt", return_value="not json")
    @patch("agents.llm.base_llm.BaseLLM.prompt_to_json")
    def test_prompt_to_json_max_retries_exceeded(self, mock_base_prompt_to_json, mock_prompt):
        # Arrange: always fail
        mock_base_prompt_to_json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)

        svc = LiteLLM()
        prompt_content = "Give me a JSON object"

        # Act & Assert: should raise after retries
        with pytest.raises(json.JSONDecodeError):
            svc.prompt_to_json(prompt_content, max_retries=1)

        assert mock_base_prompt_to_json.call_count == 2  # 1 initial + 1 retry