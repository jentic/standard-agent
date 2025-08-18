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

    @patch('os.getenv')
    @patch('agents.llm.base_llm.BaseLLM.prompt_to_json')
    # Tests prompt_to_json method
    def test_prompt_to_json(self, mock_base_prompt_to_json, mock_getenv):
        # Arrange: Configure the mock to return a predictable JSON object
        expected_json = {"key": "value"}
        mock_base_prompt_to_json.return_value = expected_json
        mock_getenv.return_value = "claude-sonnet-4"

        svc = LiteLLM()
        prompt_content = "Give me a JSON object"

        # Act: Call the method under test
        result = svc.prompt_to_json(prompt_content)

        # Assert: Check that the result is what the mock returned
        assert result == expected_json

        # Assert: Check that the base method was called correctly
        mock_base_prompt_to_json.assert_called_once_with(prompt_content)

    @patch('os.getenv')
    @patch('agents.llm.base_llm.BaseLLM.prompt_to_json')
    # Tests prompt_to_json retry logic with max_retries parameter
    def test_prompt_to_json_max_retries_used(self, mock_base_prompt_to_json, mock_getenv):
        # Arrange: Configure the mock to fail twice, then succeed on third attempt
        expected_json = {"key": "value"}
        mock_base_prompt_to_json.side_effect = [
            json.JSONDecodeError("Invalid JSON", "", 0),  # First attempt fails
            json.JSONDecodeError("Invalid JSON", "", 0),  # Second attempt fails
            expected_json  # Third attempt succeeds
        ]
        mock_getenv.return_value = "claude-sonnet-4"

        svc = LiteLLM()
        prompt_content = "Give me a JSON object"

        # Act: Call the method under test with max_retries=2 (allows 3 total attempts)
        result = svc.prompt_to_json(prompt_content, max_retries=2)

        # Assert: Check that the result is what the mock returned on success
        assert result == expected_json

        # Assert: Check that the base method was called 3 times (initial + 2 retries)
        assert mock_base_prompt_to_json.call_count == 3
        
        # Assert: Check that the first call used the original prompt
        first_call_args = mock_base_prompt_to_json.call_args_list[0]
        assert first_call_args[0][0] == prompt_content
        
        # Assert: Check that subsequent calls used modified prompts (retry logic)
        second_call_args = mock_base_prompt_to_json.call_args_list[1]
        assert "Give me a JSON object" in second_call_args[0][0]  # Original prompt in correction template
        assert "previous response was not valid JSON" in second_call_args[0][0]

    @patch('os.getenv')
    @patch('agents.llm.base_llm.BaseLLM.prompt_to_json')
    # Tests prompt_to_json when max_retries is exceeded
    def test_prompt_to_json_max_retries_exceeded(self, mock_base_prompt_to_json, mock_getenv):
        # Arrange: Configure the mock to always fail
        mock_base_prompt_to_json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
        mock_getenv.return_value = "claude-sonnet-4"

        svc = LiteLLM()
        prompt_content = "Give me a JSON object"

        # Act & Assert: Call should raise JSONDecodeError after max_retries+1 attempts
        with pytest.raises(json.JSONDecodeError):
            svc.prompt_to_json(prompt_content, max_retries=1)

        # Assert: Check that the base method was called max_retries+1 times (2 attempts with max_retries=1)
        assert mock_base_prompt_to_json.call_count == 2