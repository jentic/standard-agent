# Test for the litellm.py script

import pytest
import json
import os
from unittest.mock import patch, MagicMock
from agents.llm.litellm import LiteLLM

class TestLiteLLM:
    # Tests default initialisation of LLM service with default model from environment variable
    @patch('os.getenv')
    def test_default_init_default_model(self, mock_getenv):
        mock_getenv.return_value = "claude-sonnet-4"
        svc = LiteLLM()
        assert svc.model == "claude-sonnet-4"

    # Tests default initialisation of LLM service with default model and temperature parameter set from environment variable
    @patch('os.getenv')
    def test_default_init_default_model_temperature_parameter(self, mock_getenv):
        mock_getenv.return_value = "claude-sonnet-4"
        svc = LiteLLM(temperature=0.7)
        assert svc.model == "claude-sonnet-4"
        assert svc.temperature == 0.7

    # Tests default initialisation of LLM service with default model and max tokens parameter set from environment variable
    @patch('os.getenv')
    def test_default_init_default_model_max_tokens_parameter(self, mock_getenv):
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

    # Tests invalid provider (should not raise exception, just use fallback)
    def test_invalid_provider_parameter_override(self):
        svc = LiteLLM(
            model="invalid-model"
        )
        assert svc.model == "invalid-model"
    
    @patch('agents.llm.litellm.litellm.completion')
    # Tests completion method
    def test_completion(self, mock_litellm_completion):
        # Arrange: Configure the mock to return a predictable response
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "  Mocked response content  "
        mock_litellm_completion.return_value = mock_response

        svc = LiteLLM(
            model="gemini/gemini-2.0-flash",
        )

        # Act: Call the completion method
        messages = [{"role": "user", "content": "Hello"}]
        result = svc.completion(messages)

        # Assert: Check that the result is the stripped content from the mock
        assert result == "Mocked response content"

        # Assert: Check that litellm.completion was called with the correct arguments
        mock_litellm_completion.assert_called_once_with(
            model="gemini/gemini-2.0-flash",
            messages=messages,
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