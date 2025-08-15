# Test for the litellm.py script

import pytest
from unittest.mock import patch, MagicMock
from agents.llm.litellm import LiteLLM

class TestLiteLLM:
    # Tests default initialisation of LLM service with Gemini provider for specific default config model
    def test_default_init_default_model(self):
        svc = LiteLLM()
        assert svc.model == "claude-sonnet-4"

    # Tests initialisation of LLM service with Anthropic provider for any model
    def test_anthropic_init_any_model(self):
        svc = LiteLLM(
            model="claude-sonnet-4-20250514",
        )
        assert svc.model == "claude-sonnet-4-20250514"

    # Tests initialisation of LLM service with OpenAI provider for any model
    def test_openai_init_any_model(self):
        svc = LiteLLM(
            model="gpt-4o"
        )
        assert svc.model == "gpt-4o"

    # Tests initialisation of LLM service with Gemini provider for any model
    def test_gemini_init_any_model(self):
        svc = LiteLLM(
            model="gemini/gemini-2.0-flash",
        )
        assert svc.model == "gemini/gemini-2.0-flash"

    # Tests invalid provider (should not raise exception, just use fallback)
    def test_invalid_provider(self):
        svc = LiteLLM(
            model="invalid-model"
        )
        assert svc.model == "invalid-model"
    
    @patch('agents.llm.litellm.litellm.completion')
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