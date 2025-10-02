# test_bedrock.py

import pytest
import json
import os
from unittest.mock import patch, MagicMock
from agents.llm.bedrock import BedrockLLM, JSON_CORRECTION_PROMPT

class TestBedrockLLM:
    # Tests default initialisation of LLM service with default model from environment variable
    @patch('os.getenv')
    @patch('agents.llm.bedrock.boto3.client')
    def test_env_model_used(self, mock_boto_client, mock_getenv):
        mock_getenv.return_value = "us.anthropic.claude-3-5-haiku-20241022-v1:0"
        svc = BedrockLLM()
        assert svc.model == "us.anthropic.claude-3-5-haiku-20241022-v1:0"

    # Tests default initialisation of LLM service with default model and temperature parameter set
    @patch('os.getenv')
    @patch('agents.llm.bedrock.boto3.client')
    def test_env_temperature_used(self, mock_boto_client, mock_getenv):
        mock_getenv.return_value = "us.anthropic.claude-3-5-haiku-20241022-v1:0"
        svc = BedrockLLM(temperature=0.7)
        assert svc.model == "us.anthropic.claude-3-5-haiku-20241022-v1:0"
        assert svc.temperature == 0.7

    # Tests default initialisation of LLM service with default model and max tokens parameter set
    @patch('os.getenv')
    @patch('agents.llm.bedrock.boto3.client')
    def test_env_max_tokens_used(self, mock_boto_client, mock_getenv):
        mock_getenv.return_value = "us.anthropic.claude-3-5-haiku-20241022-v1:0"
        svc = BedrockLLM(max_tokens=10000)
        assert svc.model == "us.anthropic.claude-3-5-haiku-20241022-v1:0"
        assert svc.max_tokens == 10000

    # Tests initialisation of LLM service with Claude model parameter override
    @patch('agents.llm.bedrock.boto3.client')
    def test_claude_init_any_model_parameter_override(self, mock_boto_client):
        svc = BedrockLLM(
            model="us.anthropic.claude-3-5-sonnet-20241022-v2:0",
        )
        assert svc.model == "us.anthropic.claude-3-5-sonnet-20241022-v2:0"

    # Tests initialisation of LLM service with different Bedrock model
    @patch('agents.llm.bedrock.boto3.client')
    def test_bedrock_init_any_model_parameter_override(self, mock_boto_client):
        svc = BedrockLLM(
            model="us.anthropic.claude-3-5-haiku-20241022-v1:0"
        )
        assert svc.model == "us.anthropic.claude-3-5-haiku-20241022-v1:0"

    # Tests that AWS_REGION environment variable is used
    @patch('os.getenv')
    @patch('agents.llm.bedrock.boto3.client')
    def test_aws_region_from_env(self, mock_boto_client, mock_getenv):
        def getenv_side_effect(key, default=None):
            if key == "LLM_MODEL":
                return "us.anthropic.claude-3-5-haiku-20241022-v1:0"
            elif key == "AWS_REGION":
                return "us-west-2"
            return default
        
        mock_getenv.side_effect = getenv_side_effect
        svc = BedrockLLM()
        assert svc.region_name == "us-west-2"
        mock_boto_client.assert_called_once_with(
            service_name="bedrock-runtime",
            region_name="us-west-2"
        )

    # Tests that default region is used when AWS_REGION is not set
    @patch('os.getenv')
    @patch('agents.llm.bedrock.boto3.client')
    def test_default_region_used(self, mock_boto_client, mock_getenv):
        def getenv_side_effect(key, default=None):
            if key == "LLM_MODEL":
                return "us.anthropic.claude-3-5-haiku-20241022-v1:0"
            elif key == "AWS_REGION":
                return default
            return default
        
        mock_getenv.side_effect = getenv_side_effect
        svc = BedrockLLM()
        assert svc.region_name == "us-east-1"
        mock_boto_client.assert_called_once_with(
            service_name="bedrock-runtime",
            region_name="us-east-1"
        )
    
    @patch('agents.llm.bedrock.boto3.client')
    def test_completion(self, mock_boto_client):
        # Arrange: Configure the mock to return a predictable response
        mock_client_instance = MagicMock()
        mock_boto_client.return_value = mock_client_instance
        
        mock_response = {
            "output": {
                "message": {
                    "content": [
                        {"text": "  Mocked response content  "}
                    ]
                }
            },
            "usage": {
                "inputTokens": 10,
                "outputTokens": 20,
                "totalTokens": 30
            }
        }
        mock_client_instance.converse.return_value = mock_response

        svc = BedrockLLM(model="us.anthropic.claude-3-5-haiku-20241022-v1:0", temperature=0.7)

        # Act: Call the completion method
        messages = [{"role": "user", "content": "Hello"}]
        result = svc.completion(messages)

        # Assert: Check that the result is an LLMResponse with the stripped content
        from agents.llm.base_llm import BaseLLM
        assert isinstance(result, BaseLLM.LLMResponse)
        assert result.text == "Mocked response content"
        assert result.prompt_tokens == 10
        assert result.completion_tokens == 20
        assert result.total_tokens == 30

        # Assert: Check that converse was called with the correct arguments
        mock_client_instance.converse.assert_called_once()
        call_args = mock_client_instance.converse.call_args[1]
        assert call_args["modelId"] == "us.anthropic.claude-3-5-haiku-20241022-v1:0"
        assert call_args["messages"] == [{"role": "user", "content": [{"text": "Hello"}]}]
        assert call_args["inferenceConfig"]["temperature"] == 0.7
    
    @patch('os.getenv')
    @patch('agents.llm.bedrock.boto3.client')
    def test_completion_env_parameters_used(self, mock_boto_client, mock_getenv):
        # Arrange: Configure the mock to return a predictable response
        mock_client_instance = MagicMock()
        mock_boto_client.return_value = mock_client_instance
        
        mock_response = {
            "output": {
                "message": {
                    "content": [
                        {"text": "  Mocked response content  "}
                    ]
                }
            },
            "usage": {
                "inputTokens": 10,
                "outputTokens": 20,
                "totalTokens": 30
            }
        }
        mock_client_instance.converse.return_value = mock_response
        
        def getenv_side_effect(key, default=None):
            if key == "AWS_REGION":
                return "us-east-1"
            return default
        
        mock_getenv.side_effect = getenv_side_effect

        svc = BedrockLLM(
            model="us.anthropic.claude-3-5-haiku-20241022-v1:0",
            temperature=0.7,
            max_tokens=10000,
        )

        assert svc.model == "us.anthropic.claude-3-5-haiku-20241022-v1:0"
        assert svc.temperature == 0.7
        assert svc.max_tokens == 10000

        # Act: Call the completion method
        messages = [{"role": "user", "content": "Hello"}]
        result = svc.completion(messages)

        # Assert: Check that the result is an LLMResponse with the stripped content
        from agents.llm.base_llm import BaseLLM
        assert isinstance(result, BaseLLM.LLMResponse)
        assert result.text == "Mocked response content"

        # Assert: Check that converse was called with the correct arguments
        mock_client_instance.converse.assert_called_once()
        call_args = mock_client_instance.converse.call_args[1]
        assert call_args["modelId"] == "us.anthropic.claude-3-5-haiku-20241022-v1:0"
        assert call_args["inferenceConfig"]["temperature"] == 0.7
        assert call_args["inferenceConfig"]["maxTokens"] == 10000
    
    @patch('agents.llm.bedrock.boto3.client')
    def test_completion_kwargs_parameters_used(self, mock_boto_client):
        # Arrange: Configure the mock to return a predictable response
        mock_client_instance = MagicMock()
        mock_boto_client.return_value = mock_client_instance
        
        mock_response = {
            "output": {
                "message": {
                    "content": [
                        {"text": "  Mocked response content  "}
                    ]
                }
            },
            "usage": {
                "inputTokens": 10,
                "outputTokens": 20,
                "totalTokens": 30
            }
        }
        mock_client_instance.converse.return_value = mock_response

        svc = BedrockLLM(
            model="us.anthropic.claude-3-5-haiku-20241022-v1:0",
            temperature=0.7,
            max_tokens=10000,
        )

        assert svc.model == "us.anthropic.claude-3-5-haiku-20241022-v1:0"
        assert svc.temperature == 0.7
        assert svc.max_tokens == 10000

        # Act: Call the completion method
        messages = [{"role": "user", "content": "Hello"}]
        result = svc.completion(messages)

        # Assert: Check that the result is an LLMResponse with the stripped content
        from agents.llm.base_llm import BaseLLM
        assert isinstance(result, BaseLLM.LLMResponse)
        assert result.text == "Mocked response content"

        # Assert: Check that converse was called with the correct arguments
        mock_client_instance.converse.assert_called_once()
        call_args = mock_client_instance.converse.call_args[1]
        assert call_args["modelId"] == "us.anthropic.claude-3-5-haiku-20241022-v1:0"
        assert call_args["inferenceConfig"]["temperature"] == 0.7
        assert call_args["inferenceConfig"]["maxTokens"] == 10000

    @patch("agents.llm.bedrock.BedrockLLM.prompt", return_value='{"key": "value"}')
    @patch("agents.llm.base_llm.BaseLLM.prompt_to_json")
    @patch('agents.llm.bedrock.boto3.client')
    def test_prompt_to_json_success(self, mock_boto_client, mock_base_prompt_to_json, mock_prompt):
        mock_prompt.return_value = '{"key": "value"}'
        mock_base_prompt_to_json.return_value = {"key": "value"}

        svc = BedrockLLM(model="test-model")
        result = svc.prompt_to_json("Give me a JSON object")

        assert result == {"key": "value"}
        mock_base_prompt_to_json.assert_called_once()

    @patch("agents.llm.bedrock.BedrockLLM.prompt")
    @patch("agents.llm.base_llm.BaseLLM.prompt_to_json")
    @patch('agents.llm.bedrock.boto3.client')
    def test_prompt_to_json_jsondecode_retry_then_success(self, mock_boto_client, mock_base_prompt_to_json, mock_prompt):
        mock_prompt.return_value = '{"invalid": "json"}'
        mock_base_prompt_to_json.side_effect = [
            json.JSONDecodeError("bad json", "", 0),
            {"fixed": "json"},
        ]

        svc = BedrockLLM(model="test-model")
        result = svc.prompt_to_json("Give me a JSON object", max_retries=1)

        assert result == {"fixed": "json"}
        assert mock_base_prompt_to_json.call_count == 2

    @patch("agents.llm.bedrock.BedrockLLM.prompt")
    @patch("agents.llm.base_llm.BaseLLM.prompt_to_json")
    @patch('agents.llm.bedrock.boto3.client')
    def test_prompt_to_json_valueerror_retry_same_prompt(self, mock_boto_client, mock_base_prompt_to_json, mock_prompt):
        mock_prompt.return_value = '{"some": "content"}'
        mock_base_prompt_to_json.side_effect = [
            ValueError("malformed response"),
            {"ok": True},
        ]

        svc = BedrockLLM(model="test-model")
        result = svc.prompt_to_json("Give me a JSON object", max_retries=1)

        assert result == {"ok": True}
        assert mock_base_prompt_to_json.call_count == 2

    @patch("agents.llm.bedrock.BedrockLLM.prompt")
    @patch("agents.llm.base_llm.BaseLLM.prompt_to_json")
    @patch('agents.llm.bedrock.boto3.client')
    def test_prompt_to_json_max_retries_exceeded_jsondecode(self, mock_boto_client, mock_base_prompt_to_json, mock_prompt):
        mock_prompt.return_value = '{"bad": "json"}'
        mock_base_prompt_to_json.side_effect = json.JSONDecodeError("bad json", "", 0)

        svc = BedrockLLM(model="test-model")

        # Act & Assert: should raise after retries
        with pytest.raises(json.JSONDecodeError):
            svc.prompt_to_json("Give me a JSON object", max_retries=1)

    @patch("agents.llm.bedrock.BedrockLLM.prompt")
    @patch("agents.llm.base_llm.BaseLLM.prompt_to_json")
    @patch('agents.llm.bedrock.boto3.client')
    def test_prompt_to_json_max_retries_exceeded_valueerror(self, mock_boto_client, mock_base_prompt_to_json, mock_prompt):
        mock_prompt.return_value = ''
        mock_base_prompt_to_json.side_effect = ValueError("empty response")

        svc = BedrockLLM(model="test-model")
        with pytest.raises(ValueError):
            svc.prompt_to_json("Give me a JSON object", max_retries=1)

    @patch("agents.llm.base_llm.BaseLLM.prompt_to_json")
    @patch('agents.llm.bedrock.boto3.client')
    def test_prompt_to_json_uses_raw_content_when_available(self, mock_boto_client, mock_base_prompt_to_json):
        from agents.llm.bedrock import BedrockLLM, JSON_CORRECTION_PROMPT
        svc = BedrockLLM(model="test-model")
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
    @patch('agents.llm.bedrock.boto3.client')
    def test_prompt_to_json_fallback_when_raw_content_missing(self, mock_boto_client, mock_base_prompt_to_json):
        svc = BedrockLLM(model="test-model")
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
    @patch('agents.llm.bedrock.boto3.client')
    def test_prompt_to_json_correction_prompt_content_differs_by_raw_availability(self, mock_boto_client, mock_base_prompt_to_json):
        svc = BedrockLLM(model="test-model")
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

    @patch('agents.llm.bedrock.boto3.client')
    def test_completion_returns_empty_string_for_empty_content(self, mock_boto_client):
        """Test that empty/whitespace content returns empty string (not error)."""
        mock_client_instance = MagicMock()
        mock_boto_client.return_value = mock_client_instance
        
        mock_response = {
            "output": {
                "message": {
                    "content": [
                        {"text": "   "}
                    ]
                }
            },
            "usage": {}
        }
        mock_client_instance.converse.return_value = mock_response
        
        svc = BedrockLLM(model="test-model")
        result = svc.completion([{"role": "user", "content": "test"}])
        from agents.llm.base_llm import BaseLLM
        assert isinstance(result, BaseLLM.LLMResponse)
        assert result.text == ""  # Empty after strip

    @patch('agents.llm.bedrock.boto3.client')
    def test_completion_returns_empty_string_for_malformed_response(self, mock_boto_client):
        """Test that malformed response returns empty string (not error)."""
        mock_client_instance = MagicMock()
        mock_boto_client.return_value = mock_client_instance
        
        # Simulate missing content
        mock_response = {
            "output": {
                "message": {}
            },
            "usage": {}
        }
        mock_client_instance.converse.return_value = mock_response
        
        svc = BedrockLLM(model="test-model")
        result = svc.completion([{"role": "user", "content": "test"}])
        from agents.llm.base_llm import BaseLLM
        assert isinstance(result, BaseLLM.LLMResponse)
        assert result.text == ""  # Empty when extraction fails

    @patch("agents.llm.base_llm.BaseLLM.prompt_to_json")
    @patch('agents.llm.bedrock.boto3.client')
    def test_prompt_to_json_uses_raw_content_when_available(self, mock_boto_client, mock_base_prompt_to_json):
        from agents.llm.bedrock import BedrockLLM, JSON_CORRECTION_PROMPT
        svc = BedrockLLM(model="test-model")
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
    @patch('agents.llm.bedrock.boto3.client')
    def test_prompt_to_json_fallback_when_raw_content_missing(self, mock_boto_client, mock_base_prompt_to_json):
        svc = BedrockLLM(model="test-model")
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
    @patch('agents.llm.bedrock.boto3.client')
    def test_prompt_to_json_correction_prompt_content_differs_by_raw_availability(self, mock_boto_client, mock_base_prompt_to_json):
        svc = BedrockLLM(model="test-model")
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

    @patch('agents.llm.bedrock.boto3.client')
    def test_completion_with_json_mode(self, mock_boto_client):
        """Test that response_format parameter adds system prompt for JSON mode."""
        mock_client_instance = MagicMock()
        mock_boto_client.return_value = mock_client_instance
        
        mock_response = {
            "output": {
                "message": {
                    "content": [
                        {"text": '{"result": "json"}'}
                    ]
                }
            },
            "usage": {}
        }
        mock_client_instance.converse.return_value = mock_response
        
        svc = BedrockLLM(model="test-model")
        messages = [{"role": "user", "content": "Give me JSON"}]
        result = svc.completion(messages, response_format={"type": "json_object"})
        
        # Assert: Check that system parameter was added
        call_args = mock_client_instance.converse.call_args[1]
        assert "system" in call_args
        assert call_args["system"][0]["text"] == "You must respond with valid JSON only. Do not include any text outside the JSON object."

    @patch('agents.llm.bedrock.boto3.client')
    def test_message_format_conversion(self, mock_boto_client):
        """Test that OpenAI-style messages are converted to Bedrock format."""
        mock_client_instance = MagicMock()
        mock_boto_client.return_value = mock_client_instance
        
        mock_response = {
            "output": {
                "message": {
                    "content": [
                        {"text": "Response"}
                    ]
                }
            },
            "usage": {}
        }
        mock_client_instance.converse.return_value = mock_response
        
        svc = BedrockLLM(model="test-model")
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "How are you?"}
        ]
        svc.completion(messages)
        
        # Assert: Check that messages were converted to Bedrock format
        call_args = mock_client_instance.converse.call_args[1]
        bedrock_messages = call_args["messages"]
        assert len(bedrock_messages) == 3
        assert bedrock_messages[0] == {"role": "user", "content": [{"text": "Hello"}]}
        assert bedrock_messages[1] == {"role": "assistant", "content": [{"text": "Hi there"}]}
        assert bedrock_messages[2] == {"role": "user", "content": [{"text": "How are you?"}]}

    @patch('agents.llm.bedrock.boto3.client')
    def test_token_usage_extraction(self, mock_boto_client):
        """Test that token usage is correctly extracted from Bedrock response."""
        mock_client_instance = MagicMock()
        mock_boto_client.return_value = mock_client_instance
        
        mock_response = {
            "output": {
                "message": {
                    "content": [
                        {"text": "Response"}
                    ]
                }
            },
            "usage": {
                "inputTokens": 15,
                "outputTokens": 25
            }
        }
        mock_client_instance.converse.return_value = mock_response
        
        svc = BedrockLLM(model="test-model")
        result = svc.completion([{"role": "user", "content": "test"}])
        
        assert result.prompt_tokens == 15
        assert result.completion_tokens == 25
        assert result.total_tokens == 40  # Should be computed
