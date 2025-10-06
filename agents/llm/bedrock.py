from agents.llm.base_llm import BaseLLM, JSON_CORRECTION_PROMPT
from typing import List, Dict, Any
import json
import os
import boto3
from botocore.exceptions import ClientError

from utils.logger import get_logger
from utils.observability import observe
logger = get_logger(__name__)


def _model_supports_json_format(model_id: str) -> bool:
    """
    Check if the Bedrock model supports native JSON format via additionalModelRequestFields.

    Args:
        model_id: The Bedrock model ID (e.g., 'eu.mistral.pixtral-large-2502-v1:0')

    Returns:
        True if the model supports response_format in additionalModelRequestFields, False otherwise
    """
    is_supported = False
    if "mistral.pixtral" in model_id:
        is_supported = True
    return is_supported

def _model_supports_system_prompt(model_id: str) -> bool:
    """
    Check if the Bedrock model supports system prompts for JSON mode enforcement.

    Args:
        model_id: The Bedrock model ID (e.g., 'us.anthropic.claude-3-5-haiku-20241022-v1:0')

    Returns:
        True if the model supports system prompts, False otherwise
    """
    is_supported = False
    if "anthropic.claude" in model_id:
        is_supported = True
    elif "amazon.nova" in model_id:
        is_supported = True
    elif "meta.llama3" in model_id:
        is_supported = True
    return is_supported

class BedrockLLM(BaseLLM):
    """Wrapper around AWS Bedrock Converse API.
    
    Authentication:
        Uses Amazon Bedrock API keys via the AWS_BEARER_TOKEN_BEDROCK environment variable.
        Set the environment variable before running your code:
           export AWS_BEARER_TOKEN_BEDROCK=your-api-key
           export AWS_REGION=us-east-1
    """

    def __init__(
        self,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> None:
        """
        Initialize BedrockLLM.

        Args:
            model: Bedrock model ID (e.g., 'us.anthropic.claude-3-5-haiku-20241022-v1:0')
            temperature: Temperature for generation (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
        """
        super().__init__(model=model, temperature=temperature)
        self.max_tokens = max_tokens
        self.region_name = os.getenv("AWS_REGION", "us-east-1")
        
        # Initialize Bedrock client
        # The boto3 client will automatically use the AWS_BEARER_TOKEN_BEDROCK environment variable
        self.client = boto3.client(
            service_name="bedrock-runtime",
            region_name=self.region_name
        )

    @observe(llm=True)
    def completion(self, messages: List[Dict[str, str]], **kwargs) -> BaseLLM.LLMResponse:
        """
        Call Bedrock Converse API with the given messages.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            **kwargs: Additional parameters (temperature, max_tokens, etc.)

        Returns:
            LLMResponse with text and token usage information
        """
        # Merge default parameters with provided kwargs
        effective_temperature = kwargs.get("temperature", self.temperature)
        effective_max_tokens = kwargs.get("max_tokens", self.max_tokens)

        # Convert messages to Bedrock format
        bedrock_messages = self._convert_messages_to_bedrock_format(messages)

        # Build inference configuration
        inference_config = {}
        if effective_temperature is not None:
            inference_config["temperature"] = effective_temperature
        if effective_max_tokens is not None:
            inference_config["maxTokens"] = effective_max_tokens

        # Prepare converse API parameters
        converse_params: Dict[str, Any] = {
            "modelId": self.model,
            "messages": bedrock_messages,
        }
        
        if inference_config:
            converse_params["inferenceConfig"] = inference_config

        # Handle additional parameters like response_format for JSON mode
        additional_config = {}
        if "response_format" in kwargs and kwargs["response_format"].get("type") == "json_object":
            if _model_supports_json_format(self.model):
                additional_config["response_format"] = {"type": "json_object"}
                converse_params["additionalModelRequestFields"] = additional_config
            elif _model_supports_system_prompt(self.model):
                additional_config["system"] = [
                    {
                        "text": "You must respond with valid JSON only. Do not include any text outside the JSON object."
                    }
                ]
                converse_params["system"] = additional_config["system"]
            else:
                logger.warning("json_format_not_supported", model=self.model)

        try:
            # Call Bedrock Converse API
            response = self.client.converse(**converse_params)
            
            # Extract text from response
            text = ""
            if "output" in response and "message" in response["output"]:
                content_blocks = response["output"]["message"].get("content", [])
                for block in content_blocks:
                    if "text" in block:
                        text += block["text"]
            
            text = text.strip()

            # Extract token usage
            prompt_tokens, completion_tokens, total_tokens = self._extract_token_usage(response)

            return BaseLLM.LLMResponse(
                text=text,
                prompt_tokens=prompt_tokens if isinstance(prompt_tokens, int) else None,
                completion_tokens=completion_tokens if isinstance(completion_tokens, int) else None,
                total_tokens=total_tokens if isinstance(total_tokens, int) else None,
            )

        except ClientError as e:
            logger.error("bedrock_api_error", error=str(e), model=self.model)
            raise

    def prompt_to_json(self, content: str, max_retries: int = 3, **kwargs) -> Dict[str, Any]:
        """
        Enhanced JSON prompting with automatic retry logic.

        Handles two types of failures differently:
        - ValueError (empty/malformed responses): Retry same prompt for transient issues
        - JSONDecodeError (bad JSON syntax): Retry with correction prompt using actual failed content

        Args:
            content: The prompt content
            max_retries: Maximum number of retry attempts (default: 3)
            **kwargs: Additional arguments passed to completion()

        Returns:
            Parsed JSON object as a dictionary

        Raises:
            json.JSONDecodeError: If all retry attempts fail
            ValueError: If LLM consistently returns empty/malformed responses
        """

        original_prompt = content
        current_prompt = content

        for attempt in range(max_retries + 1):
            try:
                return super().prompt_to_json(current_prompt, **kwargs) 
            except json.JSONDecodeError as e:
                logger.warning("json_parse_failed", attempt=attempt, error=str(e))
                if attempt >= max_retries:
                    logger.error("json_decode_failed", attempt=attempt, error=str(e), msg="Exceeded max retries for JSON parsing")
                    raise

                if hasattr(e, 'raw_content'):
                    bad_json_content = e.raw_content
                else:
                    bad_json_content = "The previous response was not valid JSON"

                current_prompt = JSON_CORRECTION_PROMPT.format(
                    original_prompt=original_prompt,
                    bad_json=bad_json_content
                )

            except ValueError as e:
                logger.warning("empty/malformed_response_retry", attempt=attempt, error=str(e))
                if attempt >= max_retries:
                    raise

        # This should never be reached, but mypy requires it
        raise json.JSONDecodeError("Unexpected end of function", "", 0)

    def _convert_messages_to_bedrock_format(self, messages: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Convert OpenAI-style messages to Bedrock Converse API format.

        Args:
            messages: List of dicts with 'role' and 'content' keys

        Returns:
            List of Bedrock-formatted message dicts
        """
        bedrock_messages = []
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "assistant":
                bedrock_role = "assistant"
            else:  # user or any other role
                bedrock_role = "user"
            
            bedrock_messages.append({
                "role": bedrock_role,
                "content": [{"text": content}]
            })
        
        return bedrock_messages

    def _extract_token_usage(self, response: Dict[str, Any]) -> tuple[int | None, int | None, int | None]:
        """
        Extract token usage from Bedrock Converse API response.

        Args:
            response: Response dict from Bedrock Converse API

        Returns:
            Tuple of (prompt_tokens, completion_tokens, total_tokens)
        """
        try:
            usage = response.get("usage", {})
            
            prompt_tokens = usage.get("inputTokens")
            completion_tokens = usage.get("outputTokens")
            total_tokens = usage.get("totalTokens")

            # Compute total if missing but components available
            if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
                total_tokens = prompt_tokens + completion_tokens

            return prompt_tokens, completion_tokens, total_tokens
        except Exception as e:
            logger.warning("token_usage_extraction_failed", error=str(e))
            return None, None, None
