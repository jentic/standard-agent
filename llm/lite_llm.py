from llm.base_llm import BaseLLM
from typing import List, Dict
from utils.load_config import load_config

class LiteLLMChatLLM(BaseLLM):
    """Wrapper around litellm.completion."""

    def __init__(
        self,
        model: str,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> None:
        import litellm
        if not model:
            raise ValueError("model parameter is required and cannot be empty")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = litellm

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        resp = self._client.completion(
            model=self.model,
            messages=messages,
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
        )
        content = resp.choices[0].message.content
        return content or ""  # Avoid None propagati