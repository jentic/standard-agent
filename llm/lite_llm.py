from llm.base_llm import BaseLLM
from typing import List, Dict
from utils.load_config import load_config

class LiteLLMChatLLM(BaseLLM):
    """Wrapper around litellm.completion."""

    def __init__(
        self,
        model: str = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> None:
        import litellm
        config = load_config()
        self.model = model if model is not None else config.llm.model
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