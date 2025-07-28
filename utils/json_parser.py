"""JSON parsing utility for handling LLM responses."""

import json
import re
from typing import Any, Dict

from utils.logger import get_logger

logger = get_logger(__name__)


def parse_json(raw: str) -> Dict[str, Any]:
    """Parse JSON with fallback for markdown-wrapped responses."""
    # First try raw parsing
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    
    # Try extracting from markdown code blocks
    _JSON_FENCE_RE = re.compile(r"```(?:json)?\s*([^`]+)\s*```")
    match = _JSON_FENCE_RE.search(raw)
    if match:
        extracted = match.group(1).strip()
        try:
            return json.loads(extracted)
        except json.JSONDecodeError:
            pass
    
    # Log failure and raise error
    logger.warning(f"phase=JSON_PARSE_FAIL raw='{raw}'")
    raise json.JSONDecodeError(f"LLM returned invalid JSON: {raw}", raw, 0)