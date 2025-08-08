from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any, Dict, Tuple

from agents.tools.base import JustInTimeToolingBase
from agents.reasoner.implicit.reasoner import ImplicitState
from agents.reasoner.implicit.act import Act


class JITActPlaceholder(Act):
    def __init__(self, *, tools: JustInTimeToolingBase) -> None:
        self.tools = tools

    def __call__(self, state: ImplicitState, memory: MutableMapping) -> Tuple[str, Dict[str, Any], Any]:
        raise NotImplementedError("Provide an Act implementation (tool selection + param generation + invoke).")


