from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ReasonKind(Enum):
    THOUGHT = "THOUGHT"
    ACTION = "ACTION"
    FINAL = "FINAL"


@dataclass
class ReasonNode:
    kind: ReasonKind
    text: str


