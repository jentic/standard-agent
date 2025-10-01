"""Simple dictionary-based memory implementation."""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any, TypedDict


class ConversationHistoryEntry(TypedDict, total=False):
    """Structure for conversation history entries stored in memory."""

    goal: str
    result: str


ConversationHistory = list[ConversationHistoryEntry]
MemoryValue = Any
MemoryStore = MutableMapping[str, MemoryValue]


def DictMemory() -> MemoryStore:
    """
    Create a simple in-memory storage using a dictionary.

    This is suitable for development, testing, and single-session use cases.
    Data is lost when the process terminates.

    Returns:
        Mutable mapping that can hold agent runtime state.

    Note:
        This is just a regular Python dict. Any MutableMapping implementation can be used
        as memory storage in place of this (Redis, custom classes, etc.). Custom classes need
        only implement ``__getitem__``, ``__setitem__``, ``__delitem__``, ``__iter__``, and
        ``__len__`` methods.
    """

    memory: MemoryStore = {}
    return memory

