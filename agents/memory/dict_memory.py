"""
Simple dictionary-based memory implementation.

This module provides a factory function for creating memory storage.
Any MutableMapping implementation can be used as memory storage in the system.
"""
from collections.abc import MutableMapping
from typing import Any


def DictMemory() -> MutableMapping[str, Any]:
    """
    Create a simple in-memory storage backed by a dictionary.

    This is suitable for development, testing, and single-session use cases.
    Data is lost when the process terminates.

    Returns:
        An empty ``MutableMapping[str, Any]`` (concretely a ``dict``).
        The return type is intentionally widened to ``MutableMapping`` to
        match the type used by consumers such as ``StandardAgent`` and the
        reasoners — see ``agents/standard_agent.py`` and
        ``agents/reasoner/base.py``.

    Note:
        This is just a regular Python ``dict``. Any ``MutableMapping``
        implementation can be used as memory storage in place of this
        (Redis-backed, custom classes, etc.). Custom classes need only
        implement ``__getitem__``, ``__setitem__``, ``__delitem__``,
        ``__iter__``, and ``__len__``.
    """
    return {}
