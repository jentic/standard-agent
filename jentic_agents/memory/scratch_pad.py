"""
Simple dictionary-based memory implementation for development and testing.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from .base_memory import BaseMemory


@dataclass
class MemoryItem:
    key: str
    value: Any
    description: str
    type: str | None = None
    schema: Dict[str, Any] | None = None

    def as_prompt_line(self) -> str:
        part_type = f" ({self.type})" if self.type else ""
        return f"• {self.key}{part_type} – {self.description}"


class ScratchPadMemory(BaseMemory):
    """
    Simple in-memory storage using a dictionary.
    
    This is suitable for development, testing, and single-session use cases.
    Data is lost when the process terminates.
    
    Supports both simple key-value storage and enhanced memory items with
    descriptions, types, and placeholder resolution.
    """
    
    def __init__(self):
        """Initialize empty scratch pad memory."""
        self._storage: dict[str, Any] = {}  # Simple key-value storage
        self._store: Dict[str, MemoryItem] = {}  # Enhanced memory items
    
    # ---------- BaseMemory interface ----------
    
    def store(self, key: str, value: Any) -> None:
        """
        Store a value under the given key.
        
        Args:
            key: Unique identifier for the stored value
            value: Data to store
        """
        self._storage[key] = value
    
    def retrieve(self, key: str) -> Optional[Any]:
        """
        Retrieve a value by key.
        
        Args:
            key: Unique identifier for the value
            
        Returns:
            Stored value, or None if key not found
        """
        if key in self._store:
            return self._store[key].value
        return self._storage.get(key)
    
    def delete(self, key: str) -> bool:
        """
        Delete a stored value.
        
        Args:
            key: Unique identifier for the value to delete
            
        Returns:
            True if value was deleted, False if key not found
        """
        deleted_simple = key in self._storage
        deleted_enhanced = key in self._store
        
        if deleted_simple:
            del self._storage[key]
        if deleted_enhanced:
            del self._store[key]
            
        return deleted_simple or deleted_enhanced
    
    def clear(self) -> None:
        """
        Clear all stored values.
        """
        self._storage.clear()
        self._store.clear()
    
    def keys(self) -> list[str]:
        """
        Get all stored keys.
        
        Returns:
            List of all keys in memory
        """
        # Combine keys from both storage systems
        all_keys = set(self._storage.keys()) | set(self._store.keys())
        return list(all_keys)
    
    def __len__(self) -> int:
        """Return number of stored items."""
        all_keys = set(self._storage.keys()) | set(self._store.keys())
        return len(all_keys)
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists in storage."""
        return key in self._storage or key in self._store

    # ---------- Enhanced memory API ----------
    
    def set(self, key: str, value: Any, description: str, type_: str | None = None, schema: Dict[str, Any] | None = None) -> None:
        """
        Store a value with metadata as a MemoryItem.
        
        Args:
            key: Unique identifier for the stored value
            value: Data to store
            description: Human-readable description of the data
            type_: Optional type information
            schema: Optional schema information
        """
        self._store[key] = MemoryItem(key, value, description, type_, schema)

    def get(self, key: str) -> Any:
        """
        Get a value from enhanced storage.
        
        Args:
            key: Unique identifier for the value
            
        Returns:
            Stored value from MemoryItem
            
        Raises:
            KeyError: If key not found in enhanced storage
        """
        return self._store[key].value

    def has(self, key: str) -> bool:
        """
        Check if key exists in enhanced storage.
        
        Args:
            key: Unique identifier to check
            
        Returns:
            True if key exists in enhanced storage
        """
        return key in self._store

    def enumerate_for_prompt(self) -> str:
        """
        Generate a formatted string listing all memory items for LLM prompts.
        
        Returns:
            Formatted string describing all stored memory items
        """
        if not self._store:
            return "(memory empty)"
        lines = ["Available memory:"]
        for item in self._store.values():
            lines.append(item.as_prompt_line())
        return "\n".join(lines)

    # ---------- Placeholder resolution ----------

    _MEMORY_RE = re.compile(r"\$\{(?:\{)?memory\.([\w_\.]+)(?:\})?\}")

    def resolve_placeholders(self, obj: Any) -> Any:
        """
        Recursively substitute ${memory.foo.bar} placeholders inside *obj*.
        
        Args:
            obj: Object that may contain memory placeholders
            
        Returns:
            Object with placeholders resolved to actual memory values
        """
        if isinstance(obj, str):
            return self._MEMORY_RE.sub(lambda m: self._lookup(m.group(1)), obj)
        if isinstance(obj, list):
            return [self.resolve_placeholders(v) for v in obj]
        if isinstance(obj, dict):
            return {k: self.resolve_placeholders(v) for k, v in obj.items()}
        return obj

    # ---------- Private helpers ----------

    def _lookup(self, dotted_path: str) -> str:
        """
        Look up a value using dotted path notation.
        
        Args:
            dotted_path: Path like "key.subkey.index"
            
        Returns:
            String representation of the looked up value
        """
        key, *path = dotted_path.split(".")
        
        # Try enhanced storage first, then fall back to simple storage
        if key in self._store:
            item = self._store[key].value
        elif key in self._storage:
            item = self._storage[key]
        else:
            raise KeyError(f"Memory key '{key}' not found")
            
        # Navigate the dotted path
        for part in path:
            if isinstance(item, dict):
                item = item[part]
            elif isinstance(item, (list, tuple)):
                item = item[int(part)]
            else:
                raise KeyError(f"Cannot navigate path '{dotted_path}' - {part} not accessible")
        
        # Safely stringify the item
        if isinstance(item, str):
            return item
        
        # Try JSON serialization first
        try:
            return json.dumps(item)
        except (TypeError, ValueError):
            # If JSON serialization fails, fall back to string representation
            return str(item)
