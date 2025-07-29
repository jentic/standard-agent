"""
Simple dictionary-based memory implementation.

This module provides a factory function for creating memory storage.
Any MutableMapping implementation can be used as memory storage in the system.
"""
from typing import Any, Dict


def DictMemory() -> Dict[str, Any]:
    """
    Create a simple in-memory storage using a dictionary.
    
    This is suitable for development, testing, and single-session use cases.
    Data is lost when the process terminates.
    
    Returns:
        Empty dictionary that implements MutableMapping interface
        
    Note:
        This is just a regular Python dict. Any MutableMapping implementation
        can be used as memory storage in place of this (Redis, custom classes, etc.).
        Custom classes need only implement __getitem__, __setitem__, __delitem__, 
        __iter__, and __len__ methods.
    """
    return {}

