"""
Unit tests for memory implementations.
"""
from ..memory.scratch_pad import ScratchPadMemory


class TestScratchPadMemory:
    """Test cases for ScratchPadMemory"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.memory = ScratchPadMemory()
    
    def test_store_and_retrieve(self):
        """Test basic store and retrieve functionality"""
        self.memory.store("key1", "value1")
        assert self.memory.retrieve("key1") == "value1"
    
    def test_retrieve_nonexistent_key(self):
        """Test retrieving a key that doesn't exist"""
        assert self.memory.retrieve("nonexistent") is None
    
    def test_store_overwrite(self):
        """Test overwriting an existing key"""
        self.memory.store("key1", "original")
        self.memory.store("key1", "updated")
        assert self.memory.retrieve("key1") == "updated"
    
    def test_delete_existing_key(self):
        """Test deleting an existing key"""
        self.memory.store("key1", "value1")
        assert self.memory.delete("key1") is True
        assert self.memory.retrieve("key1") is None
    
    def test_delete_nonexistent_key(self):
        """Test deleting a key that doesn't exist"""
        assert self.memory.delete("nonexistent") is False
    
    def test_clear(self):
        """Test clearing all stored values"""
        self.memory.store("key1", "value1")
        self.memory.store("key2", "value2")
        self.memory.clear()
        assert self.memory.retrieve("key1") is None
        assert self.memory.retrieve("key2") is None
        assert len(self.memory) == 0
    
    def test_keys(self):
        """Test getting all keys"""
        self.memory.store("key1", "value1")
        self.memory.store("key2", "value2")
        keys = self.memory.keys()
        assert set(keys) == {"key1", "key2"}
    
    def test_len(self):
        """Test length functionality"""
        assert len(self.memory) == 0
        self.memory.store("key1", "value1")
        assert len(self.memory) == 1
        self.memory.store("key2", "value2")
        assert len(self.memory) == 2
    
    def test_contains(self):
        """Test __contains__ functionality"""
        assert "key1" not in self.memory
        self.memory.store("key1", "value1")
        assert "key1" in self.memory
    
    def test_store_complex_data(self):
        """Test storing complex data types"""
        data = {"nested": {"list": [1, 2, 3], "dict": {"a": "b"}}}
        self.memory.store("complex", data)
        retrieved = self.memory.retrieve("complex")
        assert retrieved == data
