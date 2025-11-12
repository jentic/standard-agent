import pytest
import warnings
from app.tools_base import FunctionTool, FunctionToolProvider

# ---------- Dummy tools ----------

def add_numbers(a: int, b: int) -> int:
    """Add two numbers.
    
    Returns the sum of a and b.
    """
    return a + b


async def async_uppercase(text: str) -> str:
    """Convert text to uppercase.
    
    Useful for async testing.
    """
    return text.upper()


def no_docstring_tool(x: int) -> int:
    return x * 2


# ---------- Tests ----------

def test_tool_registration_and_summary():
    provider = FunctionToolProvider()
    provider.tool(["add", "numbers"])(add_numbers)
    tool = provider._tools["add_numbers"]
    assert tool.id == "add_numbers"
    assert isinstance(tool, FunctionTool)
    assert "add_numbers: Add two numbers. Returns the sum of a and b." == tool.get_summary()
    assert tool.get_parameter_schema()["required"] == ["a", "b"]


def test_tool_with_missing_docstring_warns():
    provider = FunctionToolProvider()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        provider.tool()(no_docstring_tool)
        assert any("no docstring" in str(warn.message).lower() for warn in w)


def test_summary_uses_first_two_lines():
    tool = FunctionTool(add_numbers)
    summary = tool.get_summary()
    assert "Add two numbers." in summary
    assert "Returns the sum of a and b." in summary


def test_get_details_returns_full_docstring():
    tool = FunctionTool(add_numbers)
    details = tool.get_details()
    assert "Add two numbers." in details
    assert "Returns the sum" in details


def test_search_with_keyword_match():
    provider = FunctionToolProvider()
    provider.tool(keywords=["math", "addition"])(add_numbers)
    results = provider.search("I want to do addition")
    assert any(t.id == "add_numbers" for t in results)


def test_search_with_summary_word():
    provider = FunctionToolProvider()
    provider.tool()(add_numbers)
    results = provider.search("numbers")
    assert any(t.id == "add_numbers" for t in results)


def test_search_with_tool_id():
    provider = FunctionToolProvider()
    provider.tool()(add_numbers)
    results = provider.search("add_numbers")
    assert any(t.id == "add_numbers" for t in results)


def test_execute_sync_function():
    provider = FunctionToolProvider()
    provider.tool()(add_numbers)
    tool = provider._tools["add_numbers"]
    result = provider.execute(tool, {"a": 2, "b": 3})
    assert result == 5


def test_execute_async_function():
    provider = FunctionToolProvider()
    provider.tool()(async_uppercase)
    tool = provider._tools["async_uppercase"]
    result = provider.execute(tool, {"text": "hello"})
    assert result == "HELLO"


def test_load_invalid_tool_raises():
    provider = FunctionToolProvider()
    tool = FunctionTool(add_numbers)
    with pytest.raises(ValueError):
        provider.load(tool)
