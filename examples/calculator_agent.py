"""
Simple Calculator Agent Example
-------------------------------
Demonstrates custom tool creation and agent integration with a CLI interface.

Requirements:
- Implements basic math operations: add, subtract, multiply, divide
- Shows proper tool class structure
- Includes error handling and helpful comments
"""

from agents.tools.base import ToolBase
from typing import Any, Dict

class CalculatorTool(ToolBase):
    """A tool for basic calculator operations."""

    def __init__(self, id: str = "calculator"):
        super().__init__(id)

    def get_summary(self) -> str:
        """Summary for LLM tool selection."""
        return "Performs basic arithmetic: add, subtract, multiply, divide."

    def get_details(self) -> str:
        """Detailed info for LLM reflection."""
        return (
            "Calculator tool that supports addition, subtraction, multiplication, and division. "
            "Requires two numeric parameters and an operation."
        )

    def get_parameters(self) -> Dict[str, Any]:
        """Parameter schema for LLM parameter generation."""
        return {
            "operation": {
                "type": "string",
                "enum": ["add", "subtract", "multiply", "divide"],
                "description": "Math operation to perform",
            },
            "a": {"type": "number", "description": "First operand"},
            "b": {"type": "number", "description": "Second operand"},
        }

    def run(self, operation: str, a: Any, b: Any) -> Dict[str, Any]:
        """Perform the requested calculation."""
        try:
            a, b = float(a), float(b)
        except (ValueError, TypeError):
            return {"error": "Operands must be numbers."}

        if operation == "add":
            result = a + b
        elif operation == "subtract":
            result = a - b
        elif operation == "multiply":
            result = a * b
        elif operation == "divide":
            if b == 0:
                return {"error": "Division by zero is not allowed."}
            result = a / b
        else:
            return {"error": f"Unknown operation '{operation}'."}

        return {"result": result}

class CalculatorTools:
    """Manages calculator tools."""

    def __init__(self):
        self.tool = CalculatorTool()

    def execute(self, operation: str, a: Any, b: Any) -> Dict[str, Any]:
        return self.tool.run(operation, a, b)