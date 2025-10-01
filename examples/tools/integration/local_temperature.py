"""Local-only tool example for converting temperatures between units."""

from __future__ import annotations

from typing import Any, Dict, List

from agents.tools.base import JustInTimeToolingBase, ToolBase


class TemperatureConversionTool(ToolBase):
    """Simple in-process tool that performs temperature conversions."""

    TOOL_ID = "temperature.convert"

    def __init__(self) -> None:
        super().__init__(id=self.TOOL_ID)
        self.name = "Temperature Converter"
        self.description = (
            "Convert temperature values between Celsius and Fahrenheit without leaving the runtime."
        )
        self._parameters: Dict[str, Any] = {
            "value": {
                "type": "number",
                "description": "Temperature value to convert.",
            },
            "from_unit": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "description": "Unit of the provided temperature value.",
            },
            "to_unit": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "description": "Unit to convert the value into.",
            },
        }

    def get_summary(self) -> str:
        return f"{self.id}: {self.name} - {self.description}"

    def get_details(self) -> str:
        return (
            "Temperature converter. Provide `value`, `from_unit`, and `to_unit` "
            "to receive the converted number."
        )

    def get_parameters(self) -> Dict[str, Any]:
        return self._parameters


class LocalTemperatureTools(JustInTimeToolingBase):
    """Tool provider that exposes the local temperature converter."""

    def __init__(self) -> None:
        self._tool = TemperatureConversionTool()

    def search(self, query: str, *, top_k: int = 10) -> List[ToolBase]:
        if "temperature" in query.lower() or "convert" in query.lower():
            return [self._tool]
        return []

    def load(self, tool: ToolBase) -> ToolBase:
        if tool.id != self._tool.id:
            raise ValueError(f"Unknown tool requested: {tool.id}")
        return self._tool

    def execute(self, tool: ToolBase, parameters: Dict[str, Any]) -> Any:
        if tool.id != self._tool.id:
            raise ValueError(f"Unexpected tool invocation: {tool.id}")

        value = float(parameters.get("value", 0.0))
        from_unit = str(parameters.get("from_unit", "celsius")).lower()
        to_unit = str(parameters.get("to_unit", "fahrenheit")).lower()

        if from_unit == to_unit:
            return value
        if from_unit == "celsius" and to_unit == "fahrenheit":
            return (value * 9.0 / 5.0) + 32.0
        if from_unit == "fahrenheit" and to_unit == "celsius":
            return (value - 32.0) * 5.0 / 9.0

        raise ValueError("Unsupported unit conversion requested")
