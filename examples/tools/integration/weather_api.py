"""HTTP API integration example for fetching current weather information."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

import requests  # type: ignore[import-untyped]

from agents.tools.base import JustInTimeToolingBase, ToolBase


@runtime_checkable
class _SupportsJSONResponse(Protocol):
    """Subset of the requests.Response API used by the example."""

    def raise_for_status(self) -> None:
        ...

    def json(self) -> Dict[str, Any]:
        ...


class _SupportsGet(Protocol):
    """Protocol describing an HTTP client with a `get` method."""

    def get(self, url: str, params: Dict[str, Any], timeout: Optional[float] = None) -> _SupportsJSONResponse:
        ...


class WeatherAPITool(ToolBase):
    """Tool metadata describing a REST call for current weather."""

    TOOL_ID = "weather.current"

    def __init__(self) -> None:
        super().__init__(id=self.TOOL_ID)
        self.name = "Current Weather"
        self.description = "Fetch the current temperature and condition for a location via HTTP."
        self._parameters: Dict[str, Any] = {
            "location": {
                "type": "string",
                "description": "City or coordinates that the API understands.",
            },
            "units": {
                "type": "string",
                "enum": ["metric", "imperial"],
                "description": "Optional unit system supported by the API.",
            },
        }

    def get_summary(self) -> str:
        return f"{self.id}: {self.name} - {self.description}"

    def get_details(self) -> str:
        return (
            "Calls the configured weather REST endpoint with `location` and optional `units` "
            "and returns the parsed payload that the agent can reason over."
        )

    def get_parameters(self) -> Dict[str, Any]:
        return self._parameters


@dataclass
class WeatherAPIClient(JustInTimeToolingBase):
    """HTTP-backed tool provider that integrates with a weather service."""

    base_url: str
    api_key: Optional[str] = None
    timeout: float = 10.0
    session: _SupportsGet | None = None

    def __post_init__(self) -> None:
        self._tool = WeatherAPITool()
        self._session: _SupportsGet = self.session or requests.Session()

    def search(self, query: str, *, top_k: int = 10) -> List[ToolBase]:
        if {"weather", "forecast", "temperature"} & set(query.lower().split()):
            return [self._tool]
        return []

    def load(self, tool: ToolBase) -> ToolBase:
        if tool.id != self._tool.id:
            raise ValueError(f"Unknown tool requested: {tool.id}")
        return self._tool

    def execute(self, tool: ToolBase, parameters: Dict[str, Any]) -> Dict[str, Any]:
        if tool.id != self._tool.id:
            raise ValueError(f"Unexpected tool invocation: {tool.id}")

        location = str(parameters.get("location", "")).strip()
        if not location:
            raise ValueError("location parameter is required")

        units = str(parameters.get("units", "metric") or "metric").lower()

        url = f"{self.base_url.rstrip('/')}/weather"
        query_params: Dict[str, Any] = {"q": location, "units": units}
        if self.api_key:
            query_params["apikey"] = self.api_key

        response = self._session.get(url, params=query_params, timeout=self.timeout)
        response.raise_for_status()
        payload = response.json()

        return {
            "location": payload.get("location", location),
            "temperature": payload.get("temperature"),
            "conditions": payload.get("conditions"),
            "raw": payload,
        }
