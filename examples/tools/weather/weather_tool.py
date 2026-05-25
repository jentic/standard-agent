"""Reusable weather tool wiring for the Standard Agent example.

The tool calls the free `Open-Meteo <https://open-meteo.com/>`_ APIs and
requires **no API key**:

1. Geocoding API to resolve a human location string (e.g. ``"Berlin"`` or
   ``"São Paulo, BR"``) into ``(latitude, longitude)``.
2. Forecast API to fetch the current temperature, humidity, wind speed and
   weather code at that location.

The HTTP layer uses ``urllib`` from the standard library, so the example has
no extra runtime dependencies beyond what ``standard-agent`` already requires.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request

from agents.tools.base import JustInTimeToolingBase, ToolBase

logger = logging.getLogger("WeatherTool")

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
HTTP_TIMEOUT_SECONDS = 10.0

# Subset of WMO weather codes (https://open-meteo.com/en/docs#weathervariables).
# We keep the common ones so the agent gets a human-readable phrase instead of
# a bare integer.
WMO_CODES: Dict[int, str] = {
    0: "clear sky",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    48: "depositing rime fog",
    51: "light drizzle",
    53: "moderate drizzle",
    55: "dense drizzle",
    61: "slight rain",
    63: "moderate rain",
    65: "heavy rain",
    66: "light freezing rain",
    67: "heavy freezing rain",
    71: "slight snowfall",
    73: "moderate snowfall",
    75: "heavy snowfall",
    77: "snow grains",
    80: "slight rain showers",
    81: "moderate rain showers",
    82: "violent rain showers",
    85: "slight snow showers",
    86: "heavy snow showers",
    95: "thunderstorm",
    96: "thunderstorm with slight hail",
    99: "thunderstorm with heavy hail",
}


@dataclass(frozen=True)
class GeocodeHit:
    name: str
    country: Optional[str]
    latitude: float
    longitude: float


class WeatherLookupError(RuntimeError):
    """Raised for any non-recoverable error during a weather lookup."""


class WeatherTool(ToolBase):
    """Single-tool wrapper exposing the current weather lookup."""

    TOOL_ID = "weather.current"

    def __init__(self) -> None:
        super().__init__(id=self.TOOL_ID)
        self.name = "Current weather"
        self.description = (
            "Fetch the *current* weather conditions for a given location. "
            "Accepts a free-form city/place name (e.g. 'Berlin', 'San Francisco, US', "
            "'São Paulo'), resolves it via the Open-Meteo geocoding API, then returns "
            "temperature, humidity, wind speed and a human-readable weather condition. "
            "Use 'metric' for Celsius / km·h⁻¹ (default) or 'imperial' for Fahrenheit / mph."
        )
        self._parameters: Dict[str, Any] = {
            "location": {
                "type": "string",
                "description": "City or place name to look up (free-form text).",
            },
            "units": {
                "type": "string",
                "enum": ["metric", "imperial"],
                "description": "Unit system (default 'metric').",
            },
        }
        self._schema: Dict[str, Any] = {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "parameters": self._parameters,
            "required": ["location"],
        }

    def get_summary(self) -> str:
        return (
            f"{self.id}: {self.name} - Fetch current weather (temperature, "
            "humidity, wind, conditions) for a location string. API: Open-Meteo "
            "(no API key required). Parameters are supplied as JSON."
        )

    def get_details(self) -> str:
        return json.dumps(self._schema, ensure_ascii=False, indent=2)

    def get_parameter_schema(self) -> Dict[str, Any]:
        return self._parameters

    def get_parameters(self) -> Dict[str, Any]:
        return self._parameters


class WeatherTools(JustInTimeToolingBase):
    """Local provider exposing :class:`WeatherTool` to the agent."""

    def __init__(self) -> None:
        self._tool = WeatherTool()

    def search(self, query: str, *, top_k: int = 10) -> List[ToolBase]:
        return [self._tool]

    def load(self, tool: ToolBase) -> ToolBase:
        if getattr(tool, "id", None) != self._tool.id:
            logger.warning(
                "weather_load_mismatch requested_id=%s", getattr(tool, "id", None)
            )
        return self._tool

    def execute(self, tool: ToolBase, parameters: Dict[str, Any]) -> Any:
        if getattr(tool, "id", None) != self._tool.id:
            raise ValueError(f"Unknown tool: {tool}")

        location = parameters.get("location")
        if not isinstance(location, str) or not location.strip():
            raise ValueError("Parameter 'location' must be a non-empty string")

        units = str(parameters.get("units", "metric")).lower().strip()
        if units not in {"metric", "imperial"}:
            raise ValueError(
                f"Parameter 'units' must be 'metric' or 'imperial' (got {units!r})"
            )

        hit = geocode(location.strip())
        if hit is None:
            return {
                "ok": False,
                "error": f"Couldn't find a location matching '{location}'.",
                "location_query": location,
            }

        return fetch_current(hit, units=units)


def http_get_json(url: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """GET ``url`` with query ``params`` and parse the JSON response."""
    full_url = f"{url}?{urllib_parse.urlencode(params)}"
    try:
        with urllib_request.urlopen(full_url, timeout=HTTP_TIMEOUT_SECONDS) as resp:
            charset = resp.headers.get_content_charset() or "utf-8"
            return json.loads(resp.read().decode(charset))
    except urllib_error.HTTPError as exc:
        raise WeatherLookupError(
            f"Weather API HTTP {exc.code} for {url}: {exc.reason}"
        ) from exc
    except urllib_error.URLError as exc:
        raise WeatherLookupError(
            f"Weather API network error for {url}: {exc.reason}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise WeatherLookupError(
            f"Weather API returned non-JSON response from {url}"
        ) from exc


def geocode(query: str) -> Optional[GeocodeHit]:
    """Resolve a free-form location string to lat/lon via Open-Meteo."""
    data = http_get_json(
        GEOCODE_URL,
        {"name": query, "count": 1, "language": "en", "format": "json"},
    )
    results = data.get("results") or []
    if not results:
        return None
    first = results[0]
    try:
        return GeocodeHit(
            name=str(first["name"]),
            country=first.get("country"),
            latitude=float(first["latitude"]),
            longitude=float(first["longitude"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise WeatherLookupError(
            f"Geocoding response missing expected fields: {first!r}"
        ) from exc


def fetch_current(hit: GeocodeHit, *, units: str = "metric") -> Dict[str, Any]:
    """Fetch the current weather for ``hit`` and return a tidy summary."""
    temperature_unit = "celsius" if units == "metric" else "fahrenheit"
    wind_unit = "kmh" if units == "metric" else "mph"
    data = http_get_json(
        FORECAST_URL,
        {
            "latitude": hit.latitude,
            "longitude": hit.longitude,
            "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code",
            "temperature_unit": temperature_unit,
            "wind_speed_unit": wind_unit,
        },
    )
    current = data.get("current") or {}
    units_meta = data.get("current_units") or {}

    code = current.get("weather_code")
    condition = (
        WMO_CODES.get(int(code), f"weather code {code}")
        if code is not None
        else "unknown"
    )

    return {
        "ok": True,
        "location": _format_location(hit),
        "latitude": hit.latitude,
        "longitude": hit.longitude,
        "temperature": current.get("temperature_2m"),
        "temperature_unit": units_meta.get(
            "temperature_2m", "°C" if units == "metric" else "°F"
        ),
        "humidity_percent": current.get("relative_humidity_2m"),
        "wind_speed": current.get("wind_speed_10m"),
        "wind_speed_unit": units_meta.get(
            "wind_speed_10m", "km/h" if units == "metric" else "mph"
        ),
        "condition": condition,
        "weather_code": code,
        "observed_at": current.get("time"),
        "units": units,
    }


def _format_location(hit: GeocodeHit) -> str:
    return f"{hit.name}, {hit.country}" if hit.country else hit.name
