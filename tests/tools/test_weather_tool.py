"""Tests for the weather example tool (``examples/tools/weather``)."""

from __future__ import annotations

import io
import json
from typing import Any, Dict, List

import pytest

from examples.tools.weather.weather_tool import (
    GeocodeHit,
    WMO_CODES,
    WeatherLookupError,
    WeatherTool,
    WeatherTools,
    fetch_current,
    geocode,
    http_get_json,
)

_GEOCODE_BERLIN = {
    "results": [
        {
            "name": "Berlin",
            "country": "Germany",
            "latitude": 52.52,
            "longitude": 13.41,
        }
    ]
}

_FORECAST_BERLIN = {
    "current": {
        "time": "2026-05-20T08:00",
        "temperature_2m": 18.4,
        "relative_humidity_2m": 62,
        "wind_speed_10m": 11.7,
        "weather_code": 3,
    },
    "current_units": {
        "temperature_2m": "°C",
        "wind_speed_10m": "km/h",
    },
}


class _FakeResponse:
    """Minimal stand-in for the object returned by ``urllib.request.urlopen``."""

    def __init__(self, payload: Dict[str, Any]) -> None:
        self._raw = io.BytesIO(json.dumps(payload).encode("utf-8"))
        self.headers = _FakeHeaders()

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *_exc: object) -> None:
        self._raw.close()

    def read(self) -> bytes:
        return self._raw.read()


class _FakeHeaders:
    def get_content_charset(self) -> str:
        return "utf-8"


@pytest.fixture
def stubbed_http(monkeypatch: pytest.MonkeyPatch) -> List[str]:
    """Patch the HTTP layer to return canned Open-Meteo responses.

    The fixture returns the list of URLs that were "fetched" so individual
    tests can assert call order without coupling to the params dict.
    """
    called: List[str] = []

    def fake_urlopen(url: str, timeout: float = 0):  # noqa: ARG001 - signature match
        called.append(url)
        if "geocoding-api" in url:
            return _FakeResponse(_GEOCODE_BERLIN)
        if "api.open-meteo.com" in url:
            return _FakeResponse(_FORECAST_BERLIN)
        raise AssertionError(f"unexpected URL: {url}")

    monkeypatch.setattr(
        "examples.tools.weather.weather_tool.urllib_request.urlopen",
        fake_urlopen,
    )
    return called


class TestWeatherTool:
    """Schema-level checks on the ToolBase wrapper."""

    def test_tool_id_and_schema(self) -> None:
        tool = WeatherTool()

        assert tool.id == "weather.current"
        params = tool.get_parameter_schema()
        assert "location" in params
        assert "units" in params
        assert params["units"]["enum"] == ["metric", "imperial"]

        summary = tool.get_summary()
        assert "weather.current" in summary
        assert "Open-Meteo" in summary

        details = json.loads(tool.get_details())
        assert details["id"] == "weather.current"
        assert details["required"] == ["location"]


class TestWeatherToolsExecute:
    """Behavioural checks on the JustInTimeToolingBase provider."""

    def test_execute_returns_formatted_current_weather(self, stubbed_http: List[str]) -> None:
        provider = WeatherTools()
        tool = provider.search("any")[0]

        result = provider.execute(tool, {"location": "Berlin"})

        assert result["ok"] is True
        assert result["location"] == "Berlin, Germany"
        assert result["temperature"] == 18.4
        assert result["humidity_percent"] == 62
        assert result["wind_speed"] == 11.7
        assert result["condition"] == WMO_CODES[3]  # "overcast"
        assert result["units"] == "metric"
        # Sanity: both APIs were hit, geocoding before forecast.
        assert len(stubbed_http) == 2
        assert "geocoding-api" in stubbed_http[0]
        assert "api.open-meteo.com" in stubbed_http[1]

    def test_execute_imperial_units_passes_imperial_query_params(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured_urls: List[str] = []

        def fake_urlopen(url: str, timeout: float = 0):  # noqa: ARG001
            captured_urls.append(url)
            if "geocoding-api" in url:
                return _FakeResponse(_GEOCODE_BERLIN)
            return _FakeResponse(_FORECAST_BERLIN)

        monkeypatch.setattr(
            "examples.tools.weather.weather_tool.urllib_request.urlopen",
            fake_urlopen,
        )

        provider = WeatherTools()
        tool = provider.search("any")[0]
        result = provider.execute(tool, {"location": "Berlin", "units": "imperial"})

        assert result["units"] == "imperial"
        forecast_url = captured_urls[1]
        assert "temperature_unit=fahrenheit" in forecast_url
        assert "wind_speed_unit=mph" in forecast_url

    def test_execute_rejects_empty_location(self) -> None:
        provider = WeatherTools()
        tool = provider.search("any")[0]

        with pytest.raises(ValueError, match="non-empty string"):
            provider.execute(tool, {"location": "   "})

    def test_execute_rejects_unknown_units(self) -> None:
        provider = WeatherTools()
        tool = provider.search("any")[0]

        with pytest.raises(ValueError, match="metric.*imperial"):
            provider.execute(tool, {"location": "Berlin", "units": "kelvin"})

    def test_execute_returns_ok_false_when_location_not_found(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fake_urlopen(url: str, timeout: float = 0):  # noqa: ARG001
            return _FakeResponse({"results": []})

        monkeypatch.setattr(
            "examples.tools.weather.weather_tool.urllib_request.urlopen",
            fake_urlopen,
        )

        provider = WeatherTools()
        tool = provider.search("any")[0]
        result = provider.execute(tool, {"location": "Nowhere-McNoplace"})

        assert result["ok"] is False
        assert "Couldn't find a location" in result["error"]


class TestNetworkErrors:
    """Network-level failures should surface as :class:`WeatherLookupError`."""

    def test_http_get_json_wraps_url_errors(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from urllib import error as urllib_error

        def boom(url: str, timeout: float = 0):  # noqa: ARG001
            raise urllib_error.URLError("connection refused")

        monkeypatch.setattr(
            "examples.tools.weather.weather_tool.urllib_request.urlopen",
            boom,
        )

        with pytest.raises(WeatherLookupError, match="network error"):
            http_get_json("https://example.invalid", {})

    def test_fetch_current_uses_unknown_when_code_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        payload = {
            "current": {
                "time": "2026-05-20T08:00",
                "temperature_2m": 10.0,
                "relative_humidity_2m": 80,
                "wind_speed_10m": 5.0,
                # weather_code intentionally missing
            },
            "current_units": {},
        }

        def fake_urlopen(url: str, timeout: float = 0):  # noqa: ARG001
            return _FakeResponse(payload)

        monkeypatch.setattr(
            "examples.tools.weather.weather_tool.urllib_request.urlopen",
            fake_urlopen,
        )

        result = fetch_current(
            GeocodeHit(name="X", country=None, latitude=0.0, longitude=0.0),
            units="metric",
        )

        assert result["condition"] == "unknown"
        assert result["weather_code"] is None


def test_geocode_returns_none_for_empty_results(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "examples.tools.weather.weather_tool.urllib_request.urlopen",
        lambda url, timeout=0: _FakeResponse({"results": []}),  # noqa: ARG005
    )
    assert geocode("Atlantis") is None
