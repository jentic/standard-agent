from __future__ import annotations

from subprocess import CompletedProcess
from typing import Any, Dict

import pytest

from examples.tools.integration.local_temperature import LocalTemperatureTools
from examples.tools.integration.shell_command import ShellCommandTools
from examples.tools.integration.weather_api import WeatherAPIClient


def test_local_temperature_tool_executes() -> None:
    provider = LocalTemperatureTools()

    results = provider.search("How do I convert this temperature?", top_k=3)
    assert results, "search should surface the temperature tool"

    tool = provider.load(results[0])
    output = provider.execute(
        tool,
        {"value": 100, "from_unit": "celsius", "to_unit": "fahrenheit"},
    )

    assert pytest.approx(output, rel=1e-3) == 212


class _DummyResponse:
    def __init__(self, payload: Dict[str, Any]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> Dict[str, Any]:
        return self._payload


class _DummySession:
    def __init__(self) -> None:
        self.last_url: str | None = None
        self.last_params: Dict[str, Any] | None = None

    def get(self, url: str, params: Dict[str, Any], timeout: float | None = None) -> _DummyResponse:
        self.last_url = url
        self.last_params = params
        return _DummyResponse(
            {
                "location": params.get("q"),
                "temperature": 21.5,
                "conditions": "partly cloudy",
            }
        )


def test_weather_api_tool_executes() -> None:
    session = _DummySession()
    provider = WeatherAPIClient(base_url="https://weather.test", api_key="token", session=session)

    results = provider.search("weather forecast for seattle")
    assert results, "search should expose the weather tool"

    tool = provider.load(results[0])
    payload = provider.execute(tool, {"location": "Seattle", "units": "imperial"})

    assert payload["temperature"] == 21.5
    assert payload["location"] == "Seattle"
    assert payload["conditions"] == "partly cloudy"
    assert session.last_url == "https://weather.test/weather"
    assert session.last_params == {"q": "Seattle", "units": "imperial", "apikey": "token"}


def _runner_success(args, *, capture_output: bool, text: bool, timeout: float | None = None) -> CompletedProcess[str]:
    assert capture_output is True and text is True
    return CompletedProcess(args=list(args), returncode=0, stdout="hello world\n", stderr="")


def test_shell_command_tool_executes_when_allowed() -> None:
    provider = ShellCommandTools(allow_list=["echo"], runner=_runner_success)

    results = provider.search("run a shell command for me")
    assert results, "search should expose the shell command tool"

    tool = provider.load(results[0])
    outcome = provider.execute(tool, {"command": "echo", "args": ["hello", "world"]})

    assert outcome["stdout"] == "hello world"
    assert outcome["returncode"] == 0


def test_shell_command_tool_rejects_unknown_command() -> None:
    provider = ShellCommandTools(allow_list=["echo"], runner=_runner_success)
    tool = provider.load(provider.search("shell")[0])

    with pytest.raises(ValueError, match="not permitted"):
        provider.execute(tool, {"command": "rm", "args": []})
