# Weather bot example

A self-contained example agent that answers questions like *"What's the
weather in Tokyo right now?"* by calling a real weather API.

## What's in this directory

- `weather_tool.py` — the reusable `ToolBase` / `JustInTimeToolingBase`
  implementation plus pure-Python helpers for the HTTP layer. Imported by the
  agent script and by the unit tests.
- `weather_agent.py` — thin wiring script that combines `WeatherTools` with
  `LiteLLM`, `ReACTReasoner` and `StandardAgent` and runs an interactive REPL.

## Running it

```bash
# from the repo root
python -m examples.tools.weather.weather_agent
```

Set whichever LLM provider key matches the model in `weather_agent.py`
(default: `gemini/gemini-2.0-flash`) in your `.env`:

```
GEMINI_API_KEY=...
```

The weather API itself (Open-Meteo) requires **no API key**.

## Example prompts

- *"What's the weather in Tokyo right now?"*
- *"Is it raining in London?"*
- *"How warm is San Francisco in Fahrenheit?"*
- *"Compare the current temperature in Reykjavík and Dubai."*

## Swapping the weather provider

The default provider is [Open-Meteo](https://open-meteo.com/) because it's
free, no-signup, and keeps the example runnable out of the box. If you want
to use a different backend, only `weather_tool.py` needs to change:

- **OpenWeatherMap** — sign up at <https://openweathermap.org/api>, set
  `OPENWEATHERMAP_API_KEY` in your `.env`, and replace the body of
  `WeatherTools.execute` with a call to
  `https://api.openweathermap.org/data/2.5/weather?q=...&appid=...`.
- **WeatherAPI** — sign up at <https://www.weatherapi.com/>, set
  `WEATHERAPI_KEY` in your `.env`, and call
  `https://api.weatherapi.com/v1/current.json?key=...&q=...`.

The rest of the agent (reasoner, memory, agent wiring) is provider-agnostic.

## Tests

Unit tests live in `tests/tools/test_weather_tool.py` and patch
`urllib.request.urlopen` with canned Open-Meteo payloads, so they run
offline:

```bash
python -m pytest tests/tools/test_weather_tool.py -v
```
