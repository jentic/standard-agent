"""Weather bot example for Standard Agent.

A self-contained example that wires a single ``weather.current`` tool into a
``StandardAgent`` with the ReAct reasoner. The tool calls the free
`Open-Meteo <https://open-meteo.com/>`_ APIs, which require **no API key**.

The actual ``ToolBase`` implementation and HTTP helpers live next door in
``weather_tool.py`` so they can be unit-tested in isolation; this file just
wires them into a ``StandardAgent``.

API key setup
-------------
Open-Meteo is free for non-commercial use and does **not** require an API key,
so this example works out of the box. If you prefer a different provider:

- **OpenWeatherMap** — sign up at https://openweathermap.org/api, then set
  ``OPENWEATHERMAP_API_KEY`` in your ``.env`` and swap ``WeatherTools.execute``
  to call ``https://api.openweathermap.org/data/2.5/weather``.
- **WeatherAPI** — sign up at https://www.weatherapi.com/, then set
  ``WEATHERAPI_KEY`` in your ``.env``.

Run
---
::

    python -m examples.tools.weather.weather_agent

Example goals to try:

- ``"What's the weather in Tokyo right now?"``
- ``"Is it raining in London?"``
- ``"Give me the current temperature in San Francisco in Fahrenheit."``
"""

from __future__ import annotations

import logging

from dotenv import load_dotenv

from agents.llm.litellm import LiteLLM
from agents.memory.dict_memory import DictMemory
from agents.reasoner.react import ReACTReasoner
from agents.standard_agent import StandardAgent
from examples._cli_helpers import print_result, read_user_goal
from examples.tools.weather.weather_tool import WeatherTools

logging.basicConfig(level=logging.INFO)

load_dotenv()

# Step 1: Choose and configure your components
llm = LiteLLM(model="gemini/gemini-2.0-flash", max_tokens=200)
tools = WeatherTools()
memory = DictMemory()

# Step 2: Pick a reasoner profile
custom_reasoner = ReACTReasoner(llm=llm, tools=tools, memory=memory)

# Step 3: Wire everything into the StandardAgent
agent = StandardAgent(
    llm=llm,
    tools=tools,
    memory=memory,
    reasoner=custom_reasoner,
)


def main() -> None:
    print("🌤️  Weather agent is ready! Ask about any city.")
    while True:
        try:
            goal = read_user_goal()
            if not goal:
                continue
            result = agent.solve(goal)
            print_result(result)
        except KeyboardInterrupt:
            print("\n👋 Bye!")
            break


# Example prompts to try
# """What's the weather in Tokyo right now?"""
# """Is it raining in London?"""
# """How warm is San Francisco in Fahrenheit?"""
# """Compare the current temperature in Reykjavík and Dubai."""

if __name__ == "__main__":
    main()
