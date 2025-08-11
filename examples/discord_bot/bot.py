#!/usr/bin/env python3
import os
import re
import sys
import asyncio
import textwrap
from typing import Dict, Optional, List

from dotenv import load_dotenv

# Ensure project root is on sys.path so local imports work when running from examples/
# Resolve project root two levels up from this file: examples/discord_bot -> examples -> project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import discord

from agents.standard_agent import StandardAgent
from agents.prebuilt import ReWOOAgent
from agents.llm.litellm import LiteLLM
from agents.memory.dict_memory import DictMemory
from agents.reasoner.prebuilt import ReWOOReasoner
from agents.tools.jentic import JenticClient
from utils.logger import get_logger, init_logger

logger = get_logger(__name__)


def chunk(text: str, max_len: int = 2000) -> List[str]:
    if not text:
        return [""]
    parts: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_len, len(text))
        parts.append(text[start:end])
        start = end
    return parts


def parse_allowed_channels(env_val: Optional[str]) -> Optional[set[int]]:
    if not env_val:
        return None
    return {int(x.strip()) for x in env_val.split(",") if x.strip().isdigit()}


class AgentManager:
    """
    Caches a per-channel StandardAgent with shared LLM/Tools and isolated memory.
    """

    def __init__(self, model: str):
        self._llm = LiteLLM(model=model)
        self._tools = JenticClient()
        self._agents: Dict[int, StandardAgent] = {}

    def get(self, channel_id: int) -> StandardAgent:
        agent = self._agents.get(channel_id)
        if agent:
            return agent
        memory = DictMemory()
        reasoner = ReWOOReasoner(llm=self._llm, tools=self._tools, memory=memory, max_retries=2)
        agent = StandardAgent(
            llm=self._llm,
            tools=self._tools,
            memory=memory,
            reasoner=reasoner,
            conversation_history_window=5,
        )
        self._agents[channel_id] = agent
        return agent


def extract_goal_from_mention(content: str, bot_user_id: int) -> str:
    # Matches <@123> and <@!123>
    mention_regex = re.compile(rf"<@!?{bot_user_id}>")
    goal = mention_regex.sub("", content).strip()
    # Also strip a leading colon or punctuation if users write "@Bot: do X"
    goal = goal.lstrip(":,;.- ").strip()
    return goal


async def main() -> None:
    # Logging + env
    config_path = os.path.join(PROJECT_ROOT, "config.json")
    init_logger(config_path)
    load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

    token = os.getenv("DISCORD_BOT_TOKEN")
    if not token:
        raise RuntimeError("DISCORD_BOT_TOKEN is required in .env")

    model = os.getenv("LLM_MODEL", "claude-sonnet-4")
    allowed_channels = parse_allowed_channels(os.getenv("DISCORD_ALLOWED_CHANNEL_IDS"))

    intents = discord.Intents.default()
    intents.message_content = True
    intents.guilds = True
    intents.messages = True

    client = discord.Client(intents=intents)
    agents = AgentManager(model=model)

    @client.event
    async def on_ready():
        logger.info("discord_ready", user=str(client.user), user_id=getattr(client.user, "id", None))
        print(f"Logged in as {client.user} (ID: {getattr(client.user, 'id', None)})")

    @client.event
    async def on_message(message: discord.Message):
        try:
            # Ignore self and other bots
            if message.author.bot:
                return

            # Only respond if bot is mentioned or it's a DM
            bot_user = client.user
            if bot_user is None:
                return
            mentioned = bot_user in getattr(message, "mentions", [])
            is_dm = isinstance(message.channel, discord.DMChannel)
            if not (mentioned or is_dm):
                return

            # Optional channel allowlist
            if allowed_channels is not None and hasattr(message.channel, "id"):
                if message.channel.id not in allowed_channels:
                    return

            # Extract goal
            content = message.content or ""
            if mentioned:
                goal = extract_goal_from_mention(content, bot_user.id)  # type: ignore[arg-type]
            else:
                goal = content.strip()

            if not goal:
                await message.channel.send("Please provide a goal after mentioning me.")
                return

            channel_id = getattr(message.channel, "id", 0) or 0
            agent = agents.get(int(channel_id))

            async with message.channel.typing():
                # Bridge sync agent call to a thread to avoid blocking the loop
                result = await asyncio.to_thread(agent.solve, goal)

            final = result.final_answer or "(no answer)"
            for part in chunk(final, 2000):
                await message.channel.send(part)

        except Exception as exc:
            logger.exception("discord_on_message_error", error=str(exc))
            err = textwrap.shorten(str(exc), width=400, placeholder="…")
            await message.channel.send(f"Failed to process goal: {err}")

    await client.start(token)


if __name__ == "__main__":
    asyncio.run(main())


