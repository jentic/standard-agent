#!/usr/bin/env python3
import os
import re
import asyncio
import textwrap
from dataclasses import dataclass
from enum import Enum
from typing import Optional, List
from dotenv import load_dotenv

import discord

from agents.standard_agent import StandardAgent
from agents.prebuilt import ReACTAgent, ReWOOAgent
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


class ReasonerProfile(str, Enum):
    REWOO = "rewoo"
    REACT = "react"


@dataclass(slots=True)
class DiscordAgentRuntime:
    chosen_profile: ReasonerProfile = ReasonerProfile.REWOO
    current_agent: Optional[StandardAgent] = None
    bot_user_id: Optional[int] = None


def _build_agent(profile: ReasonerProfile) -> StandardAgent:
    model = os.getenv("LLM_MODEL")
    logger.info("initializing_agent", profile=profile.value, model=model)
    if profile is ReasonerProfile.REACT:
        return ReACTAgent(model=model)
    return ReWOOAgent(model=model)


def extract_goal_from_mention(content: str, bot_user_id: int) -> str:
    # Matches <@123> and <@!123>
    mention_regex = re.compile(rf"<@!?{bot_user_id}>")
    goal = mention_regex.sub("", content).strip()
    # Also strip a leading colon or punctuation if users write "@Bot: do X"
    goal = goal.lstrip(":,;.- ").strip()
    return goal


async def main() -> None:
    # Logging + env
    init_logger()
    load_dotenv()

    token = os.getenv("DISCORD_BOT_TOKEN")
    if not token:
        raise RuntimeError("DISCORD_BOT_TOKEN is required in .env")

    allowed_channels = parse_allowed_channels(os.getenv("DISCORD_ALLOWED_CHANNEL_IDS"))

    intents = discord.Intents.default()
    intents.message_content = True
    intents.guilds = True
    intents.messages = True

    client = discord.Client(intents=intents)
    runtime = DiscordAgentRuntime()
    # Preload agent if key provided
    if os.getenv("JENTIC_AGENT_API_KEY"):
        try:
            runtime.current_agent = _build_agent(runtime.chosen_profile)
        except Exception as exc:
            logger.error("agent_init_on_boot_failed", error=str(exc))

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

            # Only respond if bot is mentioned (mention-gated)
            bot_user = client.user
            if bot_user is None:
                return
            mentioned = bot_user in getattr(message, "mentions", [])
            if not mentioned:
                return

            # Optional channel allowlist
            if allowed_channels is not None and hasattr(message.channel, "id"):
                if message.channel.id not in allowed_channels:
                    return

            # Extract goal (text after mention)
            content = message.content or ""
            goal = extract_goal_from_mention(content, bot_user.id)  # type: ignore[arg-type]

            if not goal:
                await message.channel.send("Please provide a goal after mentioning me.")
                return

            # Handle simple text commands
            lower = goal.lower().strip()
            if lower.startswith("reasoner"):
                parts = lower.split()
                valid = {p.value for p in ReasonerProfile}
                if len(parts) == 2 and parts[1] == "list":
                    await message.channel.send(f"Available reasoners: {', '.join(sorted(valid))}. Current: {runtime.chosen_profile.value}")
                    return
                if len(parts) == 2 and parts[1] in valid:
                    runtime.chosen_profile = ReasonerProfile(parts[1])
                    try:
                        if os.getenv("JENTIC_AGENT_API_KEY"):
                            runtime.current_agent = _build_agent(runtime.chosen_profile)
                            await message.channel.send(f"Reasoner set to {runtime.chosen_profile.value} and agent reloaded.")
                        else:
                            await message.channel.send(f"Reasoner set to {runtime.chosen_profile.value}. Configure key first (see README).")
                    except Exception as exc:
                        logger.error("reasoner_switch_failed", error=str(exc))
                        await message.channel.send(f"Failed to switch profile: {exc}")
                    return
                await message.channel.send("Usage: @Bot reasoner <react|rewoo|list>")
                return

            if lower.startswith("configure"):
                # Warning: sharing keys in public channels is unsafe; prefer .env.
                try:
                    key = goal.split(" ", 1)[1].strip()
                except Exception:
                    await message.channel.send("Usage: @Bot configure <JENTIC_AGENT_API_KEY>. Note: sending secrets in-channel is unsafe.")
                    return
                if not key:
                    await message.channel.send("No key provided. Prefer configuring via environment variables.")
                    return
                os.environ["JENTIC_AGENT_API_KEY"] = key
                try:
                    runtime.current_agent = _build_agent(runtime.chosen_profile)
                    await message.channel.send("Agent configured. Consider deleting your message to hide the key.")
                except Exception as exc:
                    logger.error("agent_build_failed", error=str(exc))
                    await message.channel.send(f"Saved key, but failed to initialize agent: {exc}")
                return

            if lower == "kill":
                runtime.current_agent = None
                os.environ.pop("JENTIC_AGENT_API_KEY", None)
                logger.warning("agent_killed")
                await message.channel.send("Agent killed. API key cleared; new requests will be rejected until reconfigured.")
                return

            async with message.channel.typing():
                # Ensure agent exists or prompt for configuration
                if runtime.current_agent is None:
                    if not os.getenv("JENTIC_AGENT_API_KEY"):
                        await message.channel.send("Not configured. Use @Bot configure <KEY> or set environment variable.")
                        return
                    runtime.current_agent = _build_agent(runtime.chosen_profile)
                # Bridge sync agent call to a thread to avoid blocking the loop
                result = await asyncio.to_thread(runtime.current_agent.solve, goal)

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


