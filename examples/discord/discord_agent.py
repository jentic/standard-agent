#!/usr/bin/env python3
import os
import re
import asyncio
import textwrap
from dataclasses import dataclass
from typing import Optional, List, Callable, Dict
from dotenv import load_dotenv

import discord
from discord import app_commands

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



AGENT_BUILDERS: Dict[str, Callable[[Optional[str]], StandardAgent]] = {
    "rewoo": lambda model: ReWOOAgent(model=model),
    "react": lambda model: ReACTAgent(model=model),
}

def list_profiles() -> List[str]:
    return sorted(AGENT_BUILDERS.keys())

@dataclass(slots=True)
class DiscordAgentRuntime:
    chosen_profile: str = "rewoo"
    current_agent: Optional[StandardAgent] = None
    bot_user_id: Optional[int] = None


def _build_agent(profile_key: str) -> StandardAgent:
    key = (profile_key or "").strip().lower()
    builder = AGENT_BUILDERS.get(key)
    if not builder:
        available = ", ".join(sorted(AGENT_BUILDERS.keys()))
        raise ValueError(f"Unknown agent profile: {profile_key}. Available: {available}")
    model = os.getenv("LLM_MODEL")
    logger.info("initializing_agent", profile=key, model=model)
    return builder(model)


def extract_goal_from_mention(content: str, bot_user_id: int) -> str:
    mention_regex = re.compile(rf"<@!?{bot_user_id}>") # Matches <@123> and <@!123>
    goal = mention_regex.sub("", content).strip()
    return goal


class KeyConfigModal(discord.ui.Modal, title="Configure Agent"):
    def __init__(self, runtime: DiscordAgentRuntime):
        super().__init__()
        self.runtime = runtime
        self.api_key = discord.ui.TextInput(
            label="Agent API Key",
            placeholder="Paste JENTIC AGENT API KEY from app.jentic.com",
            required=True,
            style=discord.TextStyle.short,
            max_length=200,
        )
        self.add_item(self.api_key)

    async def on_submit(self, interaction: discord.Interaction) -> None:  # type: ignore[override]
        key = (self.api_key.value or "").strip()
        if not key:
            await interaction.response.send_message("No key provided.", ephemeral=True)
            return
        try:
            os.environ["JENTIC_AGENT_API_KEY"] = key
            self.runtime.current_agent = _build_agent(self.runtime.chosen_profile)
            await interaction.response.send_message("Agent configured.", ephemeral=True)
        except Exception as exc:  # pragma: no cover
            logger.error("agent_build_failed", error=str(exc))
            await interaction.response.send_message(f"Saved key, but failed to initialize agent: {exc}", ephemeral=True)


async def main() -> None:
    # Logging + env
    init_logger()
    load_dotenv()

    token = os.getenv("DISCORD_BOT_TOKEN")
    if not token:
        raise RuntimeError("DISCORD_BOT_TOKEN is required in .env")

    # No channel allowlist; rely on Discord permissions + mention-gating

    intents = discord.Intents.default()
    intents.message_content = True
    intents.guilds = True
    intents.messages = True

    class DiscordAgentClient(discord.Client):
        def __init__(self, *, intents: discord.Intents):
            super().__init__(intents=intents)
            self.tree = app_commands.CommandTree(self)

    client = DiscordAgentClient(intents=intents)
    runtime = DiscordAgentRuntime()
    # Preload agent if key provided
    if os.getenv("JENTIC_AGENT_API_KEY"):
        try:
            runtime.current_agent = _build_agent(runtime.chosen_profile)
        except Exception as exc:
            logger.error("agent_init_on_boot_failed", error=str(exc))

    # Slash commands (app commands)
    standard_group = app_commands.Group(name="standard_agent", description="Configure Standard Agent")

    @standard_group.command(name="reasoner", description="Switch or list reasoning strategy")
    @app_commands.describe(mode="Choose reasoning strategy. Leave empty to list current/available.")
    async def reasoner(interaction: discord.Interaction, reasoning_strategy: Optional[str] = None):
        try:
            valid = set(list_profiles())
            if not reasoning_strategy:
                await interaction.response.send_message(
                    f"Available reasoners: {', '.join(sorted(valid))}. Current: {runtime.chosen_profile}",
                    ephemeral=True,
                )
                return
            reasoning_strategy = reasoning_strategy.lower().strip()
            if reasoning_strategy not in valid:
                await interaction.response.send_message("Usage: /standard_agent reasoner reasoning_strategy", ephemeral=True)
                return
            runtime.chosen_profile = reasoning_strategy
            if os.getenv("JENTIC_AGENT_API_KEY"):
                runtime.current_agent = _build_agent(runtime.chosen_profile)
                await interaction.response.send_message(f"Reasoner set to {runtime.chosen_profile} and agent reloaded.", ephemeral=True)
            else:
                await interaction.response.send_message(f"Reasoner set to {runtime.chosen_profile}. Configure key first via /standard_agent configure.", ephemeral=True,)
        except Exception as exc:  # pragma: no cover
            logger.error("reasoner_switch_failed", error=str(exc))
            await interaction.response.send_message(f"Failed to switch profile: {exc}", ephemeral=True)

    @standard_group.command(name="configure", description="Open a modal to configure the Agent API key")
    async def configure(interaction: discord.Interaction):
        try:
            await interaction.response.send_modal(KeyConfigModal(runtime))
        except Exception as exc:  # pragma: no cover
            logger.error("open_config_modal_failed", error=str(exc))
            await interaction.response.send_message(f"Failed to open config modal: {exc}", ephemeral=True)

    @standard_group.command(name="kill", description="Clear the API key and reset the agent")
    async def kill(interaction: discord.Interaction):
        runtime.current_agent = None
        os.environ.pop("JENTIC_AGENT_API_KEY", None)
        logger.warning("agent_killed")
        await interaction.response.send_message("Agent killed. API key cleared; new requests will be rejected until reconfigured.", ephemeral=True)

    client.tree.add_command(standard_group)

    @client.event
    async def on_ready():
        logger.info("discord_ready", user=str(client.user), user_id=getattr(client.user, "id", None))
        logger.info(f"Logged in as {client.user} (ID: {getattr(client.user, 'id', None)})")
        try:
            await client.tree.sync()
        except Exception as exc:  # pragma: no cover
            logger.error("discord_command_sync_failed", error=str(exc))

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

            # Extract goal (text after mention)
            content = message.content or ""
            goal = extract_goal_from_mention(content, bot_user.id)  # type: ignore[arg-type]

            if not goal:
                await message.channel.send("Please provide a goal after mentioning me.")
                return

            # Handle simple text commands
            goal = goal.lower().strip()
            if goal.startswith("reasoner"):
                parts = goal.split()
                valid = set(list_profiles())
                if len(parts) == 2 and parts[1] == "list":
                    await message.channel.send(f"Available reasoners: {', '.join(sorted(valid))}. Current: {runtime.chosen_profile}")
                    return
                if len(parts) == 2 and parts[1] in valid:
                    runtime.chosen_profile = parts[1]
                    try:
                        if os.getenv("JENTIC_AGENT_API_KEY"):
                            runtime.current_agent = _build_agent(runtime.chosen_profile)
                            await message.channel.send(f"Reasoner set to {runtime.chosen_profile} and agent reloaded.")
                        else:
                            await message.channel.send(f"Reasoner set to {runtime.chosen_profile}. Configure key first (see README).")
                    except Exception as exc:
                        logger.error("reasoner_switch_failed", error=str(exc))
                        await message.channel.send(f"Failed to switch profile: {exc}")
                    return
                await message.channel.send("Usage: @Bot reasoner <react|rewoo|list>")
                return

            if goal == "kill":
                runtime.current_agent = None
                os.environ.pop("JENTIC_AGENT_API_KEY", None)
                logger.warning("agent_killed")
                await message.channel.send("Agent killed. API key cleared; new requests will be rejected until reconfigured.")
                return

            async with message.channel.typing():
                # Ensure agent exists or prompt for configuration
                if runtime.current_agent is None:
                    if not os.getenv("JENTIC_AGENT_API_KEY"):
                        await message.channel.send("Not configured. Use /standard_agent configure to set the Agent API Key.")
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


