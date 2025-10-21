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

# New header files:
from discord.ext import commands
import time

logger = get_logger(__name__)

#Rate Limit Tracking
user_last_request = {}
RATE_LIMIT_SECONDS = 30 # Making user make requests every 30 sec


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

def is_rate_limited(user_id: int) -> bool:
    curr_time = time.time()
    if user_id in user_last_request:
        time_diff = curr_time - user_last_request[user_id]
        if time_diff < RATE_LIMIT_SECONDS:
            return True
    user_last_request[user_id] = curr_time
    return False

def get_rate_limit_remaining(user_id: int) -> float:
    if user_id not in user_last_request:
        return 0.0
    curr_time = time.time()
    elapsed = curr_time - user_last_request[user_id]
    remaining = max(0, RATE_LIMIT_SECONDS - elapsed)
    return remaining

async def safe_send_message(channel, content: str, max_retries: int = 3) -> bool:
    for attempt in range(max_retries):
        try:
            await channel.send(content)
            return True
        except discord.HTTPException as exc:
            if exc.status == 429: #Rate Limit exceeded
                retry_after = getattr(exc, 'retry_after', 1)
                logger.warning("discord_rate_limited", retry_after=retry_after, attempt=attempt+1)
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_after)
                    continue
                elif exc.status == 403: # Forbidden
                    logger.error("discord_forbidden", channel_id=getattr(channel, 'id', None))
                    return False
                elif exc.status >= 500:  # Server error
                    logger.warning("discord_server_error", status=exc.status, attempt=attempt + 1)
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff
                        continue
                logger.error("discord_http_error", status=exc.status, error=str(exc))
                return False
        except discord.Forbidden:
            logger.error("discord_forbidden_explicit", channel_id=getattr(channel, 'id', None))
            return False
        except Exception as exc:
            logger.error("unexpected_message_error", error=str(exc), attempt=attempt + 1)
            if attempt == max_retries - 1:
                return False
            await asyncio.sleep(1)
        return False


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
    try:
        key = (profile_key or "").strip().lower()
        builder = AGENT_BUILDERS.get(key)
        if not builder:
            available = ", ".join(sorted(AGENT_BUILDERS.keys()))
            raise ValueError(f"Unknown agent profile: {profile_key}. Available: {available}")
        model = os.getenv("LLM_MODEL")
        logger.info("initializing_agent", profile=key, model=model)
        return builder(model)
    except Exception as exc:
        logger.error("agent_build_error", error=str(exc))
        raise


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

    async def on_submit(self, interaction: discord.Interaction) -> None:
        try:
            key = self.api_key.value.strip()
            if not key:
                await interaction.response.send_message("❌ No API key provided.", ephemeral=True)
                return
            
            if len(key) < 10:
                await interaction.response.send_message("❌ API key appears to be too short (minimum 10 characters).", ephemeral=True)
                return
            
            if len(key) > 200:
                await interaction.response.send_message("❌ API key appears to be too long (maximum 200 characters).", ephemeral=True)
                return
            
            if not key.replace('-', '').replace('_', '').replace('.', '').isalnum():
                await interaction.response.send_message("❌ API key contains invalid characters. Only alphanumeric, hyphens, underscores, and dots are allowed.", ephemeral=True)
                return
            
            os.environ["JENTIC_AGENT_API_KEY"] = key

            try:
                self.runtime.current_agent = _build_agent(self.runtime.chosen_profile)
                await interaction.response.send_message(f"✅ Agent configured Sucessfully", ephemeral=True)
                logger.info("agent_configured_sucessfully", profile=self.runtime.chosen_profile)
            except ValueError as exc:
                await interaction.response.send_message(f"❌ Configuration error: {exc}", ephemeral=True)
            except Exception as exc:
                logger.error("agent_initialization_failed", error=str(exc))
                await interaction.response.send_message(f"⚠️ Key is saved but initialisation failed: {textwrap.shorten(str(exc), width=100, placeholder='...')}", ephemeral=True)

        except discord.InteractionResponded:
            logger.warning("interaction_already_responded")
        except Exception as exc:
            logger.error("submit_error", error=str(exc))                
            if not interaction.response.is_done():
                await interaction.response.send_message("❌ Unexpected error occurred while configuring the agent.", ephemeral=True)
            
    async def on_error(self, interaction: discord.Interaction, error: Exception) -> None:
        logger.error("modal_error", error=str(error))
        if not interaction.response.is_done():
            await interaction.response.send_message("❌ Unexpected error occurred.", ephemeral=True)



async def main() -> None:
    try:
        init_logger()
        load_dotenv()

        # Validate required environment variables
        token = os.getenv("DISCORD_BOT_TOKEN")
        if not token:
            raise RuntimeError("❌ DISCORD_BOT_TOKEN is required in .env file")

        # Setup Discord client
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
                logger.info("agent_preloaded_successfully", profile=runtime.chosen_profile)
            except Exception as exc:
                logger.error("agent_preload_failed", error=str(exc))

        # Slash commands (app commands)
        standard_group = app_commands.Group(name="standard_agent", description="Configure Standard Agent")

        @standard_group.command(name="reasoner", description="Switch or list reasoning strategy")
        @app_commands.describe(reasoning_strategy="Choose reasoning strategy. Leave empty to list current/available.")
        async def reasoner(interaction: discord.Interaction, reasoning_strategy: Optional[str] = None):
            try:
                valid = set(list_profiles())
                if not reasoning_strategy:
                    await interaction.response.send_message(
                        f"📋 **Available reasoners:** {', '.join(sorted(valid))}\n🔧 **Current:** {runtime.chosen_profile}",
                        ephemeral=True,
                    )
                    return
                
                reasoning_strategy = reasoning_strategy.lower().strip()
                if reasoning_strategy not in valid:
                    await interaction.response.send_message(
                        f"❌ Invalid reasoner '{reasoning_strategy}'. Available options: {', '.join(sorted(valid))}",
                        ephemeral=True
                    )
                    return
                
                old_profile = runtime.chosen_profile
                runtime.chosen_profile = reasoning_strategy
                
                # If agent is configured, reload it
                if os.getenv("JENTIC_AGENT_API_KEY"):
                    try:
                        runtime.current_agent = _build_agent(runtime.chosen_profile)
                        await interaction.response.send_message(
                            f"✅ Reasoner changed from **{old_profile}** to **{runtime.chosen_profile}** and agent reloaded successfully!",
                            ephemeral=True
                        )
                        logger.info("reasoner_changed_with_reload", old=old_profile, new=runtime.chosen_profile)
                    except Exception as exc:
                        # Rollback on failure
                        runtime.chosen_profile = old_profile
                        logger.error("reasoner_change_failed", new_profile=reasoning_strategy, error=str(exc))
                        await interaction.response.send_message(
                            f"❌ Failed to change reasoner: {textwrap.shorten(str(exc), width=150, placeholder='...')}",
                            ephemeral=True
                        )
                else:
                    await interaction.response.send_message(
                        f"✅ Reasoner set to **{runtime.chosen_profile}**. Use `/standard_agent configure` to set your API key.",
                        ephemeral=True
                    )
                    logger.info("reasoner_changed_no_reload", old=old_profile, new=runtime.chosen_profile)
                    
            except discord.InteractionResponded:
                logger.warning("reasoner_interaction_already_responded")
            except Exception as exc:
                logger.error("reasoner_command_error", error=str(exc))
                if not interaction.response.is_done():
                    await interaction.response.send_message("❌ An error occurred while changing the reasoner.", ephemeral=True)

        @standard_group.command(name="configure", description="Open a modal to configure the Agent API key")
        async def configure(interaction: discord.Interaction):
            try:
                await interaction.response.send_modal(KeyConfigModal(runtime))
            except discord.InteractionResponded:
                logger.warning("configure_interaction_already_responded")
            except Exception as exc:
                logger.error("configure_modal_error", error=str(exc))
                if not interaction.response.is_done():
                    await interaction.response.send_message("❌ Failed to open configuration modal.", ephemeral=True)

        @standard_group.command(name="kill", description="Clear the API key and reset the agent")
        async def kill(interaction: discord.Interaction):
            try:
                runtime.current_agent = None
                os.environ.pop("JENTIC_AGENT_API_KEY", None)
                logger.warning("agent_killed_by_user", user_id=interaction.user.id)
                await interaction.response.send_message(
                    "🔥 Agent terminated successfully! API key cleared - use `/standard_agent configure` to reconfigure.",
                    ephemeral=True
                )
            except discord.InteractionResponded:
                logger.warning("kill_interaction_already_responded")
            except Exception as exc:
                logger.error("kill_command_error", error=str(exc))
                if not interaction.response.is_done():
                    await interaction.response.send_message("❌ Failed to terminate agent.", ephemeral=True)

        client.tree.add_command(standard_group)

        # Global error handler for app commands
        @client.tree.error
        async def on_app_command_error(interaction: discord.Interaction, error: app_commands.AppCommandError):
            command_name = interaction.command.name if interaction.command else "unknown"
            logger.error("app_command_error", command=command_name, user_id=interaction.user.id, error=str(error))
            
            try:
                if isinstance(error, app_commands.CommandOnCooldown):
                    await interaction.response.send_message(
                        f"⏰ Command on cooldown. Try again in {error.retry_after:.1f} seconds.",
                        ephemeral=True
                    )
                elif isinstance(error, app_commands.MissingPermissions):
                    await interaction.response.send_message(
                        "❌ You don't have permission to use this command.",
                        ephemeral=True
                    )
                elif isinstance(error, app_commands.BotMissingPermissions):
                    await interaction.response.send_message(
                        "❌ Bot is missing required permissions to execute this command.",
                        ephemeral=True
                    )
                elif isinstance(error, app_commands.CommandInvokeError):
                    await interaction.response.send_message(
                        "❌ An internal error occurred while processing the command.",
                        ephemeral=True
                    )
                else:
                    if not interaction.response.is_done():
                        await interaction.response.send_message(
                            "❌ An unexpected error occurred. Please try again later.",
                            ephemeral=True
                        )
            except Exception as exc:
                logger.error("error_handler_failed", error=str(exc))

        @client.event
        async def on_ready():
            try:
                user_info = f"{client.user} (ID: {getattr(client.user, 'id', None)})"
                logger.info("discord_ready", user=str(client.user), user_id=getattr(client.user, "id", None))
                print(f"✅ Logged in as {user_info}")
                
                # Sync commands with error handling
                try:
                    synced = await client.tree.sync()
                    logger.info("commands_synced", count=len(synced))
                    print(f"✅ Synced {len(synced)} command(s)")
                except discord.HTTPException as exc:
                    if exc.status == 429:
                        retry_after = getattr(exc, 'retry_after', 60)
                        logger.warning("command_sync_rate_limited", retry_after=retry_after)
                        print(f"⚠️ Command sync rate limited. Retry after {retry_after} seconds.")
                    else:
                        logger.error("command_sync_http_error", status=exc.status, error=str(exc))
                        print(f"❌ HTTP error during command sync: {exc}")
                except Exception as exc:
                    logger.error("command_sync_failed", error=str(exc))
                    print(f"❌ Failed to sync commands: {exc}")
                    
            except Exception as exc:
                logger.error("on_ready_error", error=str(exc))

        @client.event
        async def on_error(event: str, *args, **kwargs):
            logger.error("discord_event_error", event=event, args=str(args)[:200], kwargs=str(kwargs)[:200])

        @client.event
        async def on_disconnect():
            logger.warning("discord_disconnected")
            print("⚠️ Bot disconnected from Discord. Attempting to reconnect...")

        @client.event  
        async def on_resumed():
            logger.info("discord_resumed")
            print("✅ Bot reconnected to Discord successfully")
            
        @client.event
        async def on_connect():
            logger.info("discord_connected")
            print("🔗 Bot connected to Discord")

        @client.event
        async def on_message(message: discord.Message):
            try:
                # Ignore self and other bots
                if message.author.bot:
                    return

                # Only respond if bot is mentioned (mention-gated)
                bot_user = client.user
                if bot_user is None:
                    logger.warning("bot_user_none_in_message_handler")
                    return
                    
                mentioned = bot_user in getattr(message, "mentions", [])
                if not mentioned:
                    return

                # Check rate limiting
                user_id = message.author.id
                if is_rate_limited(user_id):
                    remaining = get_rate_limit_remaining(user_id)
                    await safe_send_message(
                        message.channel, 
                        f"⏰ Please wait {remaining:.1f} seconds before making another request."
                    )
                    logger.info("user_rate_limited", user_id=user_id, remaining=remaining)
                    return

                # Extract goal (text after mention)
                content = message.content or ""
                goal = re.sub(rf"^\s*<@!?{bot_user.id}>\s*", "", content).lstrip(":,;.- ").strip()

                if not goal:
                    await safe_send_message(message.channel, "❓ Please provide a goal after mentioning me. For example: `@bot help me with something`")
                    return

                # Validate input length and content
                if len(goal) > 1500:
                    await safe_send_message(message.channel, "❌ Your request is too long. Please keep it under 1500 characters.")
                    return
                
                if len(goal.strip()) == 0:
                    await safe_send_message(message.channel, "❌ Please provide a non-empty request after mentioning me.")
                    return

                # Check for potentially harmful content
                if any(word in goal.lower() for word in ['hack', 'exploit', 'malware', 'virus']):
                    await safe_send_message(message.channel, "❌ I cannot help with potentially harmful activities.")
                    logger.warning("harmful_content_detected", user_id=user_id, goal=goal[:100])
                    return

                async with message.channel.typing():
                    try:
                        # Initialize agent if needed
                        if runtime.current_agent is None:
                            if not os.getenv("JENTIC_AGENT_API_KEY"):
                                await safe_send_message(
                                    message.channel, 
                                    "⚙️ Agent not configured. Use `/standard_agent configure` to set up your API key."
                                )
                                return
                            
                            try:
                                runtime.current_agent = _build_agent(runtime.chosen_profile)
                                logger.info("agent_initialized_on_demand", profile=runtime.chosen_profile)
                            except Exception as exc:
                                logger.error("agent_initialization_on_demand_failed", error=str(exc))
                                await safe_send_message(
                                    message.channel, 
                                    f"❌ Failed to initialize agent: {textwrap.shorten(str(exc), width=200, placeholder='...')}"
                                )
                                return
                        
                        # Process the goal with timeout
                        try:
                            result = await asyncio.wait_for(
                                asyncio.to_thread(runtime.current_agent.solve, goal),
                                timeout=300.0  # 5 minutes timeout
                            )
                            
                            final = result.final_answer or "❓ No answer was generated."
                            
                            # Send response in chunks
                            chunks = chunk(final, 1900)  # Leave room for formatting
                            success_count = 0
                            
                            for i, part in enumerate(chunks):
                                if i == 0:
                                    formatted_part = f"💡 **Answer:**\n{part}"
                                else:
                                    formatted_part = f"📄 **Continued ({i+1}/{len(chunks)}):**\n{part}"
                                
                                success = await safe_send_message(message.channel, formatted_part)
                                if success:
                                    success_count += 1
                                else:
                                    logger.error("failed_to_send_chunk", chunk_index=i, total_chunks=len(chunks))
                                    break
                            
                            if success_count > 0:
                                logger.info("goal_processed_successfully", 
                                          user_id=user_id, goal_length=len(goal), 
                                          response_length=len(final), chunks_sent=success_count)
                            else:
                                logger.error("no_chunks_sent_successfully", user_id=user_id)
                                
                        except asyncio.TimeoutError:
                            logger.error("agent_timeout", user_id=user_id, goal=goal[:100])
                            await safe_send_message(
                                message.channel, 
                                "⏰ Your request timed out (5 minutes). Please try a simpler question or break it into smaller parts."
                            )
                        except ValueError as exc:
                            logger.error("agent_validation_error", user_id=user_id, error=str(exc))
                            await safe_send_message(
                                message.channel, 
                                f"❌ Invalid request: {textwrap.shorten(str(exc), width=300, placeholder='...')}"
                            )
                        except Exception as exc:
                            logger.error("agent_processing_error", user_id=user_id, error=str(exc), goal=goal[:100])
                            await safe_send_message(
                                message.channel, 
                                f"❌ Failed to process your request: {textwrap.shorten(str(exc), width=250, placeholder='...')}"
                            )

                    except Exception as exc:
                        logger.error("message_processing_critical_error", user_id=user_id, error=str(exc))
                        await safe_send_message(
                            message.channel, 
                            "❌ A critical error occurred. Please try again later or contact support."
                        )

            except discord.Forbidden:
                logger.error("discord_forbidden_in_channel", channel_id=getattr(message.channel, 'id', None))
            except discord.HTTPException as exc:
                logger.error("discord_http_error_in_message", status=exc.status, error=str(exc))
            except Exception as exc:
                logger.exception("on_message_critical_error", error=str(exc))

        # Start the bot
        try:
            await client.start(token)
        except discord.LoginFailure:
            logger.error("discord_login_failed")
            raise RuntimeError("❌ Invalid Discord bot token")
        except discord.HTTPException as exc:
            logger.error("discord_startup_http_error", error=str(exc))
            raise RuntimeError(f"❌ Discord HTTP error during startup: {exc}")
        except discord.ConnectionClosed as exc:
            logger.error("discord_connection_closed", error=str(exc))
            raise RuntimeError(f"❌ Discord connection closed during startup: {exc}")
        except OSError as exc:
            logger.error("discord_network_error", error=str(exc))
            raise RuntimeError(f"❌ Network error during startup: {exc}")
            
    except KeyboardInterrupt:
        logger.info("bot_shutdown_requested")
        print("\n👋 Bot shutdown requested by user")
    except Exception as exc:
        logger.error("main_function_critical_error", error=str(exc))
        print(f"❌ Critical error in main function: {exc}")
        raise


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n Shutdown Completed.")
    except Exception as exc:
        logger.error("application_startup_failed", error=str(exc))
        print(f"❌ Application failed to start: {exc}")
        exit(1)


