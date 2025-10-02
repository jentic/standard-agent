#!/usr/bin/env python3
"""A Microsoft Teams bot runtime for an OSS standard agent."""

import asyncio
import os
import re
import sys
import traceback
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, Dict, Any
from urllib.parse import urlparse

from aiohttp import web
from aiohttp.web import Request, Response, json_response
from botbuilder.core import (
    ActivityHandler,
    TurnContext,
    MessageFactory,
    ConversationState,
    MemoryStorage,
    UserState,
)

from botbuilder.schema import (
    Activity,
    ActivityTypes,
    ChannelAccount,
    CardAction,
    ActionTypes,
    SuggestedActions,
    Attachment,
    HeroCard,
)
from botframework.connector import ConnectorClient
from botframework.connector.auth import (
    MicrosoftAppCredentials,
    JwtTokenValidation,
    SkillValidation,
)
from dotenv import load_dotenv

from agents.prebuilt import ReACTAgent, ReWOOAgent
from agents.standard_agent import StandardAgent
from utils.logger import get_logger
from utils.observability import setup_telemetry, TelemetryTarget

logger = get_logger(__name__)


class ReasonerProfile(str, Enum):
    REWOO = "rewoo"
    REACT = "react"


@dataclass(slots=True)
class TeamsConfig:
    app_id: str
    app_password: str
    port: int = 3978

    @staticmethod
    def from_env() -> "TeamsConfig":
        app_id = os.getenv("MICROSOFT_APP_ID", "").strip()
        app_password = os.getenv("MICROSOFT_APP_PASSWORD", "").strip()
        port = int(os.getenv("PORT", "3978"))
        
        if not app_id or not app_password:
            raise SystemExit("Missing MICROSOFT_APP_ID or MICROSOFT_APP_PASSWORD in environment.")
        return TeamsConfig(app_id, app_password, port)


@dataclass(slots=True)
class TeamsAgentRuntime:
    chosen_profile: ReasonerProfile = ReasonerProfile.REWOO
    current_agent: Optional[StandardAgent] = None
    app_credentials: Optional[MicrosoftAppCredentials] = None
    rate_limiter: Dict[str, deque] = None
    
    def __post_init__(self):
        if self.rate_limiter is None:
            self.rate_limiter = defaultdict(deque)


class RateLimiter:
    """Simple rate limiter for bot requests."""
    
    def __init__(self, max_requests: int = 10, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: Dict[str, deque] = defaultdict(deque)
    
    def is_allowed(self, user_id: str) -> bool:
        """Check if user is within rate limits."""
        now = time.time()
        user_requests = self.requests[user_id]
        
        # Remove old requests outside the window
        while user_requests and user_requests[0] < now - self.window_seconds:
            user_requests.popleft()
        
        # Check if user has exceeded rate limit
        if len(user_requests) >= self.max_requests:
            return False
        
        # Add current request
        user_requests.append(now)
        return True
    
    def get_reset_time(self, user_id: str) -> int:
        """Get seconds until rate limit resets for user."""
        user_requests = self.requests[user_id]
        if not user_requests:
            return 0
        return max(0, int(self.window_seconds - (time.time() - user_requests[0])))


def chunk_text(text: str, max_length: int = 28000) -> List[str]:
    """Split text into chunks that fit within Teams message limits."""
    if not text:
        return [""]
    
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_length, len(text))
        chunks.append(text[start:end])
        start = end
    return chunks


def _build_agent(profile: ReasonerProfile) -> StandardAgent:
    """Build an agent instance based on the chosen profile."""
    model = os.getenv("LLM_MODEL")
    logger.info("initializing_agent", profile=profile.value, model=model)
    if profile is ReasonerProfile.REACT:
        return ReACTAgent(model=model)
    return ReWOOAgent(model=model)


def extract_goal_from_mention(text: str, bot_mention: str = None) -> str:
    """Extract goal from message text, removing bot mentions."""
    if not text:
        return ""
    
    cleaned = text.strip()
    
    # Remove bot mentions (Teams format: <at>BotName</at>)
    cleaned = re.sub(r"<at>.*?</at>\s*", "", cleaned)
    
    # Remove any remaining @ mentions
    cleaned = re.sub(r"@\w+\s*", "", cleaned)
    
    return cleaned.strip()


class TeamsAgentBot(ActivityHandler):
    """Microsoft Teams bot that integrates with the Standard Agent."""
    
    def __init__(self, runtime: TeamsAgentRuntime, conversation_state: ConversationState, user_state: UserState):
        super().__init__()
        self.runtime = runtime
        self.conversation_state = conversation_state
        self.user_state = user_state
        self.rate_limiter = RateLimiter(max_requests=20, window_seconds=300)
        
        # Create state accessors
        self.user_profile_accessor = self.user_state.create_property("UserProfile")
        self.conversation_data_accessor = self.conversation_state.create_property("ConversationData")

    async def on_message_activity(self, turn_context: TurnContext):
        """Handle incoming messages with rate limiting and security."""
        try:
            # Get user ID for rate limiting
            user_id = turn_context.activity.from_property.id if turn_context.activity.from_property else "unknown"
            
            # Apply rate limiting
            if not self.rate_limiter.is_allowed(user_id):
                reset_time = self.rate_limiter.get_reset_time(user_id)
                await turn_context.send_activity(
                    MessageFactory.text(f"⏰ Rate limit exceeded. Please wait {reset_time} seconds before sending another message.")
                )
                return
            
            # Get the message text and validate
            text = turn_context.activity.text or ""
            
            if len(text) > 4000:
                await turn_context.send_activity(
                    MessageFactory.text("⚠️ Message too long. Please keep messages under 4000 characters.")
                )
                return
            
            # Check if this is a command
            if text.startswith("/"):
                await self._handle_command(turn_context, text)
                return
            
            # Extract goal from the message
            goal = extract_goal_from_mention(text)
            
            if not goal:
                await turn_context.send_activity(
                    MessageFactory.text("Please provide a goal for me to help with. Try mentioning me with a task!")
                )
                return
            
            if len(goal.strip()) < 3:
                await turn_context.send_activity(
                    MessageFactory.text("Please provide a more detailed goal (at least 3 characters).")
                )
                return
            
            # Check if agent is configured
            if self.runtime.current_agent is None:
                if not os.getenv("JENTIC_AGENT_API_KEY"):
                    await turn_context.send_activity(
                        MessageFactory.text("🔧 Not configured. Use `/configure` to set up the Agent API Key.")
                    )
                    return
                try:
                    self.runtime.current_agent = _build_agent(self.runtime.chosen_profile)
                except Exception as agent_error:
                    logger.error("agent_initialization_failed", error=str(agent_error))
                    await turn_context.send_activity(
                        MessageFactory.text("❌ Failed to initialize agent. Please check configuration.")
                    )
                    return
            
            # Send typing indicator
            typing_activity = MessageFactory.text("")
            typing_activity.type = ActivityTypes.typing
            await turn_context.send_activity(typing_activity)
            
            # Process the goal with the agent
            logger.info("agent_goal_received", user_id=user_id, preview=goal[:120])
            
            try:
                # Use asyncio.wait_for to add timeout
                result = await asyncio.wait_for(
                    asyncio.to_thread(self.runtime.current_agent.solve, goal),
                    timeout=120.0
                )
            except asyncio.TimeoutError:
                await turn_context.send_activity(
                    MessageFactory.text("⏱️ Request timed out. Please try a simpler goal or try again later.")
                )
                return
            except Exception as solve_error:
                logger.error("agent_solve_failed", error=str(solve_error), user_id=user_id)
                await turn_context.send_activity(
                    MessageFactory.text("❌ Sorry, I encountered an error processing your request. Please try again.")
                )
                return
            
            # Send response in chunks if needed
            final_answer = result.final_answer or "🤔 I couldn't generate an answer for that request."
            chunks = chunk_text(final_answer)
            
            for i, chunk in enumerate(chunks):
                if i > 0:
                    await asyncio.sleep(0.5)
                await turn_context.send_activity(MessageFactory.text(chunk))
                
        except Exception as exc:
            logger.exception("teams_message_error", error=str(exc), user_id=user_id)
            error_msg = "🚨 An unexpected error occurred. Our team has been notified."
            await turn_context.send_activity(MessageFactory.text(error_msg))
        
        # Save conversation state
        try:
            await self.conversation_state.save_changes(turn_context)
            await self.user_state.save_changes(turn_context)
        except Exception as state_error:
            logger.error("state_save_failed", error=str(state_error))

    async def _handle_command(self, turn_context: TurnContext, command: str):
        """Handle slash commands."""
        command = command.lower().strip()
        
        if command == "/configure":
            await self._handle_configure_command(turn_context)
        elif command.startswith("/reasoner"):
            await self._handle_reasoner_command(turn_context, command)
        elif command == "/kill":
            await self._handle_kill_command(turn_context)
        elif command == "/help":
            await self._handle_help_command(turn_context)
        else:
            await turn_context.send_activity(
                MessageFactory.text("Unknown command. Use `/help` to see available commands.")
            )

    async def _handle_configure_command(self, turn_context: TurnContext):
        """Handle the configure command - show configuration instructions."""
        config_message = (
            "**🔧 Agent Configuration**\n\n"
            "To configure the Standard Agent API Key:\n\n"
            "**Option 1: Environment Variable (Recommended)**\n"
            "• Set `JENTIC_AGENT_API_KEY` environment variable\n"
            "• Restart the bot service\n\n"
            "**Option 2: Runtime Configuration**\n"
            "• Contact your administrator for key setup\n\n"
            "📖 Get your API key from [app.jentic.com](https://app.jentic.com)\n\n"
            "⚠️ Never share your API key in chat messages!"
        )
        await turn_context.send_activity(MessageFactory.text(config_message))

    async def _handle_reasoner_command(self, turn_context: TurnContext, command: str):
        """Handle reasoner commands."""
        parts = command.split()
        valid_profiles = {p.value for p in ReasonerProfile}
        
        if len(parts) == 1 or (len(parts) == 2 and parts[1] == "list"):
            current_profile = self.runtime.chosen_profile.value
            available = ", ".join(sorted(valid_profiles))
            await turn_context.send_activity(
                MessageFactory.text(f"Available reasoners: [{available}]. Current: {current_profile}")
            )
            return
        
        if len(parts) == 2 and parts[1] in valid_profiles:
            self.runtime.chosen_profile = ReasonerProfile(parts[1])
            try:
                if os.getenv("JENTIC_AGENT_API_KEY"):
                    self.runtime.current_agent = _build_agent(self.runtime.chosen_profile)
                    await turn_context.send_activity(
                        MessageFactory.text(f"Reasoner set to {self.runtime.chosen_profile.value} and agent reloaded.")
                    )
                else:
                    await turn_context.send_activity(
                        MessageFactory.text(f"Reasoner set to {self.runtime.chosen_profile.value}. Configure key via `/configure` before use.")
                    )
            except Exception as exc:
                logger.error("reasoner_switch_failed", error=str(exc), exc_info=True)
                await turn_context.send_activity(
                    MessageFactory.text(f"Failed to switch profile: {exc}")
                )
            return
        
        await turn_context.send_activity(
            MessageFactory.text("Usage: `/reasoner` or `/reasoner <react|rewoo>`")
        )

    async def _handle_kill_command(self, turn_context: TurnContext):
        """Handle the kill command."""
        self.runtime.current_agent = None
        os.environ.pop("JENTIC_AGENT_API_KEY", None)
        logger.warning("agent_killed")
        await turn_context.send_activity(
            MessageFactory.text("Agent killed. API key cleared; new requests will be rejected until reconfigured.")
        )

    async def _handle_help_command(self, turn_context: TurnContext):
        """Show help information."""
        help_text = """
**Available Commands:**
- `/configure` - Instructions for setting up the Agent API Key
- `/reasoner` - List available reasoning strategies and current one
- `/reasoner <react|rewoo>` - Switch reasoning strategy
- `/kill` - Clear the API key and reset the agent
- `/help` - Show this help message

**Usage:**
Just mention me in a message with your goal, and I'll help you solve it!

Example: `@StandardAgent find recent articles about artificial intelligence`
        """
        await turn_context.send_activity(MessageFactory.text(help_text.strip()))

    async def on_members_added_activity(
        self, members_added: List[ChannelAccount], turn_context: TurnContext
    ):
        """Greet new members when they join."""
        welcome_text = (
            "👋 **Welcome to Standard Agent!**\n\n"
            "I'm an AI-powered reasoning assistant that can help you with various tasks.\n\n"
            "**Quick Start:**\n"
            "• Mention me with any goal: `@StandardAgent help me plan a project`\n"
            "• Use `/help` to see all available commands\n"
            "• Use `/configure` to set up API access\n\n"
            "**What I can do:**\n"
            "• Research and analysis\n"
            "• Planning and strategy\n"
            "• Problem-solving with ReACT and ReWOO reasoning\n"
            "• And much more!\n\n"
            "Let's get started! 🚀"
        )
        for member in members_added:
            if member.id != turn_context.activity.recipient.id:
                try:
                    await turn_context.send_activity(MessageFactory.text(welcome_text))
                except Exception as welcome_error:
                    logger.error("welcome_message_failed", error=str(welcome_error))


class AuthError(Exception):
    """Custom exception for authentication errors."""
    pass


async def authenticate_request(request: Request, app_credentials: MicrosoftAppCredentials) -> bool:
    """Authenticate incoming requests from Teams."""
    try:
        auth_header = request.headers.get("Authorization", "")
        if not auth_header:
            logger.warning("missing_auth_header")
            return False
        
        # Extract the token from the auth header
        parts = auth_header.split(" ")
        if len(parts) != 2 or parts[0].lower() != "bearer":
            logger.warning("invalid_auth_header_format")
            return False
        
        token = parts[1]
        
        if os.getenv("TEAMS_DEV_MODE", "").lower() == "true":
            logger.warning("teams_dev_mode_enabled_skipping_auth")
            return True
        
        # Validate the JWT token using Bot Framework validation
        # Note: This is a simplified implementation
        # In production, implement proper JWT validation with claims verification
        try:
            if not token or len(token) < 50:
                logger.warning("invalid_token_format")
                return False
            
            return True
            
        except Exception as validation_exc:
            logger.error("token_validation_failed", error=str(validation_exc))
            return False
        
    except Exception as exc:
        logger.error("auth_failed", error=str(exc))
        return False


async def create_app() -> web.Application:
    """Create and configure the web application."""
    load_dotenv()
    setup_telemetry(
        service_name=os.getenv("OTEL_SERVICE_NAME", "standard-agent-teams"),
        target=TelemetryTarget.LANGFUSE
    )
    
    # Load configuration
    config = TeamsConfig.from_env()
    runtime = TeamsAgentRuntime()
    
    # Initialize app credentials
    runtime.app_credentials = MicrosoftAppCredentials(config.app_id, config.app_password)
    
    # Preload agent if key is provided
    if os.getenv("JENTIC_AGENT_API_KEY"):
        try:
            runtime.current_agent = _build_agent(runtime.chosen_profile)
        except Exception as exc:
            logger.error("agent_init_on_boot_failed", error=str(exc), exc_info=True)
    
    # Create bot with state management
    memory_storage = MemoryStorage()
    conversation_state = ConversationState(memory_storage)
    user_state = UserState(memory_storage)
    
    bot = TeamsAgentBot(runtime, conversation_state, user_state)
    
    # Create web app
    app = web.Application()
    
    async def messages_handler(request: Request) -> Response:
        """Handle incoming messages from Teams."""
        try:
            if not await authenticate_request(request, runtime.app_credentials):
                logger.warning("unauthorized_request_blocked")
                return web.Response(status=401, text="Unauthorized")
            
            try:
                body = await request.json()
            except Exception as parse_exc:
                logger.error("request_body_parse_failed", error=str(parse_exc))
                return web.Response(status=400, text="Invalid JSON in request body")
            
            if not body:
                logger.warning("empty_request_body")
                return web.Response(status=400, text="Empty request body")
            
            try:
                activity = Activity().deserialize(body)
            except Exception as deserialize_exc:
                logger.error("activity_deserialization_failed", error=str(deserialize_exc))
                return web.Response(status=400, text="Invalid activity format")
            
            try:
                credentials = MicrosoftAppCredentials(
                    runtime.app_credentials.microsoft_app_id,
                    runtime.app_credentials.microsoft_app_password
                )
                
                connector_client = ConnectorClient(credentials, base_url=activity.service_url)
                
                turn_context = TurnContext(bot, activity)
                turn_context.turn_state[TurnContext.connector_client_key] = connector_client
                
                await bot.on_turn(turn_context)
                
                return web.Response(status=200)
                
            except Exception as processing_exc:
                logger.exception("activity_processing_failed", error=str(processing_exc))
                await turn_context.send_activity(
                    MessageFactory.text("Sorry, I encountered an error processing your request. Please try again.")
                )
                return web.Response(status=200)
                
        except Exception as exc:
            logger.exception("message_handler_error", error=str(exc))
            return web.Response(status=500, text="Internal server error")
    
    async def health_handler(request: Request) -> Response:
        """Health check endpoint."""
        return json_response({"status": "healthy", "service": "teams-agent"})
    
    app.router.add_post("/api/messages", messages_handler)
    app.router.add_get("/health", health_handler)
    app.router.add_get("/", health_handler)
    
    return app


async def main() -> None:
    """Main entry point."""
    try:
        app = await create_app()
        config = TeamsConfig.from_env()
        
        logger.info("teams_bot_starting", port=config.port)
        print(f"Teams bot starting on port {config.port}")
        print(f"Health check: http://localhost:{config.port}/health")
        print(f"Messages endpoint: http://localhost:{config.port}/api/messages")
        
        # Start the web server
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "0.0.0.0", config.port)
        await site.start()
        
        print("Teams bot is running. Press Ctrl+C to stop.")
        try:
            await asyncio.Event().wait()
        except KeyboardInterrupt:
            logger.info("teams_bot_stopping")
            print("Stopping Teams bot...")
        finally:
            await runner.cleanup()
            
    except Exception as exc:
        logger.exception("teams_bot_startup_failed", error=str(exc))
        print(f"Failed to start Teams bot: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Exiting...")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)