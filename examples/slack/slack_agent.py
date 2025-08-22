from __future__ import annotations

"""A compact, readable Slack runtime for an OSS standard agent."""

import asyncio
import os
import re
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional

from dotenv import load_dotenv
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

from agents.prebuilt import ReACTAgent, ReWOOAgent
from agents.standard_agent import StandardAgent
from utils.logger import get_logger

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────

logger = get_logger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Configuration & Runtime
# ──────────────────────────────────────────────────────────────────────────────

class ReasonerProfile(str, Enum):
    REWOO = "rewoo"
    REACT = "react"


@dataclass(slots=True)
class SlackConfig:
    app_token: str
    bot_token: str
    signing_secret: Optional[str]

    @staticmethod
    def from_env() -> "SlackConfig":
        app_token = os.getenv("SLACK_APP_TOKEN", "").strip()
        bot_token = os.getenv("SLACK_BOT_TOKEN", "").strip()
        signing_secret = os.getenv("SLACK_SIGNING_SECRET")
        if not app_token or not bot_token:
            raise SystemExit("Missing SLACK_APP_TOKEN or SLACK_BOT_TOKEN in environment.")
        return SlackConfig(app_token, bot_token, signing_secret)


@dataclass(slots=True)
class Runtime:
    chosen_profile: ReasonerProfile = ReasonerProfile.REWOO
    current_agent: Optional[StandardAgent] = None
    bot_user_id: Optional[str] = None


# ──────────────────────────────────────────────────────────────────────────────
# Agent factories
# ──────────────────────────────────────────────────────────────────────────────

def _ensure_event_loop() -> None:
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())


def _build_agent(profile: ReasonerProfile) -> StandardAgent:
    _ensure_event_loop()
    model = os.getenv("LLM_MODEL")

    factories: dict[ReasonerProfile, Callable[[Optional[str]], StandardAgent]] = {
        ReasonerProfile.REWOO: lambda m: ReWOOAgent(model=m),
        ReasonerProfile.REACT: lambda m: ReACTAgent(model=m),
    }
    try:
        factory = factories[profile]
    except KeyError as e:  # pragma: no cover
        raise ValueError(f"Unknown reasoner profile: {profile}") from e

    logger.info("initializing_agent", profile=profile.value, model=model)
    return factory(model)


def _require_agent(runtime: Runtime) -> Optional[StandardAgent]:
    """Return an agent, building one iff API key exists; otherwise None."""
    if runtime.current_agent is not None:
        return runtime.current_agent
    if not os.getenv("JENTIC_AGENT_API_KEY"):
        return None
    runtime.current_agent = _build_agent(runtime.chosen_profile)
    return runtime.current_agent


def extract_goal(text: str, bot_user_id: Optional[str]) -> str:
    """Trim message and strip a leading <@BOT> mention if present."""
    if not text:
        return ""
    cleaned = text.strip()
    if bot_user_id:
        cleaned = re.sub(rf"^<@{re.escape(bot_user_id)}>\s*", "", cleaned)
    return cleaned


def is_dm(channel_id: Optional[str]) -> bool:
    return bool(channel_id) and str(channel_id).startswith("D")


# ──────────────────────────────────────────────────────────────────────────────
# Slack wiring
# ──────────────────────────────────────────────────────────────────────────────

def configure_handlers(app: App, runtime: Runtime) -> None:
    """Register all Slack handlers in one place."""

    @app.command("/standard-agent")
    def handle_command(ack, body, client, respond):  # type: ignore[no-redef]
        ack()
        text = (body.get("text") or "").strip()

        # Switch reasoner: /standard-agent reasoner <react|rewoo>
        if text.startswith("reasoner"):
            parts = text.split()
            if len(parts) != 2 or parts[1].lower() not in {p.value for p in ReasonerProfile}:
                respond(response_type="ephemeral", text="Usage: /standard-agent reasoner <react|rewoo>")
                return

            runtime.chosen_profile = ReasonerProfile(parts[1].lower())
            try:
                if os.getenv("JENTIC_AGENT_API_KEY"):
                    runtime.current_agent = _build_agent(runtime.chosen_profile)
                    respond(response_type="ephemeral", text=f"Reasoner set to {runtime.chosen_profile.value} and agent reloaded.")
                else:
                    respond(response_type="ephemeral", text=f"Reasoner set to {runtime.chosen_profile.value}. Configure key via /standard-agent configure before use.")
            except Exception as exc:  # pragma: no cover
                logger.error("reasoner_switch_failed", error=str(exc), exc_info=True)
                respond(response_type="ephemeral", text=f"Failed to switch profile: {exc}")
            return

        # Configure API key via modal
        if text == "configure":
            try:
                client.views_open(
                    trigger_id=body["trigger_id"],
                    view={
                        "type": "modal",
                        "callback_id": "configure_agent_view",
                        "title": {"type": "plain_text", "text": "Configure Agent"},
                        "submit": {"type": "plain_text", "text": "Save"},
                        "close": {"type": "plain_text", "text": "Cancel"},
                        "blocks": [
                            {
                                "type": "input",
                                "block_id": "keyb",
                                "label": {"type": "plain_text", "text": "Agent API Key"},
                                "element": {
                                    "type": "plain_text_input",
                                    "action_id": "key",
                                    "placeholder": {"type": "plain_text", "text": "Paste key from app.jentic.com"},
                                },
                            }
                        ],
                    },
                )
            except Exception as exc:  # pragma: no cover
                logger.error("open_config_modal_failed", error=str(exc), exc_info=True)
                respond(response_type="ephemeral", text=f"Failed to open config modal: {exc}")
            return

        respond(response_type="ephemeral", text="Usage: /standard-agent configure | /standard-agent reasoner <react|rewoo>")

    @app.view("configure_agent_view")
    def handle_config_submit(ack, body, client):  # type: ignore[no-redef]
        ack()
        try:
            user_id = body.get("user", {}).get("id")
            key = body["view"]["state"]["values"]["keyb"]["key"]["value"].strip()

            if not key:
                if user_id:
                    client.chat_postMessage(channel=user_id, text="No key provided.")
                return

            try:
                os.environ["JENTIC_AGENT_API_KEY"] = key
                runtime.current_agent = _build_agent(runtime.chosen_profile)
            except Exception as exc:  # pragma: no cover
                logger.error("agent_build_failed", error=str(exc), exc_info=True)
                if user_id:
                    client.chat_postMessage(channel=user_id, text=f"Saved key, but failed to initialize agent: {exc}")
                return

            if user_id:
                client.chat_postMessage(channel=user_id, text="Agent configured.")
        except Exception as exc:  # pragma: no cover
            logger.error("config_submit_error", error=str(exc), exc_info=True)

    def _answer(goal: str, say, thread_ts: Optional[str] = None) -> None:
        agent = _require_agent(runtime)
        if agent is None:
            say(text="Not configured. Run /standard-agent configure to set the Agent API Key.", thread_ts=thread_ts)
            return
        logger.info("agent_goal_received", preview=goal[:120])
        result = agent.solve(goal)
        say(text=(result.final_answer or "(No answer)")[:39000], thread_ts=thread_ts)

    @app.event("app_mention")
    def handle_mention(event, say):  # type: ignore[no-redef]
        try:
            if runtime.bot_user_id is None:
                auth = app.client.auth_test()
                runtime.bot_user_id = auth.get("user_id")

            goal = extract_goal(event.get("text", ""), runtime.bot_user_id)
            if not goal:
                say(text="Please provide a goal after mentioning me.", thread_ts=event.get("ts"))
                return
            _answer(goal, say, thread_ts=event.get("ts"))
        except Exception as exc:  # pragma: no cover
            logger.error("slack_app_mention_error", error=str(exc), exc_info=True)
            say(text=f"Error: {exc}")

    @app.message(re.compile(".*"))
    def handle_dm(message, say):  # type: ignore[no-redef]
        try:
            if not is_dm(message.get("channel")):
                return
            goal = extract_goal(message.get("text", ""), None)
            if not goal:
                say(text="Send me a goal to get started.")
                return
            _answer(goal, say)
        except Exception as exc:  # pragma: no cover
            logger.error("slack_dm_error", error=str(exc), exc_info=True)
            say(text=f"Error: {exc}")


def main() -> None:
    load_dotenv()
    config = SlackConfig.from_env()

    runtime = Runtime()
    if os.getenv("JENTIC_AGENT_API_KEY"):
        try:
            runtime.current_agent = _build_agent(runtime.chosen_profile)
        except Exception as exc:  # pragma: no cover
            logger.error("agent_init_on_boot_failed", error=str(exc), exc_info=True)

    app = App(token=config.bot_token, signing_secret=config.signing_secret)
    configure_handlers(app, runtime)

    logger.info("slack_socket_mode_starting")
    SocketModeHandler(app, config.app_token).start()


if __name__ == "__main__":
    try:
        main()
    except SystemExit as e:
        print(e, file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("Exiting…", file=sys.stderr)
