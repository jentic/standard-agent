from __future__ import annotations

import json
import os
import re
import sys
import threading
from typing import Dict, Optional, Tuple

from dotenv import load_dotenv
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

from agents.prebuilt import ReWOOAgent, ReACTAgent
from utils.logger import get_logger


logger = get_logger(__name__)


def extract_goal_from_text(text: str, bot_user_id: Optional[str]) -> str:
    if not text:
        return ""
    cleaned = text
    if bot_user_id:
        mention_pattern = re.compile(rf"<@{re.escape(bot_user_id)}>\s*")
        cleaned = mention_pattern.sub("", cleaned).strip()
    return cleaned


class AgentManager:
    """Manages per-(team_id, channel_id) StandardAgent instances and API keys."""

    def __init__(self) -> None:
        self._agents: Dict[Tuple[str, str], ReWOOAgent | ReACTAgent] = {}
        self._keys: Dict[Tuple[str, str], str] = {}
        self._lock = threading.Lock()

    def set_key(self, team_id: str, channel_id: str, api_key: str) -> None:
        with self._lock:
            self._keys[(team_id, channel_id)] = api_key
            # Drop any existing agent to force rebuild with the new key
            self._agents.pop((team_id, channel_id), None)

    def get_or_create(self, team_id: str, channel_id: str) -> ReWOOAgent | ReACTAgent:
        key = (team_id, channel_id)
        with self._lock:
            if key in self._agents:
                return self._agents[key]

        # build outside lock to avoid long lock hold
        agent = self._build_agent_for_channel(team_id, channel_id)
        with self._lock:
            self._agents[key] = agent
        return agent

    def _build_agent_for_channel(self, team_id: str, channel_id: str) -> ReWOOAgent | ReACTAgent:
        profile = (os.getenv("AGENT_PROFILE") or "rewoo").strip().lower()
        model = os.getenv("LLM_MODEL")
        api_key = self._keys.get((team_id, channel_id))

        # Temporarily override env only during construction so the underlying SDK
        # captures the token when the client is created, then restore it.
        prev = os.environ.get("JENTIC_AGENT_API_KEY")
        try:
            if api_key:
                os.environ["JENTIC_AGENT_API_KEY"] = api_key
            else:
                os.environ.pop("JENTIC_AGENT_API_KEY", None)

            if profile == "react":
                logger.info("initializing_react_agent", model=model, team_id=team_id, channel_id=channel_id)
                return ReACTAgent(model=model)
            logger.info("initializing_rewoo_agent", model=model, team_id=team_id, channel_id=channel_id)
            return ReWOOAgent(model=model)
        finally:
            if prev is None:
                os.environ.pop("JENTIC_AGENT_API_KEY", None)
            else:
                os.environ["JENTIC_AGENT_API_KEY"] = prev


def main() -> None:
    load_dotenv()

    app_token = os.getenv("SLACK_APP_TOKEN")
    bot_token = os.getenv("SLACK_BOT_TOKEN")
    signing_secret = os.getenv("SLACK_SIGNING_SECRET")

    if not app_token or not bot_token:
        print("Missing SLACK_APP_TOKEN or SLACK_BOT_TOKEN in environment.")
        sys.exit(1)

    manager = AgentManager()

    app = App(token=bot_token, signing_secret=signing_secret)

    bot_user_id: Optional[str] = None

    @app.command("/standard-agent")
    def on_cmd_standard_agent(ack, body, client, respond):  # type: ignore[no-redef]
        ack()
        team_id = body.get("team_id")
        channel_id = body.get("channel_id")
        text = (body.get("text") or "").strip()

        if text == "configure":
            try:
                client.views_open(
                    trigger_id=body["trigger_id"],
                    view={
                        "type": "modal",
                        "callback_id": "configure_agent_view",
                        "private_metadata": json.dumps({"team_id": team_id, "channel_id": channel_id}),
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

        respond(response_type="ephemeral", text="Usage: /standard-agent configure")

    @app.view("configure_agent_view")
    def on_config_submit(ack, body, client):  # type: ignore[no-redef]
        ack()
        try:
            meta = json.loads(body["view"]["private_metadata"]) if body["view"].get("private_metadata") else {}
            team_id = meta.get("team_id") or body.get("team", {}).get("id") or body.get("team_id")
            channel_id = meta.get("channel_id")
            user_id = body.get("user", {}).get("id")

            key = body["view"]["state"]["values"]["keyb"]["key"]["value"].strip()
            if not key:
                if channel_id and user_id:
                    client.chat_postEphemeral(channel=channel_id, user=user_id, text="No key provided.")
                return

            manager.set_key(str(team_id), str(channel_id), key)
            # Build the agent now to validate and warm it up
            try:
                _ = manager.get_or_create(str(team_id), str(channel_id))
            except Exception as exc:  # pragma: no cover
                logger.error("agent_build_failed", error=str(exc), exc_info=True)
                if channel_id and user_id:
                    client.chat_postEphemeral(channel=channel_id, user=user_id, text=f"Saved key, but failed to initialize agent: {exc}")
                return

            if channel_id and user_id:
                client.chat_postEphemeral(channel=channel_id, user=user_id, text="Agent configured for this channel.")
        except Exception as exc:  # pragma: no cover
            logger.error("config_submit_error", error=str(exc), exc_info=True)

    @app.event("app_mention")
    def handle_app_mention(event, say, body):  # type: ignore[no-redef]
        nonlocal bot_user_id
        try:
            text = event.get("text", "")
            channel = event.get("channel")
            thread_ts = event.get("ts")
            team_id = event.get("team") or body.get("team_id")

            if bot_user_id is None:
                auth = app.client.auth_test()
                bot_user_id = auth.get("user_id")

            goal = extract_goal_from_text(text, bot_user_id)
            if not goal:
                say(text="Please provide a goal after mentioning me.", thread_ts=thread_ts)
                return

            logger.info("slack_goal_received", channel=channel, team_id=team_id, goal_preview=goal[:120])
            agent = manager.get_or_create(str(team_id), str(channel))
            result = agent.solve(goal)
            answer = result.final_answer or "(No answer)"
            say(text=answer[:39000], thread_ts=thread_ts)
        except Exception as exc:  # pragma: no cover - best-effort guard
            logger.error("slack_app_mention_error", error=str(exc), exc_info=True)
            say(text=f"Error: {exc}")

    @app.message(re.compile(".*"))
    def handle_dm(message, say, body):  # type: ignore[no-redef]
        try:
            channel = message.get("channel")
            team_id = body.get("team_id")
            channel_is_dm = bool(channel) and str(channel).startswith("D")
            if not channel_is_dm:
                return

            text = message.get("text", "")
            goal = extract_goal_from_text(text, None)
            if not goal:
                say(text="Send me a goal to get started.")
                return

            logger.info("slack_dm_goal_received", channel=channel, team_id=team_id, goal_preview=goal[:120])
            agent = manager.get_or_create(str(team_id), str(channel))
            result = agent.solve(goal)
            answer = result.final_answer or "(No answer)"
            say(text=answer[:39000])
        except Exception as exc:  # pragma: no cover - best-effort guard
            logger.error("slack_dm_error", error=str(exc), exc_info=True)
            say(text=f"Error: {exc}")

    handler = SocketModeHandler(app, app_token)
    logger.info("slack_socket_mode_starting")
    handler.start()


if __name__ == "__main__":
    main()


