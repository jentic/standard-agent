from __future__ import annotations

import os
import re
import sys
from typing import Optional

from dotenv import load_dotenv
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

from agents.standard_agent import StandardAgent
from agents.prebuilt import ReWOOAgent, ReACTAgent
from utils.logger import get_logger


logger = get_logger(__name__)


def build_agent(jentic_key: Optional[str] = None, profile: Optional[str] = None) -> StandardAgent:
    """Construct a single global StandardAgent, with optional Jentic key and reasoner profile."""
    # Ensure an asyncio event loop exists in this worker thread (for Jentic SDK)
    import asyncio
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    chosen_profile = (profile or "rewoo").strip().lower()
    model = os.getenv("LLM_MODEL")

    # Temporarily override env key during construction if provided
    prev = os.environ.get("JENTIC_AGENT_API_KEY")
    try:
        if jentic_key is not None:
            os.environ["JENTIC_AGENT_API_KEY"] = jentic_key

        if chosen_profile == "react":
            logger.info("initializing_react_agent", model=model)
            return ReACTAgent(model=model)
        logger.info("initializing_rewoo_agent", model=model)
        return ReWOOAgent(model=model)
    finally:
        if prev is None:
            os.environ.pop("JENTIC_AGENT_API_KEY", None)
        else:
            os.environ["JENTIC_AGENT_API_KEY"] = prev


def extract_goal_from_text(text: str, bot_user_id: Optional[str]) -> str:
    if not text:
        return ""
    cleaned = text
    if bot_user_id:
        mention_pattern = re.compile(rf"<@{re.escape(bot_user_id)}>\s*")
        cleaned = mention_pattern.sub("", cleaned).strip()
    return cleaned


def main() -> None:
    load_dotenv()

    app_token = os.getenv("SLACK_APP_TOKEN")
    bot_token = os.getenv("SLACK_BOT_TOKEN")
    signing_secret = os.getenv("SLACK_SIGNING_SECRET")

    if not app_token or not bot_token:
        print("Missing SLACK_APP_TOKEN or SLACK_BOT_TOKEN in environment.")
        sys.exit(1)

    # Single global agent, configured from env or via /standard-agent configure
    import json
    env_has_key = bool(os.getenv("JENTIC_AGENT_API_KEY"))
    configured_key: Optional[str] = None
    chosen_profile: str = "rewoo"
    current_agent: Optional[StandardAgent] = build_agent(None, chosen_profile) if env_has_key else None

    app = App(token=bot_token, signing_secret=signing_secret)

    bot_user_id: Optional[str] = None

    @app.command("/standard-agent")
    def on_cmd_standard_agent(ack, body, client, respond):  # type: ignore[no-redef]
        ack()
        text = (body.get("text") or "").strip()

        # Change reasoner profile: /standard-agent reasoner-profile <react|rewoo>
        if text.startswith("reasoner-profile"):
            parts = text.split()
            if len(parts) != 2 or parts[1].lower() not in {"react", "rewoo"}:
                respond(response_type="ephemeral", text="Usage: /standard-agent reasoner-profile <react|rewoo>")
                return
            nonlocal chosen_profile, current_agent
            chosen_profile = parts[1].lower()
            # Rebuild agent if we already have credentials
            try:
                if configured_key is not None or env_has_key:
                    current_agent = build_agent(configured_key, chosen_profile)
                    respond(response_type="ephemeral", text=f"Reasoner profile set to {chosen_profile} and agent reloaded.")
                else:
                    respond(response_type="ephemeral", text=f"Reasoner profile set to {chosen_profile}. Configure key via /standard-agent configure before use.")
            except Exception as exc:  # pragma: no cover
                respond(response_type="ephemeral", text=f"Failed to switch profile: {exc}")
            return

        if text == "configure":
            try:
                client.views_open(
                    trigger_id=body["trigger_id"],
                    view={
                        "type": "modal",
                        "callback_id": "configure_agent_view",
                        "private_metadata": json.dumps({}),
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
            user_id = body.get("user", {}).get("id")

            key = body["view"]["state"]["values"]["keyb"]["key"]["value"].strip()
            if not key:
                if user_id:
                    client.chat_postMessage(channel=user_id, text="No key provided.")
                return

            nonlocal configured_key, current_agent
            configured_key = key
            # Build/replace the single agent now to validate
            try:
                current_agent = build_agent(configured_key, chosen_profile)
            except Exception as exc:  # pragma: no cover
                logger.error("agent_build_failed", error=str(exc), exc_info=True)
                if user_id:
                    client.chat_postMessage(channel=user_id, text=f"Saved key, but failed to initialize agent: {exc}")
                return

            if user_id:
                client.chat_postMessage(channel=user_id, text="Agent configured.")
        except Exception as exc:  # pragma: no cover
            logger.error("config_submit_error", error=str(exc), exc_info=True)

    @app.event("app_mention")
    def handle_app_mention(event, say):  # type: ignore[no-redef]
        nonlocal bot_user_id
        try:
            text = event.get("text", "")
            channel = event.get("channel")
            thread_ts = event.get("ts")
            # team_id not needed for single-agent mode

            if bot_user_id is None:
                auth = app.client.auth_test()
                bot_user_id = auth.get("user_id")

            goal = extract_goal_from_text(text, bot_user_id)
            if not goal:
                say(text="Please provide a goal after mentioning me.", thread_ts=thread_ts)
                return

            # Ensure a single agent exists
            nonlocal current_agent
            if current_agent is None:
                if configured_key is None and not env_has_key:
                    say(text="Not configured. Run /standard-agent configure to set the Agent API Key.", thread_ts=thread_ts)
                    return
                current_agent = build_agent(configured_key, chosen_profile)

            logger.info("slack_goal_received", channel=channel, goal_preview=goal[:120])
            result = current_agent.solve(goal)
            answer = result.final_answer or "(No answer)"
            say(text=answer[:39000], thread_ts=thread_ts)
        except Exception as exc:  # pragma: no cover - best-effort guard
            logger.error("slack_app_mention_error", error=str(exc), exc_info=True)
            say(text=f"Error: {exc}")

    # @app.message(re.compile(".*"))
    # def handle_dm(message, say):  # type: ignore[no-redef]
    #     try:
    #         channel = message.get("channel")
    #         channel_is_dm = bool(channel) and str(channel).startswith("D")
    #         if not channel_is_dm:
    #             return
    #
    #         text = message.get("text", "")
    #         goal = extract_goal_from_text(text, None)
    #         if not goal:
    #             say(text="Send me a goal to get started.")
    #             return
    #
    #         # Ensure a single agent exists
    #         with agent_lock:
    #             if current_agent is None:
    #                 if configured_key is None and not env_has_key:
    #                     say(text="Not configured. Run /standard-agent configure to set the Agent API Key.")
    #                     return
    #                 current_agent = build_agent(configured_key)
    #
    #         logger.info("slack_dm_goal_received", channel=channel, goal_preview=goal[:120])
    #         result = current_agent.solve(goal)
    #         answer = result.final_answer or "(No answer)"
    #         say(text=answer[:39000])
    #     except Exception as exc:  # pragma: no cover - best-effort guard
    #         logger.error("slack_dm_error", error=str(exc), exc_info=True)
    #         say(text=f"Error: {exc}")

    handler = SocketModeHandler(app, app_token)
    logger.info("slack_socket_mode_starting")
    handler.start()


if __name__ == "__main__":
    main()


