from __future__ import annotations

import os
import re
import sys
from typing import Optional

from dotenv import load_dotenv
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

from agents.prebuilt import ReWOOAgent, ReACTAgent
from utils.logger import get_logger


logger = get_logger(__name__)


def build_agent() -> ReWOOAgent | ReACTAgent:
    profile = (os.getenv("AGENT_PROFILE") or "rewoo").strip().lower()
    model = os.getenv("LLM_MODEL")
    if profile == "react":
        logger.info("initializing_react_agent", model=model)
        return ReACTAgent(model=model)
    logger.info("initializing_rewoo_agent", model=model)
    return ReWOOAgent(model=model)


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

    agent = build_agent()

    app = App(token=bot_token, signing_secret=signing_secret)

    bot_user_id: Optional[str] = None

    @app.event("app_mention")
    def handle_app_mention(event, say):  # type: ignore[no-redef]
        nonlocal bot_user_id
        try:
            text = event.get("text", "")
            channel = event.get("channel")
            thread_ts = event.get("ts")

            if bot_user_id is None:
                auth = app.client.auth_test()
                bot_user_id = auth.get("user_id")

            goal = extract_goal_from_text(text, bot_user_id)
            if not goal:
                say(text="Please provide a goal after mentioning me.", thread_ts=thread_ts)
                return

            logger.info("slack_goal_received", channel=channel, goal_preview=goal[:120])
            result = agent.solve(goal)
            answer = result.final_answer or "(No answer)"
            say(text=answer[:39000], thread_ts=thread_ts)
        except Exception as exc:  # pragma: no cover - best-effort guard
            logger.error("slack_app_mention_error", error=str(exc), exc_info=True)
            say(text=f"Error: {exc}")

    @app.message(re.compile(".*"))
    def handle_dm(message, say):  # type: ignore[no-redef]
        try:
            channel = message.get("channel")
            channel_is_dm = bool(channel) and str(channel).startswith("D")
            if not channel_is_dm:
                return

            text = message.get("text", "")
            goal = extract_goal_from_text(text, None)
            if not goal:
                say(text="Send me a goal to get started.")
                return

            logger.info("slack_dm_goal_received", channel=channel, goal_preview=goal[:120])
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


