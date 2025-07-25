#!/usr/bin/env python3

##############################################
#                                            #
#         HELLO WORLD REASONER               #
#                                            #
##############################################

import os, time

from dotenv import load_dotenv
from inbox.cli_inbox import CLIInbox
from outbox.cli_outbox import CLIOutbox
from agents.prebuilt_agents import get_rewoo_agent
from agents.models import Goal
from tools.exceptions import MissingAPIKeyError

POLL_DELAY = 2.0

from utils.logger import get_logger, init_logger
logger = get_logger(__name__)


def prompt_for_missing_api_key(exc: MissingAPIKeyError) -> tuple[str, str]:
    """
    Prompt the user to provide a missing API key.
    Optionally persist it to the `.env` file.
    """
    env_var = exc.env_var
    api = exc.api_name or "API"

    print(f"\n🔑 Missing key for {api}. Required: `{env_var}`")
    while True:
        user_val = input(f"Enter `{env_var}` as `{env_var}=<value>`: ").strip()
        try:
            key, val = map(str.strip, user_val.split("=", 1))
            os.environ[key] = val

            save = input("💾 Save this key to `.env` for future runs? (y/n): ").strip().lower()
            if save == "y":
                persist_api_key_to_env_file(key, val)
                print("✅ Saved to `.env`")

            return key, val
        except ValueError:
            print("❌ Format must be `ENV_VAR=value`. Try again.")


def persist_api_key_to_env_file(env_var: str, value: str) -> None:
    """
    Append or update an API key in the .env file.
    """
    from pathlib import Path

    env_path = Path(".env")
    lines = []

    if env_path.exists():
        lines = env_path.read_text().splitlines()
        lines = [line for line in lines if not line.startswith(f"{env_var}=")]

    lines.append(f"{env_var}={value}")
    env_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    init_logger("config.json")
    load_dotenv()

    agent = get_rewoo_agent(model=os.getenv("LLM_MODEL", "claude-sonnet-4"))
    inbox = CLIInbox(prompt="🤖 Enter your goal: ")
    outbox = CLIOutbox()

    logger.info("🤖 Agent started. Polling for goals…")

    retry_buffer: list[str] = []

    while True:
        goal_text = None
        try:
            if retry_buffer:
                goal_text = retry_buffer.pop(0)
            else:
                goal_text = inbox.get_next_goal()

            if goal_text is None:
                time.sleep(POLL_DELAY)
                continue

            goal = Goal(text=goal_text)
            result = agent.solve(goal)

            outbox.send(result)
            inbox.acknowledge_goal(goal_text)

        except KeyboardInterrupt:
            logger.info("🤖 Bye!")
            break

        except MissingAPIKeyError as exc:
            info = agent.get_pending_api_key_info()
            print(info.user_help_message)
            try:
                prompt_for_missing_api_key(exc)
                retry_buffer.append(goal_text)
            except KeyboardInterrupt:
                logger.info("⚠️ Aborted key entry.")
            time.sleep(POLL_DELAY)

        except Exception as exc:
            logger.exception(f"🤖 Solve failed exception: {exc}")
            inbox.reject_goal(goal_text, reason=str(exc))
            time.sleep(POLL_DELAY)

if __name__ == "__main__":
    main()
