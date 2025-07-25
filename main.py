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


def main() -> None:
    init_logger("config.json")

    load_dotenv()

    agent = get_rewoo_agent(model=os.getenv("LLM_MODEL", "claude-sonnet-4"))
    inbox = CLIInbox(prompt="🤖 Enter your goal: ")
    outbox = CLIOutbox()

    logger.info("🤖 Agent started. Polling for goals…")

    while True:
        try:
            goal_text = inbox.get_next_goal()
            if goal_text is None:
                time.sleep(POLL_DELAY)
                continue

            goal   = Goal(text=goal_text)
            result = agent.solve(goal)

            outbox.send(result)
            inbox.acknowledge_goal(goal_text)

        except KeyboardInterrupt:
            logger.info("🤖 Bye!")
            break

        except MissingAPIKeyError as exc:
            # Show friendly guidance (exc message may be LLM-generated)
            print(f"\n{exc}\n")
            env_var = exc.env_var
            api = exc.api_name or "API"
            while True:
                user_val = input(f"🔑 Provide {env_var} for {api} as {env_var}=<value>: ").strip()
                try:
                    key, val = map(str.strip, user_val.split("=", 1))
                    os.environ[key] = val
                    break
                except ValueError:
                    print("Format must be ENV_VAR=value")

            time.sleep(POLL_DELAY)

        except Exception as exc:
            logger.exception(f"🤖 Solve failed exception: {exc}")
            time.sleep(POLL_DELAY)


if __name__ == "__main__":
    main()
