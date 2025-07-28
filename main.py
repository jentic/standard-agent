#!/usr/bin/env python3

##############################################
#                                            #
#         HELLO WORLD REASONER               #
#                                            #
##############################################

import os
from dotenv import load_dotenv
from agents.prebuilt import ReWOOAgent
from utils.cli import read_user_goal, print_result

from utils.logger import get_logger, init_logger
logger = get_logger(__name__)


def main() -> None:
    init_logger("config.json")
    load_dotenv()

    agent = ReWOOAgent(model=os.getenv("LLM_MODEL", "claude-sonnet-4"))
    # Or assemble your own agent as follows:
    # agent = StandardAgent(
    #     llm = LiteLLM(model=os.getenv("LLM_MODEL", "claude-sonnet-4")),
    #     tools = JenticClient(),
    #     memory = DictMemory(),
    #     reasoner =ReWOOReasoner(),
    # )
    logger.info("🤖 Agent started. Enter goals to get started…")

    while True:
        goal_text = None
        try:
            goal_text = read_user_goal()
            if not goal_text:  # Skip empty inputs
                continue

            result = agent.solve(goal_text)
            print_result(result)

        except KeyboardInterrupt:
            logger.info("🤖 Bye!")
            break

        except Exception as exc:
            logger.exception("solve_failed", goal=goal_text, error=str(exc))


if __name__ == "__main__":
    main()
