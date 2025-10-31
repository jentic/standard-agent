#!/usr/bin/env python3

import os
import sys
from dotenv import load_dotenv

# Ensure project root is on sys.path so local imports work when running from examples/
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from agents.prebuilt import ReWOOAgentBedrock
from _cli_helpers import read_user_goal, print_result

from utils.logger import get_logger, init_logger
logger = get_logger(__name__)


def main() -> None:
    # Use absolute paths so this script is robust to the current working directory
    config_path = os.path.join(PROJECT_ROOT, "config.json")
    init_logger(config_path)
    load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

    agent = ReWOOAgentBedrock(model=os.getenv("LLM_MODEL", "eu.anthropic.claude-sonnet-4-20250514-v1:0"))
    # Or assemble your own agent as follows:
    # agent = StandardAgent(
    #     llm = BedrockLLM(model=os.getenv("LLM_MODEL", "eu.anthropic.claude-sonnet-4-20250514-v1:0")),
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
