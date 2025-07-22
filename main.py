#!/usr/bin/env python3
import logging, os, time
from dotenv import load_dotenv
from inbox.cli_inbox import CLIInbox
from outbox.cli_outbox import CLIOutbox
from agents.prebuilt_agents import get_rewoo_agent

POLL_DELAY = 2.0

from utils.logger import get_logger, init_logger
logger = get_logger(__name__)


def main() -> None:
    init_logger("config.json")

    load_dotenv()

    agent = get_rewoo_agent(model=os.getenv("LLM_MODEL", "claude-sonnet-4"))

    inbox = CLIInbox(prompt="🤖 Enter your goal: ")
    outbox = CLIOutbox()

    logger.info("Agent started. Polling for goals…")

    while True:
        try:
            processed = agent.tick(inbox, outbox)
            if not processed:
                time.sleep(POLL_DELAY)
        except ImportError as e:
            print(f"❌ ERROR: A required package is not installed. {e}")
            print("Please make sure you have run 'make install'.")
            break
        except KeyboardInterrupt:
            logging.info("🤖Bye!")
            break
        except Exception as exc:
            logging.exception("Unhandled error in agent loop: %s", exc)
            time.sleep(5)


if __name__ == "__main__":
    main()
