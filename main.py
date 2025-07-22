#!/usr/bin/env python3
import logging, os, time
from dotenv import load_dotenv
from inbox.cli_inbox import CLIInbox
from outbox.cli_outbox import CLIOutbox
from memory.scratch_pad import ScratchPadMemory
from tools.jentic_toolkit.jentic_client import JenticClient
from tools.jentic_toolkit.jentic_tool_iface import JenticToolInterface
from utils.llm import LiteLLMChatLLM
from reasoners.pre_built_reasoners import ReWOOReasoner
from agents.standard_agent import StandardAgent

POLL_DELAY = 2.0

from utils.logger import get_logger, init_logger
logger = get_logger(__name__)

def build_agent() -> StandardAgent:
    llm     = LiteLLMChatLLM(model=os.getenv("LLM_MODEL", "gpt-4o"))
    tools   = JenticToolInterface(client=JenticClient())
    memory  = ScratchPadMemory()

    reasoner = ReWOOReasoner()

    return StandardAgent(
        llm=llm,
        tools=tools,
        memory=memory,
        reasoner=reasoner,
    )


def main() -> None:
    init_logger("config.json")

    load_dotenv()

    agent = build_agent()

    inbox = CLIInbox(prompt="Enter your goal: ")
    outbox = CLIOutbox()

    logger.info("Agent service started. Polling for goals…")

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
            logging.info("Graceful shutdown")
            break
        except Exception as exc:
            logging.exception("Unhandled error in agent loop: %s", exc)
            time.sleep(5)


if __name__ == "__main__":
    main()
