#!/usr/bin/env python3
import os
import sys

from dotenv import load_dotenv

from jentic_agents.outbox import CLIOutbox
from jentic_agents.reasoners import ReWOOReasoner

# Add the package to the path
sys.path.insert(0, os.path.dirname(__file__))

from jentic_agents.agents.interactive_cli_agent import InteractiveCLIAgent
from jentic_agents.inbox.cli_inbox import CLIInbox
from jentic_agents.memory.scratch_pad import ScratchPadMemory
from jentic_agents.platform.jentic_client import JenticClient
from jentic_agents.platform.jentic_tool_iface import JenticToolInterface
from jentic_agents.utils.logger import get_logger
from jentic_agents.utils.llm import LiteLLMChatLLM


def main():
    """Run the live demo."""
    # Load environment variables from .env file
    load_dotenv()
    logger = get_logger(__name__)

    print("🚀 Starting Standard Agent Live Demo")
    print("=" * 50)
    print("This agent uses live Jentic and OpenAI services.")
    print("Type your goal below, or 'quit' to exit.")
    print("-" * 50)

    try:
        # 1. Initialize the JenticClient
        # This will use the live Jentic services.
        jentic_client = JenticClient()
        tool_interface = JenticToolInterface(client=jentic_client)

        # 2. Initialize the LLM wrapper and Reasoner
        # LLM wrapper will read model from config (config.toml)
        llm_wrapper = LiteLLMChatLLM()
        memory = ScratchPadMemory()


        reasoner = ReWOOReasoner(
            tool=tool_interface,
            memory=memory,
            llm=llm_wrapper,
        )
        print('Initializing JenticReasoner')
        # 3. Initialize Memory and Inbox
        inbox = CLIInbox(prompt="Enter your goal: ")
        outbox = CLIOutbox()

        # 4. Create and run the Agent
        agent = InteractiveCLIAgent(
            reasoner=reasoner,
            memory=memory,
            inbox=inbox,
            outbox=outbox,
            jentic_client=jentic_client,
        )

        agent.spin()

    except ImportError as e:
        print(f"❌ ERROR: A required package is not installed. {e}")
        print("Please make sure you have run 'make install'.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        logger.error("An unexpected error occurred during the demo.", exc_info=True)
        sys.exit(1)

    print("-" * 50)
    print("👋 Demo finished. Goodbye!")


if __name__ == "__main__":
    main() 