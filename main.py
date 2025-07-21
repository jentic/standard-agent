#!/usr/bin/env python3
import logging, os, time
from dotenv import load_dotenv
from jentic_agents.inbox.cli_inbox                      import CLIInbox
from jentic_agents.outbox.cli_outbox                    import CLIOutbox
from jentic_agents.memory.scratch_pad                   import ScratchPadMemory
from jentic_agents.platform.jentic_client               import JenticClient
from jentic_agents.platform.jentic_tool_iface           import JenticToolInterface
from jentic_agents.utils.llm                            import LiteLLMChatLLM
from jentic_agents.reasoners.pre_built_reasoners        import ReWOOReasoner
from jentic_agents.agents.base_agent                    import BaseAgent

POLL_DELAY = 2.0   # seconds when inbox empty

def build_agent() -> any:
    llm     = LiteLLMChatLLM(model=os.getenv("LLM_MODEL", "gpt-4o"))
    tools   = JenticToolInterface(client=JenticClient())
    memory  = ScratchPadMemory()

    reasoner = ReWOOReasoner()

    inbox = CLIInbox(prompt="Enter your goal: ")
    outbox = CLIOutbox()

    return BaseAgent(
        llm=llm,
        tools=tools,
        memory=memory,
        reasoner=reasoner,
    ), inbox, outbox


def main() -> None:
    load_dotenv()
    logging.basicConfig(level=logging.INFO)

    agent, inbox, outbox = build_agent()
    logging.info("Agent service started. Polling for goals…")

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
            time.sleep(5)                       # avoid tight crash loop


if __name__ == "__main__":
    main()
