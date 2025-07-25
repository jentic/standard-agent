from agents.standard_agent import StandardAgent
from tools.jentic_toolkit.jentic_client import  JenticClient
from tools.jentic_toolkit.jentic_tool_iface import  JenticToolInterface
from memory.scratch_pad import ScratchPadMemory
from reasoners.pre_built_reasoners import ReWOOReasoner
from llm.lite_llm import LiteLLMChatLLM


# Prebuilt ReWOO agent
def get_rewoo_agent(model: str | None = None) -> StandardAgent:
    llm     = LiteLLMChatLLM()
    tools   = JenticToolInterface(client=JenticClient())
    memory  = ScratchPadMemory()

    reasoner = ReWOOReasoner()

    return StandardAgent(
        llm=llm,
        tools=tools,
        memory=memory,
        reasoner=reasoner,
    )