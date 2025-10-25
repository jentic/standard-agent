from dotenv import load_dotenv
from agents.standard_agent import StandardAgent
from agents.llm.litellm import LiteLLM
from agents.memory.dict_memory import DictMemory
from agents.reasoner.react import ReACTReasoner
from .weather_tools import func_tools


load_dotenv()


# try changing to you own prefered model and add the API key in the .env file/ environment variable
llm = LiteLLM(max_tokens=50)

tools = func_tools
memory = DictMemory()

# Step 2: Pick a reasoner profile (single-file implementation)
custom_reasoner = ReACTReasoner(llm=llm, tools=tools, memory=memory, max_turns=5)

# Step 3: Wire everything together in the StandardAgent
weather_agent = StandardAgent(
    llm=llm,
    tools=tools,
    memory=memory,
    reasoner=custom_reasoner,
)
