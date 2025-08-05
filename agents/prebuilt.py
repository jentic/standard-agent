from agents.standard_agent import StandardAgent
from agents.tools.jentic import JenticClient
from agents.memory.dict_memory import DictMemory
from agents.reasoner.prebuilt import ReWOOReasoner
from agents.llm.litellm import LiteLLM
from agents.goal_processor.implicit_goal_resolver import ImplicitGoalResolver


class ReWOOAgent(StandardAgent):
    """
    A pre-configured StandardAgent that uses the ReWOO reasoning methodology.

    This agent combines:
    - LiteLLM for language model access
    - JenticClient for external tool access
    - DictMemory for state persistence
    - ReWOO sequential reasoner for planning, execution, and reflection
    """

    def __init__(self, *, model: str | None = None, max_retries: int = 2):
        """
        Initialize the ReWOO agent with pre-configured components.

        Args:
            model: The language model to use (defaults to LiteLLM's default)
            max_retries: Maximum number of retries for the ReWOO reflector
        """
        # Initialize the core services
        llm = LiteLLM(model=model or "claude-sonnet-4")
        tools = JenticClient()
        memory = DictMemory()
        reasoner = ReWOOReasoner(llm=llm, tools=tools, memory=memory, max_retries=max_retries)

        goal_processor = ImplicitGoalResolver(llm)

        # Call parent constructor with assembled components
        super().__init__(
            llm=llm,
            tools=tools,
            memory=memory,
            reasoner=reasoner,
            goal_resolver=goal_processor,
        )
