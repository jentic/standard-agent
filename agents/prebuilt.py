from agents.standard_agent import StandardAgent
from agents.tools.jentic import JenticClient
from agents.memory.dict_memory import DictMemory
from agents.reasoner.rewoo import ReWOOReasoner
from agents.reasoner.react import ReACTReasoner
from agents.llm.litellm import LiteLLM
from agents.goal_preprocessor.conversational import ConversationalGoalPreprocessor


class ReWOOAgent(StandardAgent):
    """
    A pre-configured StandardAgent that uses the ReWOO reasoning methodology.

    This agent combines:
    - LiteLLM for language model access
    - JenticClient for external tool access
    - DictMemory for state persistence
    - ReWOO sequential reasoner for planning, execution, and reflection
    - ConversationalGoalPreprocessor to enable conversational follow-ups.
    """

    def __init__(self, *, model: str | None = None, max_retries: int = 2):
        """
        Initialize the ReWOO agent with pre-configured components.

        Args:
            model: The language model to use.
            max_retries: Maximum number of retries for the ReWOO reflector
        """
        # Initialize the core services
        llm = LiteLLM(model=model)
        tools = JenticClient()
        memory = DictMemory()
        reasoner = ReWOOReasoner(llm=llm, tools=tools, memory=memory, max_retries=max_retries)

        goal_processor = ConversationalGoalPreprocessor(llm=llm)

        # Call parent constructor with assembled components
        super().__init__(
            llm=llm,
            tools=tools,
            memory=memory,
            reasoner=reasoner,
            goal_preprocessor=goal_processor,
        )


class ReACTAgent(StandardAgent):
    """
    A pre-configured StandardAgent that uses the ReACT reasoning methodology.

    This agent combines:
    - LiteLLM for language model access
    - JenticClient for external tool access
    - DictMemory for state persistence
    - ReACT reasoner for think/act loop
    - ConversationalGoalPreprocessor to enable conversational follow-ups.
    """

    def __init__(self, *, model: str | None = None, max_turns: int = 20, top_k: int = 25):
        """
        Initialize the ReACT agent with pre-configured components.

        Args:
            model: The language model to use (defaults to LiteLLM's default)
            max_turns: Maximum number of think/act turns
            top_k: Number of tools to consider during selection
        """
        # Initialize the core services
        llm = LiteLLM(model=model)
        tools = JenticClient()
        memory = DictMemory()
        reasoner = ReACTReasoner(llm=llm, tools=tools, memory=memory, max_turns=max_turns, top_k=top_k)

        goal_processor = ConversationalGoalPreprocessor(llm=llm)

        # Call parent constructor with assembled components
        super().__init__(
            llm=llm,
            tools=tools,
            memory=memory,
            reasoner=reasoner,
            goal_preprocessor=goal_processor,
        )
