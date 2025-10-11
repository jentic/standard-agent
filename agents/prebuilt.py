import os
from agents.standard_agent import StandardAgent
from agents.tools.jentic import JenticClient
from agents.memory.dict_memory import DictMemory
from agents.reasoner.rewoo import ReWOOReasoner
from agents.reasoner.react import ReACTReasoner
from agents.llm.litellm import LiteLLM
from agents.llm.bedrock import BedrockLLM
from agents.goal_preprocessor.conversational import ConversationalGoalPreprocessor


def _validate_litellm_environment(model: str | None = None) -> None:
    """
    Validate environment variables for LiteLLM based on the model being used.
    
    LiteLLM supports many providers and this function checks for the appropriate
    API key based on the model prefix or common environment variables.
    
    Args:
        model: The model string which may indicate the provider
        
    Raises:
        ValueError: If required environment variables are missing
    """
    # If no model is specified, check for LLM_MODEL env var as per BaseLLM
    if not model:
        model = os.getenv("LLM_MODEL")
        if not model:
            # BaseLLM will handle this error, so we don't need to validate here
            return
    
    # Common environment variables that LiteLLM looks for
    # This is not exhaustive but covers the most common providers
    provider_env_vars = {
        "gpt": ["OPENAI_API_KEY"],
        "claude": ["ANTHROPIC_API_KEY"],
        "gemini": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
        "command": ["COHERE_API_KEY"],
        "llama": ["REPLICATE_API_TOKEN", "TOGETHER_API_KEY", "OPENAI_API_KEY"],
        "mistral": ["MISTRAL_API_KEY"],
        "azure": ["AZURE_API_KEY", "AZURE_API_BASE"],
        "bedrock": ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"],
        "vertex": ["GOOGLE_APPLICATION_CREDENTIALS", "VERTEX_PROJECT"],
    }
    
    # Try to determine provider from model name
    model_lower = model.lower()
    required_vars = []
    
    for provider_prefix, env_vars in provider_env_vars.items():
        if provider_prefix in model_lower:
            required_vars = env_vars
            break
    
    # If we can't determine the provider from the model name, 
    # check for any common API keys that might work
    if not required_vars:
        common_vars = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"]
        if any(os.getenv(var) for var in common_vars):
            # At least one common API key is set, assume it will work
            return
        else:
            # No common API keys found, provide helpful guidance
            raise ValueError(
                f"No API key found for model '{model}'. "
                f"Please set one of the following environment variables: "
                f"{', '.join(common_vars)}, or other provider-specific API keys. "
                f"See https://docs.litellm.ai/docs/providers for full list of supported providers."
            )
    
    # Check if any of the required environment variables are set
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars == required_vars:  # All required vars are missing
        raise ValueError(
            f"Missing required environment variables for model '{model}'. "
            f"Please set one of: {', '.join(required_vars)}"
        )


def _validate_bedrock_environment() -> None:
    """
    Validate environment variables for AWS Bedrock.
    
    Raises:
        ValueError: If required AWS credentials are missing
    """
    # Check for AWS credentials - Bedrock can use multiple auth methods
    aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    aws_bearer_token = os.getenv("AWS_BEARER_TOKEN_BEDROCK")
    aws_profile = os.getenv("AWS_PROFILE")
    
    # Check if we have any valid authentication method
    has_credentials = bool(aws_access_key and aws_secret_key)
    has_bearer_token = bool(aws_bearer_token)
    has_profile = bool(aws_profile)
    
    if not (has_credentials or has_bearer_token or has_profile):
        raise ValueError(
            "Missing AWS credentials for Bedrock. Please set one of the following:\n"
            "1. AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY\n"
            "2. AWS_BEARER_TOKEN_BEDROCK\n"
            "3. AWS_PROFILE (for profile-based authentication)\n"
            "Or configure AWS credentials through other methods (IAM roles, etc.)"
        )


def _validate_jentic_environment() -> None:
    """
    Validate environment variables for Jentic tools.
    
    Raises:
        ValueError: If required Jentic API key is missing
    """
    jentic_api_key = os.getenv("JENTIC_AGENT_API_KEY")
    
    if not jentic_api_key:
        raise ValueError(
            "Missing JENTIC_AGENT_API_KEY environment variable. "
            "Please set your Jentic API key. "
            "Get your agent API key by visiting: https://app.jentic.com"
        )


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
            
        Raises:
            ValueError: If required environment variables for the LLM provider are missing
        """
        # Validate environment variables for LiteLLM and Jentic
        _validate_litellm_environment(model)
        _validate_jentic_environment()
        
        # Initialize the core services
        llm = LiteLLM(model=model)

        tools = JenticClient()
        memory = DictMemory()
        reasoner = ReWOOReasoner(llm=llm, tools=tools, memory=memory, max_retries=max_retries)

        goal_processor = ConversationalGoalPreprocessor(llm=llm, memory=memory)

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
            
        Raises:
            ValueError: If required environment variables for the LLM provider are missing
        """
        # Validate environment variables for LiteLLM and Jentic
        _validate_litellm_environment(model)
        _validate_jentic_environment()
        
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

class ReWOOAgentBedrock(StandardAgent):
    """
    A pre-configured StandardAgent that uses the ReWOO reasoning methodology with AWS Bedrock.

    This agent combines:
    - BedrockLLM for language model access via AWS Bedrock
    - JenticClient for external tool access
    - DictMemory for state persistence
    - ReWOO sequential reasoner for planning, execution, and reflection
    - ConversationalGoalPreprocessor to enable conversational follow-ups.
    """

    def __init__(self, *, model: str | None = None, max_retries: int = 2):
        """
        Initialize the ReWOO agent with pre-configured components.

        Args:
            model: The Bedrock model ID to use (e.g., 'us.anthropic.claude-3-5-haiku-20241022-v1:0')
            max_retries: Maximum number of retries for the ReWOO reflector
            
        Raises:
            ValueError: If required AWS credentials for Bedrock are missing
        """
        # Validate environment variables for Bedrock and Jentic
        _validate_bedrock_environment()
        _validate_jentic_environment()
        
        # Initialize the core services
        llm = BedrockLLM(model=model)
        tools = JenticClient()
        memory = DictMemory()
        reasoner = ReWOOReasoner(llm=llm, tools=tools, memory=memory, max_retries=max_retries)

        goal_processor = ConversationalGoalPreprocessor(llm=llm, memory=memory)

        # Call parent constructor with assembled components
        super().__init__(
            llm=llm,
            tools=tools,
            memory=memory,
            reasoner=reasoner,
            goal_preprocessor=goal_processor,
        )


class ReACTAgentBedrock(StandardAgent):
    """
    A pre-configured StandardAgent that uses the ReACT reasoning methodology with AWS Bedrock.

    This agent combines:
    - BedrockLLM for language model access via AWS Bedrock
    - JenticClient for external tool access
    - DictMemory for state persistence
    - ReACT reasoner for think/act loop
    - ConversationalGoalPreprocessor to enable conversational follow-ups.
    """

    def __init__(self, *, model: str | None = None, max_turns: int = 20, top_k: int = 25):
        """
        Initialize the ReACT agent with pre-configured components.

        Args:
            model: The Bedrock model ID to use (e.g., 'us.anthropic.claude-3-5-haiku-20241022-v1:0')
            max_turns: Maximum number of think/act turns
            top_k: Number of tools to consider during selection
            
        Raises:
            ValueError: If required AWS credentials for Bedrock are missing
        """
        # Validate environment variables for Bedrock and Jentic
        _validate_bedrock_environment()
        _validate_jentic_environment()
        
        # Initialize the core services
        llm = BedrockLLM(model=model)
        tools = JenticClient()
        memory = DictMemory()
        reasoner = ReACTReasoner(llm=llm, tools=tools, memory=memory, max_turns=max_turns, top_k=top_k)

        goal_processor = ConversationalGoalPreprocessor(llm=llm, memory=memory)

        # Call parent constructor with assembled components
        super().__init__(
            llm=llm,
            tools=tools,
            memory=memory,
            reasoner=reasoner,
            goal_preprocessor=goal_processor,
        )
