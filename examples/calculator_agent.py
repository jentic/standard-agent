from typing import Any, Dict, List
from agents.tools.base import ToolBase, JustInTimeToolingBase
from dotenv import load_dotenv

# Import the core agent class
from agents.standard_agent import StandardAgent

# Import different implementations for each layer
from agents.llm.litellm import LiteLLM
from agents.memory.dict_memory import DictMemory

# Import reasoner components
from agents.reasoner.react import ReACTReasoner
from _cli_helpers import read_user_goal, print_result


# ---- Tool Implementations ----
class CalculatorTool(ToolBase):
    """Base class for calculator tools."""

    _id = "CalculatorBase"

    def __init__(self) -> None:
        super().__init__(self._id)

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "a": {
                "type": "int or float",
                "description": "The first number to operate with",
            },
            "b": {
                "type": "int or float",
                "description": "The second number to operate with",
            },
        }

    def validator(self, parameters: Dict[str, Any]) -> Any:
        a, b = parameters.get("a"), parameters.get("b")
        if not a or not b:
            return {"error": "Missing parameters 'a' or 'b'"}
        if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
            return {"error": "Parameters must be numbers"}
        return True


class AddTool(CalculatorTool):
    """A simple tool that adds two numbers."""

    _id = "Add"

    def get_summary(self) -> str:
        return f"{self.id}: This is calculator tool for adding two numbers. a+b"

    def get_details(self) -> str:
        return "Given parameters 'a' and 'b', returns their sum (a + b)."

    def execute(self, parameters: Dict[str, Any]) -> Any:
        validation = self.validator(parameters)
        if validation is not True:
            return validation
        a, b = parameters.get("a"), parameters.get("b")
        return {"result": a + b}


class SubTool(CalculatorTool):
    """A simple tool that adds two numbers."""

    _id = "Sub"

    def get_summary(self) -> str:
        return f"{self.id}: This is calculator tool for substract a numbers from another. like a-b"

    def get_details(self) -> str:
        return "Given parameters 'a' and 'b', returns their sum (a - b)."

    def execute(self, parameters: Dict[str, Any]) -> Any:
        validation = self.validator(parameters)
        if validation is not True:
            return validation
        a, b = parameters.get("a"), parameters.get("b")
        return {"result": a - b}


class MultiplyTool(CalculatorTool):
    """A simple tool that adds two numbers."""

    _id = "Multiply"

    def get_summary(self) -> str:
        return f"{self.id}: This is calculator tool for multiply two numbers. like a*b"

    def get_details(self) -> str:
        return "Given parameters 'a' and 'b', returns their sum (a * b)."

    def execute(self, parameters: Dict[str, Any]) -> Any:
        validation = self.validator(parameters)
        if validation is not True:
            return validation
        a, b = parameters.get("a"), parameters.get("b")
        return {"result": a * b}


class DivTool(CalculatorTool):
    """A simple tool that adds two numbers."""

    _id = "Divide"

    def get_summary(self) -> str:
        return f"{self.id}: This is calculator tool for divsion of two nmbers.like a/ b"

    def get_details(self) -> str:
        return "Given parameters 'a' and 'b', returns their sum (a / b)."

    def validator(self, parameters: Dict[str, Any]) -> Any:
        validation = super().validator(parameters)
        if validation is not True:
            return validation
        b = parameters.get("b")
        if b == 0:  # Check for division by zero
            return {"error": "Division by zero is not allowed"}
        return True

    def execute(self, parameters: Dict[str, Any]) -> Any:
        validation = self.validator(parameters)
        if validation is not True:
            return validation
        a, b = parameters.get("a"), parameters.get("b")
        return {"result": a / b}


class CalculatorTools(JustInTimeToolingBase):
    """A simple tool provider that offers basic calculator operations."""

    def __init__(self) -> None:
        self.tools = {
            AddTool._id: AddTool(),
            SubTool._id: SubTool(),
            MultiplyTool._id: MultiplyTool(),
            DivTool._id: DivTool(),
        }

    def search(self, query: str, *, top_k: int = 10) -> List[ToolBase]:
        """Search for tools matching a natural language query."""

        return self.tools.values()

    def load(self, tool: ToolBase) -> ToolBase:
        """Load the full specification for a single tool."""
        if tool.id in self.tools:
            return self.tools[tool.id]
        raise ValueError(f"Tool with id {tool.id} not found")

    def execute(self, tool: ToolBase, parameters: Dict[str, Any]) -> Any:
        """Execute a tool with the given parameters."""
        if tool.id in self.tools:
            return self.tools[tool.id].execute(parameters)
        raise ValueError(f"Tool with id {tool.id} not found")


load_dotenv()

# Step 1: Choose and configure your components
# try changing to you own prefered model and add the API key in the .env file/ environment variable
llm = LiteLLM(model="gemini/gemini-2.0-flash", max_tokens=50)
tools = CalculatorTools()
memory = DictMemory()

# Step 2: Pick a reasoner profile (single-file implementation)
custom_reasoner = ReACTReasoner(llm=llm, tools=tools, memory=memory)

# Step 3: Wire everything together in the StandardAgent
agent = StandardAgent(
    llm=llm,
    tools=tools,
    memory=memory,
    reasoner=custom_reasoner,
)


def main():
    # Step 4: Use your custom agent
    print("🤖 Custom Agent is ready! with calculator")
    while True:
        try:
            goal = read_user_goal()
            if not goal:
                continue

            result = agent.solve(goal)
            print_result(result)

        except KeyboardInterrupt:
            print("\n🤖 Bye!")
            break


# Example prompts to try out
# Each prompt should ideally be solvable using the provided tools
# You could adjust the tools summary to get more relevant results and 
#   - consideration of the LLM tool selection and parameter generation
# You can also try your own prompts!

prompts = [
    """What is the sum of 15 and 27?""",
    """First, multiply 23 by 7. Then subtract 45 from that result. What is the final answer? only give the number""",
    """What is 15% of 590?"""
    """If I buy 3 notebooks at ₹45 each and a pen for ₹30, how much will I spend in total?""",
    """I had ₹2,000. After paying ₹1,250 for groceries, how much do I have left?""",
    """A shirt costs ₹799. If I buy 2 of them and get ₹200 off, what’s the final bill?""",
    """I worked 7.5 hours yesterday and 6 hours today. How many total hours is that?""",
    """If it takes 15 minutes to make one sandwich, how many can I make in 2 hours?""",
    """The bus leaves at 3:45 pm and the journey takes 1 hour 20 minutes. What time will I reach?""",
    """A recipe needs 250 g of sugar. If I want to make half the recipe, how much sugar do I need?""",
    """If I buy a 12-pack of eggs and use 7, how many are left?""",
    """One pizza is cut into 8 slices. If 3 people share equally, how many slices per person?""",
    """If I drive 60 km in 1 hour, how far will I go in 2.5 hours?""",
    """A train travels 120 km in 2 hours. What’s the average speed?""",
    """I have 400 km to travel. If I drive at 80 km/h, how long will it take?""",
    """My monthly salary is ₹50,000. If rent is ₹15,000 and bills are ₹8,000, what’s left?""",
    """I bought 5 apples and 3 oranges. Apples cost ₹12 each, oranges ₹8 each. How much in total?""",
    """If a water tank holds 1,500 liters and I use 275 liters daily, how many days will it last?""",
]


if __name__ == "__main__":
    main()
