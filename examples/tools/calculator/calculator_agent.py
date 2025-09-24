from typing import Any, Dict, List
import json
from agents.tools.base import ToolBase, JustInTimeToolingBase
from dotenv import load_dotenv
import logging

# Import the core agent class
from agents.standard_agent import StandardAgent

# Import different implementations for each layer
from agents.llm.litellm import LiteLLM
from agents.memory.dict_memory import DictMemory

# Import reasoner components
from agents.reasoner.react import ReACTReasoner
from examples._cli_helpers import read_user_goal, print_result

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CalculatorTool")


class CalculatorTool(ToolBase):
    TOOL_ID = "calculator.basic"

    def __init__(self) -> None:
        super().__init__(id=self.TOOL_ID)
        self.name = "Calculator"
        self.description = """
            This tool performs basic arithmetic operations (addition, subtraction, multiplication, division, percentage, etc.) 
            on two numbers at a time.
            It can be invoked multiple times in sequence to handle more complex calculations. 
            For example:
            Multiply two numbers, then subtract another number.
            Add two numbers, then calculate a percentage of the result.
            Chain operations step by step, such as multiply → add → divide.
            This tool accept argumets in son format
        """
        self._parameters: Dict[str, Any] = {
            "operation": {
                "type": "string",
                "enum": ["add", "subtract", "multiply", "divide", "percent_of"],
            },
            "a": {"type": "number"},
            "b": {"type": "number"},
        }
        self._schema: Dict[str, Any] = {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "parameters": self._parameters,
        }

    def get_summary(self) -> str:
        return f"{self.id}: {self.name} - {self.description} (API: local) parameters are supplied as json"

    def get_details(self) -> str:
        return json.dumps(self._schema, ensure_ascii=False, indent=2)

    def get_parameters(self) -> Dict[str, Any]:
        return self._parameters


class CalculatorTools(JustInTimeToolingBase):
    """
    Local provider that exposes the calculator tool via the standard interface.
    """

    def __init__(self) -> None:
        self._tool = CalculatorTool()

    def search(self, query: str, *, top_k: int = 10) -> List[ToolBase]:
        return [self._tool]

    def load(self, tool: ToolBase) -> ToolBase:
        if getattr(tool, "id", None) != self._tool.id:
            logger.warning(
                "calculator_load_mismatch", requested_id=getattr(tool, "id", None)
            )
        return self._tool

    def execute(self, tool: ToolBase, parameters: Dict[str, Any]) -> Any:
        if getattr(tool, "id", None) != self._tool.id:
            raise ValueError(f"Unknown tool: {tool}")

        op = str(parameters.get("operation", "")).strip().lower()

        try:
            a = float(parameters.get("a"))
            b = float(parameters.get("b"))
        except Exception:
            raise ValueError(
                "Parameters 'a' and 'b' must be numbers or numeric strings"
            )

        if op == "add":
            return a + b
        if op == "subtract":
            return a - b
        if op == "multiply":
            return a * b
        if op == "divide":
            if b == 0:
                raise ZeroDivisionError("Division by zero")
            return a / b
        if op == "percent_of":
            return (a / 100.0) * b

        raise ValueError(f"Unsupported operation: {op}")


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

# """What is the sum of 15 and 27?"""
# """What is 600/0?"""
# """First, multiply 23 by 7. Then subtract 45 from that result. What is the final answer? only give the number"""
# """What is 15% of 590?""
# """If I buy 3 notebooks at ₹45 each and a pen for ₹30, how much will I spend in total?"""
# """I had ₹2,000. After paying ₹1,250 for groceries, how much do I have left?"""
# """A shirt costs ₹799. If I buy 2 of them and get ₹200 off, what’s the final bill?"""
# """I worked 7.5 hours yesterday and 6 hours today. How many total hours is that?"""
# """If it takes 15 minutes to make one sandwich, how many can I make in 2 hours?"""
# """A recipe needs 250 g of sugar. If I want to make half the recipe, how much sugar do I need?"""
# """If I buy a 12-pack of eggs and use 7, how many are left?"""
# """One pizza is cut into 8 slices. If 3 people share equally, how many slices per person?"""
# """If I drive 60 km in 1 hour, how far will I go in 2.5 hours?"""
# """A train travels 120 km in 2 hours. What’s the average speed?"""
# """I have 400 km to travel. If I drive at 80 km/h, how long will it take?"""
# """My monthly salary is ₹50,000. If rent is ₹15,000 and bills are ₹8,000, what’s left?"""
# """If a water tank holds 1,500 liters and I use 275 liters daily, how many days will it last?"""


if __name__ == "__main__":
    main()
