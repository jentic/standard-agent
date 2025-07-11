"""An outbox that formats and prints the result to the command line."""
from jentic_agents.reasoners.models import ReasoningResult
from .base_outbox import BaseOutbox

__all__ = ["CLIOutbox"]


class CLIOutbox(BaseOutbox):
    """An outbox that formats and prints the result to the command line."""

    def send(self, result: ReasoningResult) -> None:
        """
        Handle output to the user/environment.

        Formats and prints the reasoning result to stdout.

        Args:
            result: Reasoning result to present
        """
        if result.success:
            print(f"✅ **Answer:** {result.final_answer}")

            if result.tool_calls:
                print(f"\n📋 **Used {len(result.tool_calls)} tool(s) in {result.iterations} iteration(s):**")
                for i, call in enumerate(result.tool_calls, 1):
                    tool_name = call.get('tool_name', call.get('tool_id', 'Unknown'))
                    print(f"  {i}. {tool_name}")
        else:
            print(f"❌ **Failed:** {result.final_answer}")
            if result.error_message:
                print(f"   Error: {result.error_message}")
