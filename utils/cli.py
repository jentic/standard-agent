"""CLI utility functions for user interaction."""
import sys


def read_user_goal(prompt: str = "🤖 Enter your goal: ") -> str:
    """Read a goal from user input via stdin."""
    print(prompt, end="", flush=True)
    try:
        line = sys.stdin.readline()
        if not line:  # EOF
            raise KeyboardInterrupt
        
        goal = line.strip()
        if goal.lower() in {"bye", "quit", "exit", "q"}:
            raise KeyboardInterrupt
            
        return goal
    except (EOFError, KeyboardInterrupt):
        raise


def print_result(result) -> None:
    """Print the reasoning result to stdout."""
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