#!/usr/bin/env python3
"""
Benchmarking script for standard-agent performance testing.
Supports both deterministic testing with mocks and real agent testing.
"""
import argparse
import json
import sys
import time
import tracemalloc
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
from statistics import mean, median, stdev

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import project modules after adding to path
from agents.standard_agent import StandardAgent
from agents.prebuilt import ReACTAgent, ReWOOAgent
from agents.reasoner.base import BaseReasoner, ReasoningResult
from agents.llm.base_llm import BaseLLM
from agents.tools.base import JustInTimeToolingBase, ToolBase
from agents.memory.dict_memory import DictMemory
from utils.logger import get_logger

# Constants
DEFAULT_MODEL = "gpt-3.5-turbo"


@dataclass
class BenchmarkResult:
    """Represents the result of a single benchmark run."""
    scenario_name: str
    operation: str
    success: bool
    duration_ms: float
    memory_peak_mb: float
    memory_current_mb: float
    iterations: int = 0
    tool_calls: int = 0
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkSummary:
    """Summary statistics for a series of benchmark runs."""
    scenario_name: str
    operation: str
    total_runs: int
    successful_runs: int
    failed_runs: int
    avg_duration_ms: float
    median_duration_ms: float
    min_duration_ms: float
    max_duration_ms: float
    duration_stddev_ms: float
    avg_memory_peak_mb: float
    avg_memory_current_mb: float
    success_rate: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class DeterministicLLM(BaseLLM):
    """Mock LLM for deterministic benchmarking."""

    def __init__(self, response_time_ms: float = 100):
        super().__init__(model="deterministic-mock", temperature=0.0)
        self.response_time_ms = response_time_ms
        self.call_count = 0

    def completion(self, messages: List[Dict[str, str]], **kwargs) -> str:
        time.sleep(self.response_time_ms / 1000)  # Simulate network latency
        self.call_count += 1
        return f"Mock response #{self.call_count}"

    def prompt(self, text: str) -> str:
        time.sleep(self.response_time_ms / 1000)
        self.call_count += 1
        return f"Mock response #{self.call_count} for: {text[:50]}..."

    def prompt_to_json(self, text: str, max_retries: int = 0) -> Dict[str, Any]:
        time.sleep(self.response_time_ms / 1000)
        self.call_count += 1
        return {"mock": True, "response_id": self.call_count}


class DeterministicTool(ToolBase):
    """Mock tool for deterministic benchmarking."""

    def __init__(self, tool_id: str, execution_time_ms: float = 50):
        super().__init__(tool_id)
        self.execution_time_ms = execution_time_ms
        self.call_count = 0

    def get_summary(self) -> str:
        return f"Mock tool {self.id} for testing"

    def get_details(self) -> str:
        return f"Detailed mock tool {self.id} - simulates {self.execution_time_ms}ms execution time"

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "input": {"type": "string", "description": "Test input"}
            },
            "required": ["input"]
        }

    def execute(self, **kwargs) -> str:
        time.sleep(self.execution_time_ms / 1000)
        self.call_count += 1
        return f"Mock result #{self.call_count} with input: {kwargs.get('input', 'none')}"


class DeterministicTools(JustInTimeToolingBase):
    """Mock tools interface for deterministic benchmarking."""

    def __init__(self):
        self.tools = [
            DeterministicTool("mock_calculator", 25),
            DeterministicTool("mock_search", 75),
            DeterministicTool("mock_file_reader", 50),
        ]

    def search(self, query: str, *, top_k: int = 10) -> List[ToolBase]:
        """Search for tools matching a query - returns all tools for testing."""
        return self.tools[:min(top_k, len(self.tools))]

    def load(self, tool: ToolBase) -> ToolBase:
        """Load full specification - returns the tool as-is for testing."""
        return tool

    def execute(self, tool: ToolBase, parameters: Dict[str, Any]) -> Any:
        """Execute a tool with parameters."""
        if hasattr(tool, 'execute'):
            return tool.execute(**parameters)
        return f"Executed {tool.id} with {parameters}"

    def get_tools(self, goal: str, top_k: int = 10) -> List[ToolBase]:
        """Compatibility method for existing code."""
        return self.search(goal, top_k=top_k)


class DeterministicReasoner(BaseReasoner):
    """Mock reasoner for deterministic benchmarking."""

    def __init__(self, llm: BaseLLM, tools: JustInTimeToolingBase, memory,
                 iterations: int = 3, tool_calls: int = 2):
        super().__init__(llm=llm, tools=tools, memory=memory)
        self.target_iterations = iterations
        self.target_tool_calls = tool_calls

    def run(self, goal: str) -> ReasoningResult:
        # Simulate reasoning process
        transcript_parts = [f"Goal: {goal}"]
        tool_calls = []

        for i in range(self.target_iterations):
            # Simulate thinking
            thought = self.llm.prompt(f"Think about step {i+1}")
            transcript_parts.append(f"Think {i+1}: {thought}")

            # Simulate tool usage
            if i < self.target_tool_calls:
                available_tools = self.tools.get_tools(goal, top_k=5)
                if available_tools:
                    tool = available_tools[i % len(available_tools)]
                    result = tool.execute(input=f"step_{i+1}")
                    transcript_parts.append(f"Action {i+1}: Used {tool.id}, got: {result}")
                    tool_calls.append({
                        "tool_id": tool.id,
                        "input": f"step_{i+1}",
                        "result": result
                    })

        # Final answer
        final_answer = self.llm.prompt("Provide final answer")
        transcript_parts.append(f"Final: {final_answer}")

        return ReasoningResult(
            final_answer=final_answer,
            iterations=self.target_iterations,
            tool_calls=tool_calls,
            success=True,
            transcript="\n".join(transcript_parts)
        )


class BenchmarkRunner:
    """Main benchmarking orchestrator."""

    def __init__(self, deterministic: bool = False):
        self.deterministic = deterministic
        self.logger = get_logger(__name__)
        self.results: List[BenchmarkResult] = []

    def _measure_memory_usage(self) -> tuple[float, float]:
        """Get current and peak memory usage in MB."""
        if not tracemalloc.is_tracing():
            return 0.0, 0.0

        current, peak = tracemalloc.get_traced_memory()
        return current / 1024 / 1024, peak / 1024 / 1024

    def _create_deterministic_agent(self, agent_type: str) -> StandardAgent:
        """Create an agent with deterministic components for consistent benchmarking."""
        llm = DeterministicLLM(response_time_ms=100)
        tools = DeterministicTools()
        memory = DictMemory()

        if agent_type == "react":
            reasoner = DeterministicReasoner(llm, tools, memory, iterations=3, tool_calls=2)
        elif agent_type == "rewoo":
            reasoner = DeterministicReasoner(llm, tools, memory, iterations=4, tool_calls=3)
        else:
            reasoner = DeterministicReasoner(llm, tools, memory, iterations=2, tool_calls=1)

        return StandardAgent(
            llm=llm,
            tools=tools,
            memory=memory,
            reasoner=reasoner
        )

    def _create_real_agent(self, agent_type: str) -> StandardAgent:
        """Create a real agent with actual components."""
        if agent_type == "react":
            return ReACTAgent(model=DEFAULT_MODEL, max_turns=5)  # Use faster model for benchmarking
        elif agent_type == "rewoo":
            return ReWOOAgent(model=DEFAULT_MODEL, max_retries=1)
        else:
            # Fallback to ReACT
            return ReACTAgent(model=DEFAULT_MODEL, max_turns=3)

    def run_benchmark(self, scenario_name: str, operation: str,
                     benchmark_fn: Callable[[], Any], iterations: int = 5) -> List[BenchmarkResult]:
        """Run a benchmark function multiple times and collect results."""
        results = []

        self.logger.info("Running benchmark: %s/%s (%d iterations)", scenario_name, operation, iterations)

        for _ in range(iterations):
            tracemalloc.start()
            start_time = time.perf_counter()

            try:
                result = benchmark_fn()
                success = True
                error_message = None

                # Extract metadata from result if it's a ReasoningResult
                metadata = {}
                iterations_count = 0
                tool_calls_count = 0

                if isinstance(result, ReasoningResult):
                    metadata["final_answer_length"] = len(result.final_answer) if result.final_answer else 0
                    metadata["transcript_length"] = len(result.transcript) if result.transcript else 0
                    iterations_count = result.iterations
                    tool_calls_count = len(result.tool_calls) if result.tool_calls else 0

            except Exception as e:
                success = False
                error_message = str(e)
                metadata = {"error_type": type(e).__name__}
                iterations_count = 0
                tool_calls_count = 0

            end_time = time.perf_counter()
            duration_ms = (end_time - start_time) * 1000

            memory_current, memory_peak = self._measure_memory_usage()
            tracemalloc.stop()

            benchmark_result = BenchmarkResult(
                scenario_name=scenario_name,
                operation=operation,
                success=success,
                duration_ms=duration_ms,
                memory_peak_mb=memory_peak,
                memory_current_mb=memory_current,
                iterations=iterations_count,
                tool_calls=tool_calls_count,
                error_message=error_message,
                metadata=metadata
            )

            results.append(benchmark_result)
            self.results.append(benchmark_result)

            # Brief pause between iterations
            time.sleep(0.1)

        return results

    def benchmark_agent_initialization(self, agent_type: str, iterations: int = 5) -> List[BenchmarkResult]:
        """Benchmark agent initialization time."""
        def create_agent():
            if self.deterministic:
                return self._create_deterministic_agent(agent_type)
            else:
                return self._create_real_agent(agent_type)

        return self.run_benchmark(
            scenario_name=f"{agent_type}_agent",
            operation="initialization",
            benchmark_fn=create_agent,
            iterations=iterations
        )

    def benchmark_goal_solving(self, agent_type: str, goal: str, iterations: int = 3) -> List[BenchmarkResult]:
        """Benchmark goal solving performance."""
        def solve_goal():
            agent = self._create_deterministic_agent(agent_type) if self.deterministic else self._create_real_agent(agent_type)
            return agent.solve(goal)

        return self.run_benchmark(
            scenario_name=f"{agent_type}_agent",
            operation="solve_goal",
            benchmark_fn=solve_goal,
            iterations=iterations
        )

    def benchmark_memory_operations(self, agent_type: str, iterations: int = 5) -> List[BenchmarkResult]:
        """Benchmark memory operations."""
        def memory_ops():
            agent = self._create_deterministic_agent(agent_type) if self.deterministic else self._create_real_agent(agent_type)

            # Test memory operations
            for i in range(10):
                agent.memory[f"key_{i}"] = f"value_{i}" * 100  # Store some data

            # Simulate conversation history
            # Manually append to conversation_history to avoid using private _record_interaction
            if "conversation_history" not in agent.memory:
                agent.memory["conversation_history"] = []
                
            history = agent.memory["conversation_history"]
            for i in range(5):
                 history.append({"goal": f"test_goal_{i}", "result": f"test_result_{i}" * 50})
            
            # Simulate window trimming (mocking the behavior of _record_interaction)
            if len(history) > 5:
                agent.memory["conversation_history"] = history[-5:]

            return len(agent.memory)

        return self.run_benchmark(
            scenario_name=f"{agent_type}_agent",
            operation="memory_operations",
            benchmark_fn=memory_ops,
            iterations=iterations
        )

    def _group_results_by_scenario(self, results: List[BenchmarkResult]) -> Dict[str, List[BenchmarkResult]]:
        """Group benchmark results by scenario and operation."""
        grouped = defaultdict(list)
        for result in results:
            key = f"{result.scenario_name}_{result.operation}"
            grouped[key].append(result)
        return grouped

    def _calculate_duration_stats(self, durations: List[float]) -> Dict[str, float]:
        """Calculate duration statistics from a list of durations."""
        return {
            "avg": mean(durations),
            "median": median(durations),
            "min": min(durations),
            "max": max(durations),
            "stddev": stdev(durations) if len(durations) > 1 else 0.0,
        }

    def _calculate_metadata(self, successful_results: List[BenchmarkResult]) -> Dict[str, float]:
        """Calculate metadata from successful results."""
        if not successful_results:
            return {"avg_iterations": 0, "avg_tool_calls": 0}

        return {
            "avg_iterations": mean([r.iterations for r in successful_results]),
            "avg_tool_calls": mean([r.tool_calls for r in successful_results]),
        }

    def _create_summary_from_results(self, group_results: List[BenchmarkResult]) -> Optional[BenchmarkSummary]:
        """Create a BenchmarkSummary from a group of results."""
        successful = [r for r in group_results if r.success]
        if not successful:
            return None

        failed = [r for r in group_results if not r.success]
        durations = [r.duration_ms for r in successful]
        memory_peaks = [r.memory_peak_mb for r in successful]
        memory_currents = [r.memory_current_mb for r in successful]

        duration_stats = self._calculate_duration_stats(durations)
        metadata = self._calculate_metadata(successful)

        return BenchmarkSummary(
            scenario_name=group_results[0].scenario_name,
            operation=group_results[0].operation,
            total_runs=len(group_results),
            successful_runs=len(successful),
            failed_runs=len(failed),
            avg_duration_ms=duration_stats["avg"],
            median_duration_ms=duration_stats["median"],
            min_duration_ms=duration_stats["min"],
            max_duration_ms=duration_stats["max"],
            duration_stddev_ms=duration_stats["stddev"],
            avg_memory_peak_mb=mean(memory_peaks),
            avg_memory_current_mb=mean(memory_currents),
            success_rate=len(successful) / len(group_results),
            metadata=metadata
        )

    def generate_summary(self, results: List[BenchmarkResult]) -> Dict[str, BenchmarkSummary]:
        """Generate summary statistics from benchmark results."""
        grouped = self._group_results_by_scenario(results)
        summaries = {}

        for key, group_results in grouped.items():
            summary = self._create_summary_from_results(group_results)
            if summary:
                summaries[key] = summary

        return summaries

    def print_results(self, summaries: Dict[str, BenchmarkSummary]):
        """Print benchmark results in a readable format."""
        print("\n" + "=" * 80)
        print("BENCHMARK RESULTS")
        print("=" * 80)

        for summary in summaries.values():
            print(f"\n{summary.scenario_name.upper()} - {summary.operation.replace('_', ' ').title()}")
            print("-" * 60)
            print(f"Total Runs:        {summary.total_runs}")
            print(f"Success Rate:      {summary.success_rate:.1%} ({summary.successful_runs}/{summary.total_runs})")
            print(f"Avg Duration:      {summary.avg_duration_ms:.1f}ms")
            print(f"Median Duration:   {summary.median_duration_ms:.1f}ms")
            print(f"Duration Range:    {summary.min_duration_ms:.1f}ms - {summary.max_duration_ms:.1f}ms")
            print(f"Duration StdDev:   {summary.duration_stddev_ms:.1f}ms")
            print(f"Avg Memory Peak:   {summary.avg_memory_peak_mb:.2f}MB")
            print(f"Avg Memory Current:{summary.avg_memory_current_mb:.2f}MB")

            if summary.metadata:
                print(f"Avg Iterations:    {summary.metadata.get('avg_iterations', 0):.1f}")
                print(f"Avg Tool Calls:    {summary.metadata.get('avg_tool_calls', 0):.1f}")

    def save_results(self, filename: str, summaries: Dict[str, BenchmarkSummary]):
        """Save benchmark results to a JSON file."""
        output_data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "deterministic": self.deterministic,
            "summaries": {key: asdict(summary) for key, summary in summaries.items()},
            "raw_results": [asdict(result) for result in self.results]
        }

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2)

        print(f"\nResults saved to: {filename}")


def main():
    """Main entry point for the benchmark script."""
    parser = argparse.ArgumentParser(description="Benchmark standard-agent performance")
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=["react", "rewoo"],
        default=["react", "rewoo"],
        help="Agent scenarios to benchmark"
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use test doubles for deterministic results"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Number of iterations per benchmark"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file for results (JSON format)"
    )
    parser.add_argument(
        "--goals",
        nargs="+",
        default=["Calculate 15 * 23", "What is the weather like?", "Find information about Python"],
        help="Goals to test for goal-solving benchmarks"
    )

    args = parser.parse_args()

    print("Standard Agent Performance Benchmark")
    print("=" * 40)
    print(f"Scenarios: {', '.join(args.scenarios)}")
    print(f"Deterministic: {args.deterministic}")
    print(f"Iterations: {args.iterations}")
    print()

    runner = BenchmarkRunner(deterministic=args.deterministic)

    # Run benchmarks for each scenario
    for scenario in args.scenarios:
        print(f"Benchmarking {scenario.upper()} agent...")

        # Benchmark initialization
        runner.benchmark_agent_initialization(scenario, args.iterations)

        # Benchmark goal solving with different goals
        for goal in args.goals[:2]:  # Limit to 2 goals to keep runtime reasonable
            runner.benchmark_goal_solving(scenario, goal, max(1, args.iterations // 2))

        # Benchmark memory operations
        runner.benchmark_memory_operations(scenario, args.iterations)

    # Generate and display results
    summaries = runner.generate_summary(runner.results)
    runner.print_results(summaries)

    # Save results if requested
    if args.output:
        runner.save_results(args.output, summaries)

    print(f"\nBenchmark completed! Total runs: {len(runner.results)}")


if __name__ == "__main__":
    main()