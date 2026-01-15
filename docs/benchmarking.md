# Performance Benchmarking

This document describes how to use the performance benchmarking script to measure and analyze standard-agent performance.

## Overview

The `scripts/benchmark.py` script provides comprehensive performance testing for the standard-agent library, measuring:

- **Agent initialization time** - How long it takes to create and configure agents
- **Goal solving performance** - End-to-end time to process and solve goals
- **Memory usage monitoring** - Peak and current memory consumption during operations
- **Tool interaction overhead** - Performance impact of external tool usage

## Quick Start

Run all benchmarks with default settings:

```bash
python scripts/benchmark.py
```

This will benchmark both ReACT and ReWOO agents using real components (actual LLMs and tools). To use deterministic test doubles for consistent results without API calls, add the `--deterministic` flag.

## Command Line Options

### Basic Usage

```bash
# Run specific agent scenarios
python scripts/benchmark.py --scenarios react rewoo

# Use deterministic test doubles (recommended for CI/development)
python scripts/benchmark.py --deterministic

# Customize iteration count
python scripts/benchmark.py --iterations 10

# Save results to file
python scripts/benchmark.py --output benchmark_results.json

# Test with custom goals
python scripts/benchmark.py --goals "Calculate 2+2" "Search for Python docs"
```

### Complete Example

```bash
python scripts/benchmark.py \
  --scenarios react rewoo \
  --deterministic \
  --iterations 5 \
  --output results.json \
  --goals "Calculate fibonacci of 10" "What is machine learning?"
```

## Benchmark Scenarios

### 1. Agent Initialization

Measures the time required to:
- Create LLM, tools, memory, and reasoner components
- Wire components together into a StandardAgent
- Prepare the agent for goal processing

**What it tests**: Component initialization overhead, dependency injection performance

### 2. Goal Solving

Measures end-to-end goal processing including:
- Goal preprocessing (if applicable)
- Reasoning loop execution
- Tool selection and execution
- Final answer generation and summarization

**What it tests**: Complete agent workflow performance, reasoning efficiency

### 3. Memory Operations

Measures memory subsystem performance:
- Storing and retrieving data
- Conversation history management
- Memory cleanup and optimization

**What it tests**: Memory backend efficiency, data persistence overhead

## Deterministic vs Real Mode

### Deterministic Mode (Recommended for Development)

Enabled with the `--deterministic` flag. Uses real ReACT and ReWOO reasoners with mocked LLM and tools that return canned but realistic responses:

- **DeterministicLLM**: Mock language model with configurable response times and realistic response patterns
- **DeterministicTools**: Mock tools with simulated execution delays
- **DeterministicReasoner**: Mock reasoner that mimics Key-Value pairs of "Think", "Act", "Observe" without actual logic inference

**Benefits**:
- Consistent, reproducible results
- No external API calls or costs
- Fast execution
- Controlled test scenarios

**Use when**:
- Developing the agent framework
- CI/CD pipeline testing
- Regression testing
- Comparing performance changes

### Real Mode (Default)

This is the default mode when no `--deterministic` flag is specified. Uses actual components with real LLMs and tools:

- **Real LLM**: Actual API calls to language models
- **Real Tools**: Actual external tool executions
- **Real Reasoners**: Full reasoning loops with real complexity

**Benefits**:
- Real-world performance measurements
- Actual API latency and costs
- True end-to-end validation

**Use when**:
- Production performance analysis
- Capacity planning
- Real-world scenario testing
- Performance optimization validation

## Understanding Results

### Sample Output

```text
BENCHMARK RESULTS
================================================================================

REACT_AGENT - Initialization
------------------------------------------------------------
Total Runs:        5
Success Rate:      100.0% (5/5)
Avg Duration:      125.3ms
Median Duration:   123.1ms
Duration Range:    118.2ms - 134.7ms
Duration StdDev:   6.2ms
Avg Memory Peak:   15.42MB
Avg Memory Current:12.33MB
Avg Iterations:    0.0
Avg Tool Calls:    0.0

REACT_AGENT - Solve Goal
------------------------------------------------------------
Total Runs:        3
Success Rate:      100.0% (3/3)
Avg Duration:      2456.7ms
Median Duration:   2401.2ms
Duration Range:    2389.1ms - 2578.9ms
Duration StdDev:   102.4ms
Avg Memory Peak:   28.91MB
Avg Memory Current:18.77MB
Avg Iterations:    3.0
Avg Tool Calls:    2.0
```

### Key Metrics

- **Success Rate**: Percentage of successful benchmark runs
- **Avg Duration**: Mean execution time across all successful runs
- **Median Duration**: Middle value when execution times are sorted
- **Duration Range**: Fastest and slowest execution times
- **Duration StdDev**: Standard deviation indicating consistency
- **Memory Peak**: Highest memory usage during execution
- **Memory Current**: Memory usage at completion
- **Avg Iterations**: Average reasoning loop iterations
- **Avg Tool Calls**: Average number of external tool calls

### Performance Interpretation

#### Good Performance Indicators
- High success rate (>95%)
- Low standard deviation (consistent timing)
- Reasonable memory usage for workload
- Appropriate iteration/tool call counts

#### Performance Issues
- Low success rate indicates reliability problems
- High standard deviation suggests inconsistent performance
- Excessive memory usage may indicate memory leaks
- Too many iterations might indicate inefficient reasoning

## Using Results for Optimization

### Identifying Bottlenecks

1. **Initialization Performance**: If initialization is slow, check component creation and dependency injection
2. **Goal Solving Performance**: If solving is slow, analyze reasoning efficiency and tool selection
3. **Memory Usage**: If memory is high, check for leaks or inefficient data structures

### Comparative Analysis

Run benchmarks before and after changes to measure impact:

```bash
# Before changes
python scripts/benchmark.py --output before.json

# Make your changes...

# After changes  
python scripts/benchmark.py --output after.json
```

Compare the JSON outputs to quantify performance improvements or regressions.

### Continuous Integration

Add benchmarking to your CI pipeline:

```bash
# In your CI script
python scripts/benchmark.py --deterministic --iterations 3 --output ci_results.json

# Optionally, compare against baseline
python scripts/compare_benchmarks.py baseline.json ci_results.json
```

## Extending the Benchmark

### Adding New Scenarios

To benchmark additional agent types or configurations:

1. Add scenario to `benchmark_runner.py`
2. Implement agent creation logic
3. Add to CLI argument choices
4. Update documentation

### Adding New Operations

To benchmark additional operations:

1. Create new benchmark method in `BenchmarkRunner`
2. Add operation to main benchmark loop
3. Document expected behavior

### Custom Test Doubles

For specialized testing, create custom test doubles:

```python
class CustomLLM(BaseLLM):
    def __init__(self, delay_ms=100, failure_rate=0.1):
        self.delay_ms = delay_ms
        self.failure_rate = failure_rate
    
    def completion(self, messages, **kwargs):
        if random.random() < self.failure_rate:
            raise Exception("Simulated failure")
        time.sleep(self.delay_ms / 1000)
        return "Custom test response"
```

## Troubleshooting

### Common Issues

**ImportError**: Ensure you're running from the project root and all dependencies are installed:
```bash
pip install -e .
```

**Permission Denied**: Make the script executable:
```bash
chmod +x scripts/benchmark.py
```

**Memory Errors**: Reduce iteration count or use deterministic mode:
```bash
python scripts/benchmark.py --deterministic --iterations 1
```

**API Rate Limits**: Use deterministic mode for development:
```bash
python scripts/benchmark.py --deterministic
```

### Environment Setup

For real mode benchmarking, ensure environment variables are set:
```bash
export LLM_MODEL="gpt-3.5-turbo"
export OPENAI_API_KEY="your-api-key"
```

For deterministic mode, no external dependencies are required.

## Best Practices

1. **Use deterministic mode for development** to avoid API costs and ensure reproducibility
2. **Run multiple iterations** to get statistically meaningful results
3. **Save results to files** for later analysis and comparison
4. **Monitor both timing and memory** to catch different types of performance issues
5. **Test with realistic goals** that represent your actual use cases
6. **Run benchmarks in CI** to catch performance regressions early
7. **Compare results over time** to track performance trends