"""
RAII-style timer for performance measurement and logging.
"""
import time
from typing import Optional, Callable, Any, Dict
from contextlib import contextmanager
from .logger import get_logger, get_config


class Timer:
    """
    RAII-style timer that can be used as a context manager.
    
    Automatically logs timing information when the context exits,
    if performance logging is enabled in the config.
    """
    
    def __init__(self, name: str, logger_name: Optional[str] = None):
        """
        Initialize the timer.
        
        Args:
            name: Name/description of the operation being timed.
            logger_name: Name of logger to use (defaults to a timer-specific logger).
        """
        self.name = name
        self.logger = get_logger(logger_name or __name__)
        self.config = get_config().get('performance', {})
        
        self.start_time: Optional[float] = None
        self.duration_ms: Optional[float] = None
        
    def __enter__(self) -> 'Timer':
        """Start the timer when entering the context."""
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Stop the timer and log the duration when exiting the context."""
        if self.start_time is None:
            return
            
        self.duration_ms = (time.perf_counter() - self.start_time) * 1000
        
        if self.config.get('enabled', True):
            slow_threshold = self.config.get('slow_threshold_ms', 1000)
            
            if self.duration_ms >= slow_threshold:
                self.logger.warning(
                    f"SLOW OPERATION: {self.name} took {self.duration_ms:.2f}ms "
                    f"(threshold: {slow_threshold}ms)"
                )
            else:
                self.logger.info(f"{self.name} completed in {self.duration_ms:.2f}ms")
    
    def get_duration_ms(self) -> Optional[float]:
        """Get the duration in milliseconds after the timer has stopped."""
        return self.duration_ms


class TimerStats:
    """
    Utility class to collect and analyze timing statistics.
    """
    
    def __init__(self):
        self.timings: Dict[str, list[float]] = {}
        self.logger = get_logger(__name__)
    
    def record(self, name: str, duration_ms: float) -> None:
        """Record a timing measurement."""
        if name not in self.timings:
            self.timings[name] = []
        self.timings[name].append(duration_ms)
    
    def get_stats(self, name: str) -> Optional[Dict[str, float]]:
        """
        Get statistics for a named operation.
        
        Returns:
            Dictionary with min, max, avg, count, total
        """
        if name not in self.timings or not self.timings[name]:
            return None
        
        durations = self.timings[name]
        return {
            'count': len(durations),
            'total_ms': sum(durations),
            'min_ms': min(durations),
            'max_ms': max(durations),
            'avg_ms': sum(durations) / len(durations),
            'median_ms': sorted(durations)[len(durations) // 2]
        }
    
    def log_stats(self, name: str) -> None:
        """Log statistics for a named operation."""
        stats = self.get_stats(name)
        if stats:
            self.logger.info(f"Stats for '{name}': {stats}")
        else:
            self.logger.warning(f"No timing data available for '{name}'")
    
    def log_all_stats(self) -> None:
        """Log statistics for all recorded operations."""
        if not self.timings:
            self.logger.info("No timing data recorded")
            return
        
        self.logger.info("=== Timing Statistics ===")
        for name in sorted(self.timings.keys()):
            self.log_stats(name)
    
    def clear(self, name: Optional[str] = None) -> None:
        """Clear timing data for a specific operation or all operations."""
        if name:
            self.timings.pop(name, None)
        else:
            self.timings.clear()


# Global timer stats instance
_timer_stats = TimerStats()


def get_timer_stats() -> TimerStats:
    """Get the global timer stats instance."""
    return _timer_stats


@contextmanager
def time_operation(
    name: str, 
    logger_name: Optional[str] = None,
    auto_log: bool = True,
    collect_stats: bool = True
):
    """
    Context manager for timing operations.
    
    Args:
        name: Name of the operation
        logger_name: Logger name to use
        auto_log: Whether to auto-log the duration
        collect_stats: Whether to collect stats in global stats collector
    
    Example:
        with time_operation("database_query"):
            # Your code here
            pass
    """
    callback = None
    if collect_stats:
        callback = lambda n, d: _timer_stats.record(n, d)
    
    timer = Timer(name, logger_name)
    
    with timer:
        yield timer


def benchmark(func: Callable, *args, iterations: int = 1, **kwargs) -> Dict[str, Any]:
    """
    Benchmark a function by running it multiple times.
    
    Args:
        func: Function to benchmark
        *args: Arguments to pass to function
        iterations: Number of times to run the function
        **kwargs: Keyword arguments to pass to function
    
    Returns:
        Dictionary with timing statistics and results
    """
    logger = get_logger(__name__)
    func_name = getattr(func, '__name__', str(func))
    
    results = []
    durations = []
    
    logger.info(f"Benchmarking {func_name} with {iterations} iterations")
    
    for i in range(iterations):
        with Timer(f"{func_name}_iter_{i}", auto_log=False) as timer:
            result = func(*args, **kwargs)
            results.append(result)
        
        durations.append(timer.get_duration_ms())
    
    stats = {
        'function_name': func_name,
        'iterations': iterations,
        'total_ms': sum(durations),
        'min_ms': min(durations),
        'max_ms': max(durations),
        'avg_ms': sum(durations) / len(durations),
        'median_ms': sorted(durations)[len(durations) // 2],
        'results': results
    }
    
    logger.info(f"Benchmark results for {func_name}: avg={stats['avg_ms']:.2f}ms, "
                f"min={stats['min_ms']:.2f}ms, max={stats['max_ms']:.2f}ms")
    
    return stats 