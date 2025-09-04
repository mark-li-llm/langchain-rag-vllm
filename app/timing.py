"""Timing utilities for performance measurement.

This module provides context managers and utilities for measuring
execution time across different components.
"""

import time
import functools
from typing import Dict, Any, Optional, Callable
from contextlib import contextmanager


class Timer:
    """Simple timer for measuring execution time."""
    
    def __init__(self):
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.elapsed_ms: Optional[float] = None
    
    def start(self) -> None:
        """Start the timer."""
        self.start_time = time.time()
        self.end_time = None
        self.elapsed_ms = None
    
    def stop(self) -> float:
        """Stop the timer and return elapsed time in milliseconds.
        
        Returns:
            Elapsed time in milliseconds
        """
        if self.start_time is None:
            raise RuntimeError("Timer not started")
        
        self.end_time = time.time()
        self.elapsed_ms = (self.end_time - self.start_time) * 1000
        return self.elapsed_ms
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


@contextmanager
def time_operation(operation_name: str = "operation"):
    """Context manager for timing operations.
    
    Args:
        operation_name: Name of the operation being timed
        
    Yields:
        Dictionary that will contain timing results
    """
    timing_info = {"operation": operation_name}
    
    start_time = time.time()
    try:
        yield timing_info
    finally:
        end_time = time.time()
        timing_info.update({
            "start_time": start_time,
            "end_time": end_time,
            "duration_ms": (end_time - start_time) * 1000
        })


class TimingCollector:
    """Collects timing information for multiple operations."""
    
    def __init__(self):
        self.timings: Dict[str, float] = {}
        self.active_timers: Dict[str, float] = {}
    
    def start(self, operation_name: str) -> None:
        """Start timing an operation.
        
        Args:
            operation_name: Name of the operation
        """
        self.active_timers[operation_name] = time.time()
    
    def stop(self, operation_name: str) -> float:
        """Stop timing an operation.
        
        Args:
            operation_name: Name of the operation
            
        Returns:
            Elapsed time in milliseconds
            
        Raises:
            KeyError: If operation was not started
        """
        if operation_name not in self.active_timers:
            raise KeyError(f"Operation '{operation_name}' was not started")
        
        start_time = self.active_timers.pop(operation_name)
        elapsed_ms = (time.time() - start_time) * 1000
        self.timings[operation_name] = elapsed_ms
        return elapsed_ms
    
    @contextmanager
    def time(self, operation_name: str):
        """Context manager for timing an operation.
        
        Args:
            operation_name: Name of the operation
        """
        self.start(operation_name)
        try:
            yield
        finally:
            self.stop(operation_name)
    
    def get_timings(self) -> Dict[str, float]:
        """Get all recorded timings.
        
        Returns:
            Dictionary of operation names to elapsed times in milliseconds
        """
        return self.timings.copy()
    
    def get_total_time(self) -> float:
        """Get total time across all operations.
        
        Returns:
            Total time in milliseconds
        """
        return sum(self.timings.values())
    
    def reset(self) -> None:
        """Reset all timings."""
        self.timings.clear()
        self.active_timers.clear()


def timed_function(operation_name: Optional[str] = None):
    """Decorator for timing function execution.
    
    Args:
        operation_name: Name for the operation (defaults to function name)
        
    Returns:
        Decorated function that includes timing information
    """
    def decorator(func: Callable) -> Callable:
        name = operation_name or func.__name__
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                success = True
                error = None
            except Exception as e:
                result = None
                success = False
                error = str(e)
                raise
            finally:
                end_time = time.time()
                duration_ms = (end_time - start_time) * 1000
                
                # Add timing info to result if it's a dict
                if success and isinstance(result, dict):
                    if "_timing" not in result:
                        result["_timing"] = {}
                    result["_timing"][name] = duration_ms
                
                # Log timing (could be replaced with proper logging)
                status = "SUCCESS" if success else "ERROR"
                print(f"[TIMING] {name}: {duration_ms:.2f}ms ({status})")
                
            return result
        
        return wrapper
    return decorator


def format_timing_summary(timings: Dict[str, float]) -> str:
    """Format timing information as a readable summary.
    
    Args:
        timings: Dictionary of operation names to times in milliseconds
        
    Returns:
        Formatted timing summary string
    """
    if not timings:
        return "No timing information available"
    
    lines = ["Timing Summary:"]
    total_time = sum(timings.values())
    
    # Sort by time descending
    sorted_timings = sorted(timings.items(), key=lambda x: x[1], reverse=True)
    
    for operation, time_ms in sorted_timings:
        percentage = (time_ms / total_time * 100) if total_time > 0 else 0
        lines.append(f"  {operation}: {time_ms:.2f}ms ({percentage:.1f}%)")
    
    lines.append(f"Total: {total_time:.2f}ms")
    
    return "\n".join(lines)


def get_performance_metrics(timings: Dict[str, float]) -> Dict[str, Any]:
    """Calculate performance metrics from timing data.
    
    Args:
        timings: Dictionary of operation names to times in milliseconds
        
    Returns:
        Dictionary with performance metrics
    """
    if not timings:
        return {"error": "No timing data available"}
    
    times = list(timings.values())
    total_time = sum(times)
    
    return {
        "total_time_ms": total_time,
        "operation_count": len(timings),
        "average_time_ms": total_time / len(times),
        "min_time_ms": min(times),
        "max_time_ms": max(times),
        "slowest_operation": max(timings.items(), key=lambda x: x[1])[0],
        "fastest_operation": min(timings.items(), key=lambda x: x[1])[0],
        "detailed_timings": timings
    }