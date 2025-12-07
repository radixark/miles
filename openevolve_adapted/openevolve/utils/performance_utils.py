"""
Performance monitoring utilities for OpenEvolve
"""

import functools
import logging
import time
from typing import Callable, Any

logger = logging.getLogger(__name__)


def timed_operation(operation_name: str = None, log_level: int = logging.INFO):
    """
    Decorator to measure and log execution time of functions

    Args:
        operation_name: Custom name for the operation (defaults to function name)
        log_level: Logging level for the time message

    Usage:
        @timed_operation("Database Save")
        def save(self, path):
            ...

    Output:
        [TIME] Database Save completed in 2.34s
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            op_name = operation_name or func.__name__
            start_time = time.time()

            result = func(*args, **kwargs)

            elapsed = time.time() - start_time
            log_msg = f"[TIME] {op_name} completed in {elapsed:.3f}s"

            logger.log(log_level, log_msg)
            print(log_msg)  # Also print for immediate visibility

            return result

        return wrapper
    return decorator


def timed_async_operation(operation_name: str = None, log_level: int = logging.INFO):
    """
    Async version of timed_operation decorator

    Usage:
        @timed_async_operation("Async Evaluation")
        async def evaluate(self, program):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            op_name = operation_name or func.__name__
            start_time = time.time()

            result = await func(*args, **kwargs)

            elapsed = time.time() - start_time
            log_msg = f"[TIME] {op_name} completed in {elapsed:.3f}s"

            logger.log(log_level, log_msg)
            print(log_msg)

            return result

        return wrapper
    return decorator
