import logging
from contextlib import contextmanager
from datetime import datetime
from functools import wraps
from time import time

import torch.distributed

from .misc import SingletonMeta

__all__ = ["Timer", "timer", "log_experiment_start"]

logger = logging.getLogger(__name__)

LOGFILE = "miles_timer"


class Timer(metaclass=SingletonMeta):
    def __init__(self):
        self.timers = {}
        self.start_time = {}

    def start(self, name, log_info=True):
        assert name not in self.start_time, f"Timer {name} already started."
        self.start_time[name] = time()
        if log_info and torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            logger.info(f"Timer {name} start")

    def end(self, name, log_info=True):
        assert name in self.start_time, f"Timer {name} not started."
        elapsed_time = time() - self.start_time[name]
        self.add(name, elapsed_time)
        del self.start_time[name]
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        if rank == 0:
            if log_info:
                logger.info(f"Timer {name} end (elapsed: {elapsed_time:.1f}s)")
            with open(f"{LOGFILE}_{rank}.log", "a") as f:
                f.write(f"Timer {name} end (elapsed: {elapsed_time*1000:.3f}ms)\n")

    def reset(self, name=None):
        if name is None:
            self.timers = {}
        elif name in self.timers:
            del self.timers[name]

    def add(self, name, elapsed_time):
        self.timers[name] = self.timers.get(name, 0) + elapsed_time

    def log_dict(self):
        return self.timers

    def log_experiment_start(self, config_dict: dict):
        """Log experiment start marker with configuration and timestamp."""
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        if rank == 0:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            separator_line = "=" * 80
            config_lines = []
            for key, value in sorted(config_dict.items()):
                config_lines.append(f"  {key}: {value}")

            log_content = (
                f"\n{separator_line}\n"
                f"EXPERIMENT START: {timestamp}\n"
                f"Configuration:\n" + "\n".join(config_lines) + "\n"
                f"{separator_line}\n\n"
            )

            with open(f"{LOGFILE}_{rank}.log", "a") as f:
                f.write(log_content)

    @contextmanager
    def context(self, name, log_info=True):
        self.start(name, log_info=log_info)
        try:
            yield
        finally:
            self.end(name, log_info=log_info)


def timer(name_or_func, log_info=True):
    """
    Can be used either as a decorator or a context manager:

    @timer
    def func():
        ...

    or

    with timer("block_name"):
        ...

    or (to suppress logging):

    with timer("block_name", log_info=False):
        ...
    """
    # When used as a context manager
    if isinstance(name_or_func, str):
        name = name_or_func
        return Timer().context(name, log_info=log_info)

    func = name_or_func

    @wraps(func)
    def wrapper(*args, **kwargs):
        with Timer().context(func.__name__, log_info=log_info):
            return func(*args, **kwargs)

    return wrapper


def log_experiment_start(config_dict: dict):
    """
    Log experiment start marker with configuration and timestamp.

    Args:
        config_dict: Dictionary containing experiment configuration parameters

    Example:
        log_experiment_start({
            "mode": "rdma",
            "num_train_gpus": 1,
            "num_rollout_gpus": 1,
            "pipelined_transfer": True
        })
    """
    Timer().log_experiment_start(config_dict)


@contextmanager
def inverse_timer(name):
    Timer().end(name)
    try:
        yield
    finally:
        Timer().start(name)


def with_defer(deferred_func):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            finally:
                deferred_func()

        return wrapper

    return decorator
