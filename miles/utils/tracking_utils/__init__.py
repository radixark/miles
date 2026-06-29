import logging

from .base import TrackingManager
from .ci_history import RECORD_DIR_ENV, TARGET_METRIC_KEYS, CiHistoryBackend

logger = logging.getLogger(__name__)
_manager = TrackingManager()

__all__ = [
    "CiHistoryBackend",
    "RECORD_DIR_ENV",
    "TARGET_METRIC_KEYS",
    "finish_tracking",
    "init_tracking",
    "log",
]


def init_tracking(args, primary: bool = True, **kwargs):
    _manager.init(args, primary=primary, **kwargs)


def log(args, metrics, step_key: str):
    step = metrics.get(step_key)
    _manager.log(metrics, step=step, step_key=step_key)


def finish_tracking():
    _manager.finish()
