import logging

import torch

from miles.utils.audit_utils.event_logger.logger import get_event_logger, is_event_logger_initialized
from miles.utils.audit_utils.event_logger.models import MetricEvent

from .base import TrackingManager

logger = logging.getLogger(__name__)
_manager = TrackingManager()


def init_tracking(args, primary: bool = True, **kwargs):
    _manager.init(args, primary=primary, **kwargs)


def log(args, metrics, step_key: str):
    step = metrics.get(step_key)
    _manager.log(metrics, step=step, step_key=step_key)

    if is_event_logger_initialized():
        serializable_metrics = {k: (v.item() if isinstance(v, torch.Tensor) else v) for k, v in metrics.items()}
        get_event_logger().log(MetricEvent, {"metrics": serializable_metrics}, print_log=False)


def finish_tracking():
    _manager.finish()
