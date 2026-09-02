import logging

import numpy as np
import torch

from miles.utils.audit_utils.event_logger.logger import get_event_logger, is_event_logger_initialized
from miles.utils.audit_utils.event_logger.models import MetricEvent

from .base import (
    MilesDashboardBackend,
    MlflowBackend,
    PrometheusBackend,
    TensorboardBackend,
    TrackingBackend,
    TrackingManager,
    WandbBackend,
)
from .ci_history import CiHistoryBackend

# The full registry lives here, not base.py: base must never import a backend
# module (ci_history imports TrackingBackend from base -> circular). This
# module is the tracking entry point and the one place that imports every
# backend, so it owns the registry.
BACKEND_REGISTRY: dict[str, tuple[type[TrackingBackend], str]] = {
    "wandb": (WandbBackend, "use_wandb"),
    "tensorboard": (TensorboardBackend, "use_tensorboard"),
    "mlflow": (MlflowBackend, "use_mlflow"),
    "prometheus": (PrometheusBackend, "use_prometheus"),
    "ci_history": (CiHistoryBackend, "ci_enable_metrics_capture"),
    "miles_dashboard": (MilesDashboardBackend, "use_miles_dashboard"),
}

logger = logging.getLogger(__name__)
_manager = TrackingManager(BACKEND_REGISTRY)


def init_tracking(args, primary: bool = True, **kwargs):
    _manager.init(args, primary=primary, **kwargs)


def define_step_key_metric_group(prefix: str, step_key: str) -> None:
    """Declare a metric group plotted against its own step key (e.g. ``{name}/*`` vs ``{name}/step``).
    Only wandb acts on this; must be called from the primary tracking process or definitions may be lost."""
    _manager.define_step_key_metric_group(prefix, step_key)


def jsonable_metrics(metrics: dict) -> dict:
    """Unwrap metric values that pydantic cannot serialise.

    Torch tensors were already unwrapped here. Numpy scalars were not, and
    ``np.int64`` is not a Python ``int``: ``MetricEvent.model_dump_json()`` raised
    ``PydanticSerializationError: Unable to serialize unknown type`` and took the
    training step down with it.

    The path is only reachable once the event logger is initialised (via
    ``--dump-details`` / ``--save-debug-event-data``), which is why it went
    unnoticed -- a metric hook that emits a numpy scalar, such as the multi-turn
    rollout metrics under ``--log-multi-turn``, crashes only for runs that also
    asked for event dumps. Both are unwrapped by ``.item()``; anything else is
    handed to pydantic unchanged.
    """
    return {k: (v.item() if isinstance(v, (torch.Tensor, np.generic)) else v) for k, v in metrics.items()}


def log(args, metrics, step_key: str):
    step = metrics.get(step_key)
    _manager.log(metrics, step=step, step_key=step_key)

    if is_event_logger_initialized():
        get_event_logger().log(MetricEvent, {"metrics": jsonable_metrics(metrics)}, print_log=False)


def finish_tracking():
    _manager.finish()
