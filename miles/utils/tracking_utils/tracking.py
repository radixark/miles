import logging

import torch

from miles.utils.audit_utils.event_logger.logger import get_event_logger, is_event_logger_initialized
from miles.utils.audit_utils.event_logger.models import MetricEvent

from .base import MlflowBackend, PrometheusBackend, TensorboardBackend, TrackingBackend, TrackingManager, WandbBackend
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
}

logger = logging.getLogger(__name__)
_manager = TrackingManager(BACKEND_REGISTRY)


def init_tracking(args, primary: bool = True, **kwargs):
    _manager.init(args, primary=primary, **kwargs)


def define_step_key_metric_group(prefix: str, step_key: str) -> None:
    """Declare a runtime metric group plotted against its own step key
    (e.g. a multi-LoRA adapter's ``{name}/*`` against ``{name}/step``).

    Fans out to the active backends; only wandb does anything (chart axes are
    configuration there), and it deduplicates internally. Must be called from
    the primary tracking process (the driver): definitions asserted by
    secondary shared-mode writers are lost nondeterministically.
    """
    _manager.define_step_key_metric_group(prefix, step_key)


def log(args, metrics, step_key: str):
    step = metrics.get(step_key)
    _manager.log(metrics, step=step, step_key=step_key)

    if is_event_logger_initialized():
        serializable_metrics = {k: (v.item() if isinstance(v, torch.Tensor) else v) for k, v in metrics.items()}
        get_event_logger().log(MetricEvent, {"metrics": serializable_metrics}, print_log=False)


def finish_tracking():
    _manager.finish()
