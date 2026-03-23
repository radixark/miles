import logging

import wandb
from miles.utils.tensorboard_utils import _TensorboardAdapter

from . import wandb_utils
from .prometheus_utils import PrometheusAdapter

logger = logging.getLogger(__name__)


def init_tracking(args, primary: bool = True, **kwargs):
    if primary:
        wandb_utils.init_wandb_primary(args, **kwargs)
    else:
        wandb_utils.init_wandb_secondary(args, **kwargs)

    if getattr(args, "use_prometheus", False):
        PrometheusAdapter(args, start_server=primary)


# TODO further refactor, e.g. put TensorBoard init to the "init" part
def log(args, metrics, step_key: str):
    if args.use_wandb:
        wandb.log(metrics)

    if args.use_tensorboard:
        metrics_except_step = {k: v for k, v in metrics.items() if k != step_key}
        _TensorboardAdapter(args).log(data=metrics_except_step, step=metrics[step_key])

    if getattr(args, "use_prometheus", False):
        PrometheusAdapter(args).update(metrics)
