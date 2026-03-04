from __future__ import annotations

from typing import TYPE_CHECKING

import wandb
from miles.utils.tensorboard_utils import _TensorboardAdapter

from . import wandb_utils

if TYPE_CHECKING:
    from miles.utils.ft.agents.tracking_agent import FtTrackingAgent

_ft_tracking_agent: FtTrackingAgent | None = None


def set_ft_tracking_agent(agent: FtTrackingAgent | None) -> None:
    global _ft_tracking_agent
    _ft_tracking_agent = agent


def init_tracking(args, primary: bool = True, **kwargs):
    if primary:
        wandb_utils.init_wandb_primary(args, **kwargs)
    else:
        wandb_utils.init_wandb_secondary(args, **kwargs)


# TODO further refactor, e.g. put TensorBoard init to the "init" part
def log(args, metrics, step_key: str):
    if args.use_wandb:
        wandb.log(metrics)

    if args.use_tensorboard:
        metrics_except_step = {k: v for k, v in metrics.items() if k != step_key}
        _TensorboardAdapter(args).log(data=metrics_except_step, step=metrics[step_key])

    if _ft_tracking_agent is not None:
        step_value = metrics.get(step_key, 0)
        metrics_without_step = {k: v for k, v in metrics.items() if k != step_key}
        _ft_tracking_agent.log(metrics=metrics_without_step, step=int(step_value))
