from __future__ import annotations

import wandb

from miles.utils.ft.agents.tracking_agent import FtTrackingAgent
from miles.utils.tensorboard_utils import _TensorboardAdapter

from . import wandb_utils

_ft_tracking_agent: FtTrackingAgent | None = None


def set_ft_tracking_agent(agent: FtTrackingAgent | None) -> None:
    global _ft_tracking_agent
    _ft_tracking_agent = agent


def init_tracking(args, primary: bool = True, **kwargs) -> None:
    global _ft_tracking_agent

    if primary:
        wandb_utils.init_wandb_primary(args, **kwargs)
    else:
        wandb_utils.init_wandb_secondary(args, **kwargs)

    if "train" in args.ft_components and _ft_tracking_agent is None:
        _ft_tracking_agent = FtTrackingAgent()


# TODO further refactor, e.g. put TensorBoard init to the "init" part
def log(args, metrics, step_key: str):
    if args.use_wandb:
        wandb.log(metrics)

    if args.use_tensorboard or _ft_tracking_agent is not None:
        step_value = int(metrics.get(step_key, 0))
        metrics_without_step = {k: v for k, v in metrics.items() if k != step_key}

        if args.use_tensorboard:
            _TensorboardAdapter(args).log(data=metrics_without_step, step=step_value)

        if _ft_tracking_agent is not None:
            _ft_tracking_agent.log(metrics=metrics_without_step, step=step_value)
