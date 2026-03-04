from __future__ import annotations

import torch.distributed as dist
import wandb

from miles.utils.ft.agents.tracking_agent import FtTrackingAgent
from miles.utils.tensorboard_utils import _TensorboardAdapter

from . import wandb_utils

_ft_tracking_agent: FtTrackingAgent | None = None


def init_tracking(args, primary: bool = True, **kwargs) -> None:
    global _ft_tracking_agent

    if primary:
        wandb_utils.init_wandb_primary(args, **kwargs)
    else:
        wandb_utils.init_wandb_secondary(args, **kwargs)

    if args.use_fault_tolerance and _ft_tracking_agent is None and dist.is_initialized():
        _ft_tracking_agent = FtTrackingAgent(rank=dist.get_rank())


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
