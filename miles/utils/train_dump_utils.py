import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def save_debug_train_data(args, *, rollout_id, rollout_data):
    if (path_template := args.save_debug_train_data) is not None:
        rank = torch.distributed.get_rank()
        path = Path(path_template.format(rollout_id=rollout_id, rank=rank))
        logger.info(f"Save debug train data to {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            dict(
                rollout_id=rollout_id,
                rank=rank,
                rollout_data=rollout_data,
            ),
            path,
        )


def save_debug_loss_data(args, batch: dict, loss_data: dict):
    if (path_template := getattr(args, "save_debug_loss_data", None)) is None:
        return

    rank = torch.distributed.get_rank()

    path = Path(path_template.format(
        rollout_id=batch["debug_rollout_id"],
        step_id=batch["debug_step_id"],
        microbatch_id=batch["debug_microbatch_offset"],
        rank=rank,
    ))
    assert not path.exists(), f"Debug loss file already exists: {path}"
    logger.info(f"Save debug loss data to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        dict(
            rank=rank,
            batch=_detach_clone_cpu(batch),
            loss_data=_detach_clone_cpu(loss_data),
        ),
        path,
    )


def _detach_clone_cpu(x):
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.detach().clone().cpu()
    if isinstance(x, (list, tuple)):
        return [_detach_clone_cpu(item) for item in x]
    if isinstance(x, dict):
        return {k: _detach_clone_cpu(v) for k, v in x.items()}
    return x
