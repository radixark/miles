import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def save_debug_train_data(args, *, rollout_id, rollout_data):
    if args.save_debug_train_data is not None:
        save_debug_train_data_for_rank(
            args, rollout_id=rollout_id, rollout_data=rollout_data, rank=torch.distributed.get_rank()
        )


def save_debug_train_data_for_rank(args, *, rollout_id, rollout_data, rank):
    if (path_template := args.save_debug_train_data) is not None:
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
