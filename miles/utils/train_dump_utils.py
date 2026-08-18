import logging
from pathlib import Path

import torch

from miles.backends.training_utils.parallel import get_parallel_state

logger = logging.getLogger(__name__)


def save_debug_train_data(args, *, rollout_id, rollout_data):
    if args.save_debug_train_data is not None:
        parallel_state = get_parallel_state()
        save_debug_train_data_for_rank(
            args,
            rollout_id=rollout_id,
            rollout_data=rollout_data,
            rank=torch.distributed.get_rank(),
            cp_rank=parallel_state.cp.rank,
            cp_size=parallel_state.cp.size,
        )


def save_debug_train_data_for_rank(args, *, rollout_id, rollout_data, rank, cp_rank=0, cp_size=1):
    """Write one rank's slice of a rollout.

    ``cp_rank`` / ``cp_size`` are passed in for the same reason ``rank`` is:
    this runs outside the process group in tests and offline tools.
    """
    if (path_template := args.save_debug_train_data) is not None:
        path = Path(path_template.format(rollout_id=rollout_id, rank=rank))
        logger.info(f"Save debug train data to {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            dict(
                rollout_id=rollout_id,
                rank=rank,
                rollout_data=rollout_data,
                cp_rank=cp_rank,
                cp_size=cp_size,
                qkv_format=args.qkv_format,
            ),
            path,
        )
