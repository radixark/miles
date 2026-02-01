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


class _LossDataDumper:
    def __init__(self):
        self._buffer: list[dict] = []

    def accumulate(self, args, batch: dict, loss_data: dict):
        if getattr(args, "save_debug_loss_data", None) is None:
            return
        self._buffer.append(
            dict(
                microbatch_offset=batch["debug_microbatch_offset"],
                batch=_detach_clone_cpu(batch),
                loss_data=_detach_clone_cpu(loss_data),
            )
        )

    def flush(self, args):
        if getattr(args, "save_debug_loss_data", None) is None:
            return
        if not self._buffer:
            return

        rollout_id = _assert_all_equal([mb["batch"]["debug_rollout_id"] for mb in self._buffer])
        step_id = _assert_all_equal([mb["batch"]["debug_step_id"] for mb in self._buffer])
        rank = torch.distributed.get_rank()

        path_template = args.save_debug_loss_data
        path = Path(path_template.format(rollout_id=rollout_id, step_id=step_id, rank=rank))
        assert not path.exists(), f"Debug loss file already exists: {path}"
        logger.info(f"Save debug loss data to {path} start ({len(self._buffer)} microbatches)")
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(
            dict(microbatches=list(self._buffer)),
            path,
        )
        self._buffer.clear()
        logger.info(f"Save debug loss data end")


def _assert_all_equal(values: list):
    assert values, "Empty list"
    first = values[0]
    for v in values[1:]:
        assert v == first, f"Values not equal: {first} vs {v}"
    return first


loss_data_dumper = _LossDataDumper()


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
