"""torchtitan as a full miles training backend (experimental).

torchtitan supplies the training side wholesale: native models/kernels, parallelize
composition (FSDP2/TP over ParallelDims), streaming HF-checkpoint load, optimizer/
lr-scheduler/grad-clip. miles keeps RL orchestration: rollout via colocated SGLang,
data packing, loss, the weight-sync wire protocol, checkpoint conventions.
"""

import logging

from . import compat  # noqa: F401  (must run before any other torchtitan import)

try:
    from .actor import TorchTitanTrainRayActor
    from .arguments import load_torchtitan_args
except ImportError as e:  # torchtitan not importable in this environment
    logging.warning(f"torchtitan backend unavailable: {e}")

    def _raise_import_error(*args, **kwargs):
        raise ImportError(
            "torchtitan backend is unavailable. Ensure torchtitan is installed "
            "(pip install --no-deps 'git+https://github.com/pytorch/torchtitan@"
            f"{compat.TORCHTITAN_PINNED_COMMIT}') and torch has FSDP2 support."
        )

    TorchTitanTrainRayActor = _raise_import_error
    load_torchtitan_args = _raise_import_error

__all__ = ["load_torchtitan_args", "TorchTitanTrainRayActor"]
