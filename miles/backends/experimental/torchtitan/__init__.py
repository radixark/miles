"""TorchTitan training backend for miles RL (experimental).

Wraps a stock HuggingFace model with torchtitan's ParallelDims device mesh and
torch-native FSDP2, mirroring the experimental FSDP backend's RL contract so all
of miles' shared training/rollout/weight-sync machinery is reused unchanged.
"""

import logging

try:
    from .actor import TorchTitanTrainRayActor
    from .arguments import load_torchtitan_args
except ImportError as e:  # torchtitan / FSDP2 deps not importable
    logging.warning(f"torchtitan backend unavailable: {e}")

    def _raise_import_error(*args, **kwargs):
        raise ImportError(
            "torchtitan backend is unavailable. Ensure torchtitan is importable "
            "(PYTHONPATH must include the torchtitan checkout) and torch has FSDP2 support."
        )

    TorchTitanTrainRayActor = _raise_import_error
    load_torchtitan_args = _raise_import_error

__all__ = ["load_torchtitan_args", "TorchTitanTrainRayActor"]
