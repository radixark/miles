"""The training process's accelerator, chosen once by MILES_ACCELERATOR (cuda or tpu).

Everything `torch.accelerator` covers (synchronize, empty_cache, memory stats, the device index) is called
directly; this module holds only what differs between CUDA and TPU (torch_tpu).
"""

import functools
import os

import torch

DEVICE_TYPE = os.environ.get("MILES_ACCELERATOR", "cuda")
if DEVICE_TYPE not in ("cuda", "tpu"):
    raise ValueError(f"MILES_ACCELERATOR={DEVICE_TYPE!r}; expected cuda or tpu")
IS_TPU = DEVICE_TYPE == "tpu"

if IS_TPU:
    import torch_tpu  # noqa: F401  registers the tpu device with torch


def bind_local_device(local_rank: int) -> None:
    """Bind this process to its device. Call after init_process_group: TPU binds its one visible device at
    process-group init (touching it earlier breaks the rendezvous), CUDA binds lazily."""
    torch.accelerator.set_device_index(0 if IS_TPU else local_rank)


def current_device() -> torch.device:
    return torch.device(DEVICE_TYPE, torch.accelerator.current_device_index())


@functools.cache
def device_name() -> str | None:
    """The CUDA device's marketing name (for peak-FLOPs lookup); None where there is none to report."""
    if IS_TPU or not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_name()


def compile_dynamic(fn):
    # the TPU compile backend rejects dynamic shapes; these helpers only fuse kernels
    return fn if IS_TPU else torch.compile(dynamic=True)(fn)
