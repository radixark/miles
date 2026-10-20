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


def bind_device_before_process_group(local_rank: int) -> None:
    # CUDA binds before init_process_group, as it always has; a TPU touched this early breaks the rendezvous
    if not IS_TPU:
        torch.cuda.set_device(f"cuda:{local_rank}")


def bind_device_after_process_group() -> None:
    # TPU binds its one visible device once the process group is up
    if IS_TPU:
        torch.accelerator.set_device_index(0)


def current_device() -> torch.device:
    return torch.device(DEVICE_TYPE, torch.accelerator.current_device_index())


@functools.cache
def device_name() -> str | None:
    """The CUDA device's marketing name (for peak-FLOPs lookup); None where there is none to report."""
    if IS_TPU or not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_name()


def compile_dynamic(fn):
    """`torch.compile(dynamic=True)` on CUDA; on TPU, whose compile backend rejects dynamic shapes, `fn` runs eager.

    Only for helpers without data-dependent control flow (`compute_approx_kl`, `compute_policy_loss`, the entropy's
    `mul_reduce`). Eager results are numerically close to the compiled ones, not bit-identical: fusion can change
    reduction order and intermediate rounding. Eager `mul_reduce` also materializes the [rows, vocab] product the
    fused kernel avoids, one more fp32 transient per log-prob chunk.
    """
    return fn if IS_TPU else torch.compile(dynamic=True)(fn)
