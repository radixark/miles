"""Device calls `torch.accelerator` does not cover, split between CUDA and TPU (torch_tpu) by MILES_ACCELERATOR."""

import os

import torch

DEVICE_TYPE = os.environ.get("MILES_ACCELERATOR", "cuda")
assert DEVICE_TYPE in ("cuda", "tpu"), f"MILES_ACCELERATOR={DEVICE_TYPE!r}; expected cuda or tpu"
IS_TPU = DEVICE_TYPE == "tpu"

if IS_TPU:
    import torch_tpu  # noqa: F401  registers the tpu device with torch


def set_local_device(local_rank: int) -> None:
    # on TPU, touching the device before init_process_group breaks the multi-process rendezvous
    if not IS_TPU:
        torch.cuda.set_device(f"cuda:{local_rank}")


def current_device() -> torch.device:
    if IS_TPU:
        return torch.device("tpu", 0)
    return torch.device("cuda", torch.cuda.current_device())


def compile_dynamic(fn):
    # the TPU compile backend rejects dynamic shapes; these helpers only fuse kernels
    return fn if IS_TPU else torch.compile(dynamic=True)(fn)
