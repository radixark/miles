from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache

import torch

__all__ = ["batch_invariant_log_softmax"]

if not all(hasattr(torch.library, name) for name in ("custom_op", "register_autocast")):
    raise RuntimeError(
        "The sglang_batch_invariant log-softmax backend requires a recent PyTorch "
        "with torch.library custom-op and autocast registration support. "
        "Use the pinned Miles container."
    )


@lru_cache(maxsize=1)
def _load_sglang_log_softmax() -> Callable[[torch.Tensor, int], torch.Tensor]:
    try:
        from sglang.srt.batch_invariant_ops import log_softmax
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "The sglang_batch_invariant log-softmax backend requires an SGLang "
            "installation that exports sglang.srt.batch_invariant_ops.log_softmax"
        ) from exc

    if not callable(log_softmax):
        raise RuntimeError("sglang.srt.batch_invariant_ops.log_softmax exists but is not callable")
    return log_softmax


@torch.library.custom_op(
    "miles::sglang_batch_invariant_log_softmax",
    mutates_args=(),
    device_types="cuda",
)
def _batch_invariant_log_softmax(input: torch.Tensor, dim: int) -> torch.Tensor:
    if input.dtype != torch.float32:
        raise TypeError(f"batch-invariant log_softmax requires FP32 input, got {input.dtype}")
    if input.ndim == 0 or dim not in (-1, input.ndim - 1):
        raise ValueError("batch-invariant log_softmax only supports the last dimension")
    if input.numel() == 0:
        raise ValueError("batch-invariant log_softmax does not support empty tensors")

    log_softmax = _load_sglang_log_softmax()
    try:
        return log_softmax(input, dim=dim)
    except TypeError as exc:
        raise RuntimeError(
            "Incompatible SGLang batch-invariant log_softmax API; expected " "log_softmax(input, dim=-1)"
        ) from exc


@_batch_invariant_log_softmax.register_fake
def _batch_invariant_log_softmax_fake(input: torch.Tensor, dim: int) -> torch.Tensor:
    return torch.empty_like(input, memory_format=torch.contiguous_format)


def _setup_context(ctx, inputs, output) -> None:
    input, dim = inputs
    ctx.save_for_backward(output)
    ctx.dim = dim
    ctx.input_dtype = input.dtype


def _backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
    (output,) = ctx.saved_tensors
    grad_input = torch.ops.aten._log_softmax_backward_data.default(
        grad_output.contiguous(),
        output,
        ctx.dim,
        ctx.input_dtype,
    )
    return grad_input, None


_batch_invariant_log_softmax.register_autograd(
    _backward,
    setup_context=_setup_context,
)
torch.library.register_autocast(
    "miles::sglang_batch_invariant_log_softmax",
    "cuda",
    torch.float32,
)


def batch_invariant_log_softmax(input: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Apply SGLang's FP32 batch-invariant log-softmax with Miles autograd."""
    if input.layout is not torch.strided:
        raise TypeError(f"batch-invariant log_softmax requires a strided tensor, got {input.layout}")
    if input.ndim == 0:
        raise ValueError("batch-invariant log_softmax requires at least one dimension")
    if dim not in (-1, input.ndim - 1):
        raise ValueError("batch-invariant log_softmax only supports the last dimension")
    if input.shape[-1] == 0:
        raise ValueError("batch-invariant log_softmax requires a non-empty last dimension")
    if input.device.type != "cuda":
        raise RuntimeError("batch-invariant log_softmax only supports CUDA tensors")
    if input.dtype != torch.float32:
        raise TypeError(f"batch-invariant log_softmax requires FP32 input, got {input.dtype}")
    if input.numel() == 0:
        raise ValueError("batch-invariant log_softmax does not support empty tensors")
    return _batch_invariant_log_softmax(input.contiguous(), -1)
