"""DeepSeek-V4.1 32x32 block-FP8 projections: BF16 masters behind serving's bytes, forward on SGLang's fp8_b32_linear.

The checkpoint's E4M3 payloads with power-of-two 32x32 block scales dequantize exactly to BF16; that dequantization,
moved once off the sync grid (``off_grid``), is the master. Serving keeps the payload and its FP32 block scales as
loaded (no requantization for 32x32), so the processed pair is the checkpoint's bytes at install and the sync
quantizer's bytes (``off_grid.requantize`` on ``E4M3_32_UE8M0``) after every weight event. Backward is the BF16 GEMM
at the served weight (``served``: the processed bytes' dequantization, rebuilt in backward), straight through to the
master, which sits up to half a cell off it (``linear_surrogate`` with ``value_fn``, as fp8_forward's 128x128 FP8).
"""

from __future__ import annotations

import torch

from miles_plugins.top.fp8_masters import weights_epoch
from miles_plugins.top.off_grid import E4M3_32_UE8M0, offset_off_grid, requantize

BLOCK = 32


def dequantize_b32(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """E4M3 payload x power-of-two 32x32 block scales -> BF16, exactly."""
    if weight.dtype != torch.float8_e4m3fn or weight.ndim != 2:
        raise ValueError(f"[top] expected a 2-D E4M3 payload, got {weight.dtype} {tuple(weight.shape)}")
    n, k = weight.shape
    if tuple(scale.shape) != (-(-n // BLOCK), -(-k // BLOCK)):
        raise ValueError(f"[top] scale {tuple(scale.shape)} is not the 32x32 block grid of {tuple(weight.shape)}")
    s = scale.float()
    if not bool(torch.all(torch.frexp(s).mantissa == 0.5)):
        raise ValueError("[top] 32x32 block-FP8 scales must be powers of two (UE8M0)")
    full = weight.float() * s.repeat_interleave(BLOCK, 0)[:n].repeat_interleave(BLOCK, 1)[:, :k]
    out = full.to(torch.bfloat16)
    if not torch.equal(out.float(), full):
        raise ValueError("[top] 32x32 block-FP8 checkpoint does not dequantize exactly to BF16")
    return out


def quantize_b32(master: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The weight-sync rule: E4M3 payload and FP32 power-of-two 32x32 block scales, as serving then holds them."""
    codes, scale = requantize(master.detach().contiguous(), E4M3_32_UE8M0)
    return codes.view(torch.float8_e4m3fn), scale


class FP8B32Master:
    """One 32x32 block-FP8 weight: the BF16 master Parameter and the processed (payload, FP32 scale) pair."""

    def __init__(self, payload: torch.Tensor, scale: torch.Tensor):
        self.master = torch.nn.Parameter(
            offset_off_grid(dequantize_b32(payload, scale), E4M3_32_UE8M0), requires_grad=True
        )
        self._cache = (payload.contiguous(), scale.float().contiguous())
        self._epoch = weights_epoch()

    def processed(self):
        if self._epoch != weights_epoch():
            self._cache = None
            self._cache = quantize_b32(self.master)
            self._epoch = weights_epoch()
        return self._cache

    def served(self) -> torch.Tensor:
        """The BF16 weight the forward computes with; rebuilt on each call, so nothing beside the master stays."""
        return dequantize_b32(*self.processed())

    def jacobian_weight(self) -> torch.Tensor:
        """The served weight's value, bit for bit, with its gradient going to the master."""
        from miles_plugins.top.fp8_forward import _StraightThrough

        return _StraightThrough.apply(self.served(), self.master)


class TrainableFP8B32Linear(torch.nn.Module):
    """SGLang's 32x32 FP8 forward on the processed bytes; backward at the served weight, straight through to
    ``weight`` (the master)."""

    def __init__(self, payload: torch.Tensor, scale: torch.Tensor, *, name: str = "fp8_b32_linear"):
        super().__init__()
        self._fp8 = FP8B32Master(payload, scale)
        self.weight = self._fp8.master
        self.dump_name = name
        self.is_top_fp8 = True

    def processed(self):
        return self._fp8.processed()

    def served(self):
        return self._fp8.served()

    def reference(self, x):
        """``x @ served.T`` in BF16 for a gradient reference, with the weight gradient going to the master."""
        from miles_plugins.top.kernels.surrogate import linear_surrogate

        return linear_surrogate(
            lambda x, m: x @ self.served().t(),
            x,
            self.weight,
            name=f"{self.dump_name}_reference",
            value_fn=self.served,
        )

    def forward(self, x, weight=None, runtime_gather_output=None):
        from sglang.srt.true_on_policy.fp8.fp8_b32_linear import linear

        from miles_plugins.top.binding import canon_dump
        from miles_plugins.top.kernels.surrogate import linear_surrogate

        if weight is not None or runtime_gather_output:
            raise NotImplementedError("[top] trainable FP8 32x32 linear admits no external weight or gather")
        w, s = self.processed()
        canon_dump(f"{self.dump_name}_in", x)
        out = linear_surrogate(
            lambda x, m: linear(x, w, s), x, self.weight, name="fp8_b32_linear", value_fn=self.served
        )
        canon_dump(f"{self.dump_name}_out", out)
        return out, None


def trainable_clamped_swiglu_mlp(x, fc1: TrainableFP8B32Linear, fc2: TrainableFP8B32Linear, swiglu_limit):
    """DeepSeek-V4.1 shared expert: SGLang's 32x32 FP8 clamped MLP on the processed bytes; backward at the served
    weights, straight through to the BF16 masters."""
    from sglang.srt.true_on_policy.fp8.fp8_b32_linear import SWIGLU_LIMIT, clamped_swiglu_mlp

    from miles_plugins.top.binding import canon_dump
    from miles_plugins.top.fp8_forward import _StraightThrough, clamped_swiglu
    from miles_plugins.top.kernels.surrogate import surrogate

    if not (isinstance(fc1, TrainableFP8B32Linear) and isinstance(fc2, TrainableFP8B32Linear)):
        raise NotImplementedError("[top] the V4.1 shared expert reads two 32x32 FP8 projections")
    if swiglu_limit != SWIGLU_LIMIT:
        raise NotImplementedError(f"[top] the V4.1 shared expert clamps at {SWIGLU_LIMIT}, got {swiglu_limit}")
    p1, p2 = fc1.processed(), fc2.processed()
    canon_dump("shared_expert_in", x)
    out = surrogate(
        lambda x, w1, w2: clamped_swiglu_mlp(x, p1, p2, swiglu_limit),
        lambda x, w1, w2: clamped_swiglu(x @ _StraightThrough.apply(fc1.served(), w1).t(), swiglu_limit)
        @ _StraightThrough.apply(fc2.served(), w2).t(),
        x,
        fc1.weight,
        fc2.weight,
        name="clamped_swiglu_b32",
    )
    canon_dump("shared_expert_out", out)
    return out
