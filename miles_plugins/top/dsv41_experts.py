"""DeepSeek-V4.1 routed experts: BF16 masters behind the release's MXFP4 bytes, served cast to 128x128 FP8.

Serving loads each expert's MXFP4 pair (E2M1 nibbles, UE8M0 scale per 32) and casts it losslessly to E4M3 with
128x128 scales (SGLANG_DSV4_FP4_DEQUANT), which TOP requantizes to packed UE8M0 for DS4's FP8 expert program. The
master is the exact dequantization moved off the MXFP4 grid (off_grid). The weight sync quantizes a master on that
grid, casts and requantizes it the same way, and sends the result; serving keeps the bytes as sent, and the trainer's
processed experts are those bytes. At install they are the checkpoint's own cast. The expert backward
(fp8_forward.trainable_expert_rows) is evaluated at the values those bytes hold (``served``), straight through to the
masters, which sit up to half an MXFP4 cell (~10% of the weight) off them.
"""

from __future__ import annotations

import torch

from miles_plugins.top.fp8_masters import weights_epoch
from miles_plugins.top.off_grid import MXFP4, offset_off_grid, requantize

_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def dequantize_mxfp4(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """[N, K/2] int8 E2M1 pairs (low nibble first) x [N, K/32] UE8M0 -> [N, K] BF16, exactly."""
    if packed.dtype not in (torch.int8, torch.uint8) or packed.ndim != 2:
        raise ValueError(f"[top] expected packed MXFP4 int8 [N, K/2], got {packed.dtype} {tuple(packed.shape)}")
    n, k = packed.shape[0], packed.shape[1] * 2
    if tuple(scale.shape) != (n, k // 32):
        raise ValueError(f"[top] MXFP4 scale {tuple(scale.shape)} is not one per 32 of {(n, k)}")
    u = packed.view(torch.uint8)
    levels = torch.tensor(_E2M1, device=packed.device)
    codes = torch.stack([u & 15, u >> 4], dim=-1).flatten(-2)
    value = torch.where((codes & 8) != 0, -levels[(codes & 7).long()], levels[(codes & 7).long()])
    full = value * scale.float().repeat_interleave(32, dim=-1)
    out = full.to(torch.bfloat16)
    if not torch.equal(out.float(), full):
        raise ValueError("[top] MXFP4 expert does not dequantize exactly to BF16")
    return out


def _pack_mxfp4(codes: torch.Tensor) -> torch.Tensor:
    """[N, K] E2M1 nibbles -> [N, K/2] int8, low nibble first (SGLang's cast_e2m1fn_to_e4m3fn order)."""
    pairs = codes.view(codes.shape[0], -1, 2)
    return (pairs[..., 0] | (pairs[..., 1] << 4)).view(torch.int8)


def served_form(packed: torch.Tensor, scale: torch.Tensor):
    """What serving holds for one expert projection's MXFP4 pair: the lossless cast to E4M3 / 128x128, then TOP's
    UE8M0 requant (packed scales)."""
    from sglang.srt.layers.quantization.fp8 import cast_e2m1fn_to_e4m3fn

    from miles_plugins.top.fp8_forward import prepare_weight

    payload, cast_scale = cast_e2m1fn_to_e4m3fn(packed.view(torch.int8), scale.float())
    # Serving stacks the cast's scales as FP32 before its requant (Fp8MoEMethod.process_weights_after_loading).
    return prepare_weight(payload, cast_scale.float())


def sync_form(master: torch.Tensor):
    """The weight-sync rule for one expert projection: MXFP4 on the master, then serving's cast and requant."""
    codes, scale = requantize(master.detach().contiguous(), MXFP4)
    return served_form(_pack_mxfp4(codes), scale)


class V41ExpertMasters:
    """A rank's local experts: per-expert BF16 masters (owned by the SequentialMLP experts) behind one stacked
    processed FP8 pair in DS4's expert layout (w13 = [w1; w3] per expert)."""

    def __init__(self, w13, w2, packed):
        self.w13, self.w2 = list(w13), list(w2)
        self._cache = packed
        self._epoch = weights_epoch()

    def processed(self):
        if self._epoch != weights_epoch():
            self._cache = None
            self._cache = (*_stack(sync_form(m) for m in self.w13), *_stack(sync_form(m) for m in self.w2))
            self._epoch = weights_epoch()
        return self._cache

    def served(self) -> tuple[torch.Tensor, torch.Tensor]:
        """The stacked BF16 weights the forward computes with, (w13 [E, 2I, H], w2 [E, H, I]); rebuilt on each call."""
        from miles_plugins.top.fp8_forward import dequantize_processed

        w13, s13, w2, s2 = self.processed()
        return dequantize_processed(w13, s13), dequantize_processed(w2, s2)


def _stack(pairs):
    payloads, scales = zip(*pairs, strict=True)
    return torch.stack(payloads), torch.stack(scales)


def install_expert_masters(experts, fp4_13, fp4_2):
    """``fp4_13`` / ``fp4_2``: per local expert, ((w1 packed, w1 scale), (w3 packed, w3 scale)) and (w2 packed, w2
    scale) from the checkpoint. Masters go into the SequentialMLP experts; the forward reads the stacked cast."""
    served13, served2 = [], []
    for expert, ((p1, s1), (p3, s3)), (p2, s2) in zip(experts.local_experts, fp4_13, fp4_2, strict=True):
        w13 = torch.cat([dequantize_mxfp4(p1, s1), dequantize_mxfp4(p3, s3)])
        expert.linear_fc1.weight.data = offset_off_grid(w13, MXFP4)
        expert.linear_fc2.weight.data = offset_off_grid(dequantize_mxfp4(p2, s2), MXFP4)
        served13.append(served_form(torch.cat([p1, p3]), torch.cat([s1, s3])))
        served2.append(served_form(p2, s2))
    experts.fp8_masters = V41ExpertMasters(
        [e.linear_fc1.weight for e in experts.local_experts],
        [e.linear_fc2.weight for e in experts.local_experts],
        (*_stack(served13), *_stack(served2)),
    )
