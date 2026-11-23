"""Trainable FP8 weights: BF16 master parameters behind serving's processed FP8 representation.

The master is the dequantized checkpoint (exact for E4M3 with power-of-two block scales), moved
once off the weight-sync grid by ``master_from_checkpoint`` (``off_grid``) without changing its
requantized bytes.
The forward reads a cached processed (payload, scale) pair. At install the cache holds the
checkpoint's own bytes, so step-0 scores equal rollout; requantizing the dequantized
checkpoint does not reproduce them. After each optimizer step the cache is rebuilt from the
master with the weight-sync quantizer and serving's loader requantization, the same bytes
rollout receives.

Backward is evaluated at the served weight (``served``: the dequantization of the processed bytes the forward ran),
and the gradient goes straight through to the master (``jacobian_weight``). The master is off the served values by
up to half a quantization cell, so a backward evaluated at it would differentiate another network than the forward.
"""

from __future__ import annotations

import weakref

import torch

from miles_plugins.top.off_grid import E4M3_128_UE8M0, offset_off_grid

# The grid quantize_checkpoint_format writes, which the off-grid offset keys on.
SYNC_GRID = E4M3_128_UE8M0
_EPOCH = [0]
# Every FP8Master / ExpertFP8Masters, so sleep can drop the caches an optimizer step made stale.
_LIVE: weakref.WeakSet = weakref.WeakSet()


def weights_epoch() -> int:
    return _EPOCH[0]


def advance_weights_epoch() -> None:
    """The masters changed, or serving now holds the sync quantizer's bytes (weight_events)."""
    _EPOCH[0] += 1


def drop_stale_caches() -> int:
    """Free every cache whose masters changed since it was built; returns the bytes released.

    A stale cache is never read again (``processed`` rebuilds it from the master), but while it
    lives the train offload backs it up to disk and reads it back, and it pins the allocator
    segments it was carved from. Fresh caches (the checkpoint's own bytes before the first sync)
    are kept: requantizing the master need not reproduce them."""
    freed = 0
    for m in list(_LIVE):
        if m._cache is not None and m._epoch != weights_epoch():
            freed += sum(t.numel() * t.element_size() for t in _tensors(m._cache))
            m._cache = None
    return freed


def materialize_caches() -> None:
    """Rebuild stale caches now (after the masters are restored, before any activation exists),
    so they are allocated together instead of between the first forward's activations."""
    for m in list(_LIVE):
        m.processed()


def _tensors(x):
    if isinstance(x, torch.Tensor):
        yield x
    elif isinstance(x, (tuple, list)):
        for y in x:
            yield from _tensors(y)


def dequantize_block_fp8(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """E4M3 payload x 128x128 block scales -> BF16: exact for power-of-two scales (DeepSeek-V4),
    nearest BF16 otherwise (GLM's FP32 scales). The forward reads the checkpoint bytes either way;
    the master is where the straight-through gradient is taken and later requantization starts."""
    *batch, n, k = weight.shape
    blocks = (*batch, -(-n // 128), -(-k // 128))
    # Other formats (FP4 experts, 1x128 or 32x32 scales) would broadcast into a wrong BF16 weight without an error.
    if weight.dtype != torch.float8_e4m3fn or tuple(scale.shape) != blocks:
        raise ValueError(
            f"[top] expected an E4M3 weight with 128x128 block scales {blocks}, got {weight.dtype} "
            f"{tuple(weight.shape)} with scales {tuple(scale.shape)}"
        )
    s = scale.float()
    pow2 = bool(torch.all(torch.frexp(s).mantissa == 0.5))
    s = s.repeat_interleave(128, -2)[..., :n, :].repeat_interleave(128, -1)[..., :k]
    full = weight.float() * s
    out = full.to(torch.bfloat16)
    if pow2 and not torch.equal(out.float(), full):
        raise ValueError("[top] power-of-two block-FP8 checkpoint does not dequantize exactly to BF16")
    return out


def master_from_checkpoint(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """The BF16 master of a checkpoint block-FP8 weight: its dequantization, with every element on the weight-sync
    grid moved off it (requantized bytes unchanged). Power-of-two-scaled checkpoints (DeepSeek-V4) dequantize exactly
    onto that grid; FP32-scaled ones (GLM) land on it only where a value happens to be representable there."""
    return offset_off_grid(dequantize_block_fp8(weight, scale), SYNC_GRID)


def served_from_checkpoint(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """The BF16 weight serving computes with at the first rollout for a checkpoint block-FP8 pair: Miles syncs the
    masters before it, so serving holds the sync quantizer's bytes of the master (for linears and experts the bytes of
    the checkpoint's dequantization, which the off-grid offset leaves unchanged)."""
    return dequantize_block_fp8(*quantize_checkpoint_format(master_from_checkpoint(weight, scale)))


def quantize_checkpoint_format(master: torch.Tensor):
    """The weight-sync rule: E4M3 payload with FP32 power-of-two 128x128 block scales."""
    from sglang.srt.layers.quantization.fp8_utils import quant_weight_ue8m0

    return quant_weight_ue8m0(master.detach().contiguous(), weight_block_size=[128, 128])


class FP8Master:
    """One block-FP8 weight: a BF16 master Parameter and the cached processed form.

    ``process(payload, scale)`` maps checkpoint-format tensors to what the forward kernel reads
    (serving's loader transform for this weight). ``synced`` maps the weight-sync quantizer's output
    to what serving holds after a sync, when that differs from loading it as a checkpoint. ``value(processed)`` takes
    whatever ``process`` returns (one tensor or a pair) and is the BF16 weight those bytes hold (exact), where the
    backward is evaluated."""

    def __init__(self, payload, scale, process, value, synced=None):
        self.master = torch.nn.Parameter(master_from_checkpoint(payload, scale), requires_grad=True)
        self.process = process
        self.value = value
        self.synced = synced or process
        self._cache = process(payload, scale)
        self._epoch = weights_epoch()
        _LIVE.add(self)

    def processed(self):
        if self._epoch != weights_epoch():
            self._cache = None  # the stale pair is never read again; free it before building the new one
            self._cache = self.synced(*quantize_checkpoint_format(self.master))
            self._epoch = weights_epoch()
        return self._cache

    def served(self) -> torch.Tensor:
        """The BF16 weight the forward computes with; rebuilt on each call, so nothing beside the master stays."""
        return self.value(self.processed())

    def jacobian_weight(self) -> torch.Tensor:
        """The served weight's value, bit for bit, with its gradient going to the master."""
        from miles_plugins.top.fp8_forward import _StraightThrough

        return _StraightThrough.apply(self.served(), self.master)


class ExpertFP8Masters:
    """A rank's local experts: per-expert BF16 masters (owned by the SequentialMLP experts, so
    weight sync sees ordinary expert names) behind one stacked processed FP8 pair. ``packed`` is
    the checkpoint's processed pair; ``process`` maps the sync quantizer's stacked output to what
    serving holds after a sync; ``value`` maps a stacked processed pair to the BF16 weights it holds."""

    def __init__(self, w13, w2, packed, process, value):
        self.w13, self.w2 = list(w13), list(w2)
        self.process = process
        self.value = value
        self._cache = packed
        self._epoch = weights_epoch()
        _LIVE.add(self)

    def processed(self):
        if self._epoch != weights_epoch():
            # Rebuilt inside the first forward of a step, beside its activations: free the stale pair first and
            # quantize one expert at a time, not a stacked layer (6 GiB of FP32 temporaries per layer at EP4).
            self._cache = None
            self._cache = (*self.process(*_quantize_experts(self.w13)), *self.process(*_quantize_experts(self.w2)))
            self._epoch = weights_epoch()
        return self._cache

    def served(self) -> tuple[torch.Tensor, torch.Tensor]:
        """The stacked BF16 weights the forward computes with, (w13 [E, 2I, H], w2 [E, H, I]); rebuilt on each call."""
        w13, s13, w2, s2 = self.processed()
        return self.value(w13, s13), self.value(w2, s2)


def _quantize_experts(masters):
    """quantize_checkpoint_format of the stacked masters, one expert at a time; its 128x128 blocks never span two
    experts (row counts are multiples of 128), so the bytes are the stacked call's."""
    payloads = scales = None
    for e, master in enumerate(masters):
        if master.shape[-2] % 128:
            raise ValueError(f"[top] expert rows {master.shape[-2]} are not a multiple of the 128-row block")
        payload, scale = quantize_checkpoint_format(master)
        if payloads is None:
            payloads = payload.new_empty((len(masters), *payload.shape))
            scales = scale.new_empty((len(masters), *scale.shape))
        payloads[e], scales[e] = payload, scale
    return payloads, scales


def install_expert_masters(experts, ws13, ss13, ws2, ss2):
    """A rank's local experts from checkpoint E4M3 payloads and block scales (fused gate/up, then down).

    The per-expert BF16 masters live in the SequentialMLP experts, so the optimizer and weight sync see them;
    the forward reads one stacked processed FP8 pair (ExpertFP8Masters)."""
    from miles_plugins.top.fp8_forward import dequantize_processed, prepare_weight, synced_weight

    w13, s13 = prepare_weight(torch.stack(ws13), torch.stack(ss13))
    w2, s2 = prepare_weight(torch.stack(ws2), torch.stack(ss2))
    for expert, w, sc, w_2, sc2 in zip(experts.local_experts, ws13, ss13, ws2, ss2, strict=True):
        expert.linear_fc1.weight.data = master_from_checkpoint(w, sc)
        expert.linear_fc2.weight.data = master_from_checkpoint(w_2, sc2)
    experts.fp8_masters = ExpertFP8Masters(
        [e.linear_fc1.weight for e in experts.local_experts],
        [e.linear_fc2.weight for e in experts.local_experts],
        (w13, s13, w2, s2),
        synced_weight,
        dequantize_processed,
    )
