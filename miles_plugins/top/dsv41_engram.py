"""DeepSeek-V4.1 Engram on SGLang's engram program (``sglang.srt.true_on_policy.dsv41_engram``).

The table stays the checkpoint's host-mapped FP8 rows (DeepSeekV41Engram.lookup, an exact E4M3 x UE8M0 dequant to
BF16, as serving's lookup). Hash ids come from SGLang's hash kernel over the packed requests, each from position 0.
``linear_wkv`` is installed as a 32x32 FP8 master (fp8_b32_linear). The gate is SGLang's fused kernel on the BF16 q/k
weights serving holds; its gradient is the torch gate's (``engram_gate``) at the FP32 masters.
"""

from __future__ import annotations

import torch

from miles_plugins.models.deepseek_v4_1.engram import DeepSeekV41Engram, engram_gate
from miles_plugins.top.binding import CaptureTap, canon_dump
from miles_plugins.top.kernels.surrogate import surrogate


def packed_hash_ids(hasher, input_ids, lengths, layer_index, *, image_token_id=None):
    """[T, n_hash_cols] hash ids of one engram layer for requests packed back to back; n-grams stop at images."""
    from sglang.srt.true_on_policy.dsv41_engram import hash_packed

    if input_ids.ndim != 1 or sum(int(n) for n in lengths) != input_ids.shape[0]:
        raise NotImplementedError("[top] V4.1 engram hashes packed requests whose lengths cover every token id")
    return hash_packed(hasher, input_ids, lengths, image_token_id=image_token_id)[:, layer_index]


def top_engram_gate(x, kv, q_weight, k_weight, eps, clamp_value, input_ids=None, image_token_id=None):
    """SGLang's fused gate on serving's BF16 q/k values; gradient of the torch gate at the masters. With
    ``input_ids`` [T], rows holding ``image_token_id`` keep x, so their gradient passes straight through."""
    from sglang.srt.true_on_policy.dsv41_engram import gate

    if x.ndim != 3 or kv.ndim != 2:
        raise NotImplementedError(
            "[top] V4.1 engram gate takes streams [T, hc_mult, dim] and keys [T, (hc_mult+1)*dim]"
        )
    image = None if input_ids is None else (input_ids == image_token_id)[:, None, None]

    def reference(x, kv, q, k):
        out = engram_gate(x, kv, q, k, eps, clamp_value, chunk=0)
        return out if image is None else torch.where(image, x, out)

    return surrogate(
        lambda x, kv, q, k: gate(
            x, kv, q.to(torch.bfloat16), k.to(torch.bfloat16), eps, clamp_value, input_ids, image_token_id
        ),
        reference,
        x,
        kv,
        q_weight,
        k_weight,
        name="engram_gate",
    )


class TopV41Engram(DeepSeekV41Engram):
    """One engram layer of the TOP block; ``hasher`` is SGLang's EngramHasher for the pinned config."""

    def __init__(self, config, layer_id, tp_group, cp_group=None, *, hasher=None, image_token_id=None):
        super().__init__(config, layer_id, tp_group, cp_group)
        self.hasher = hasher
        # Serving's Engram for a checkpoint with a vision tower: image tokens block n-grams and keep their streams.
        self.image_token_id = image_token_id
        # The rows the lookup returned, a capture point of the per-layer VJP (top_vjp replays them).
        self.top_engram_rows = CaptureTap()

    def forward(self, hidden_states, input_ids, lengths=None):
        if self.hasher is None or lengths is None:
            raise NotImplementedError("[top] V4.1 engram needs SGLang's hasher and the packed request lengths")
        if self.tp_size > 1 or self.cp_size > 1 or self.sequence_parallel:
            raise NotImplementedError("[top] V4.1 engram runs at TP1 without CP or sequence parallelism")
        tokens, bsz, width = hidden_states.shape
        with torch.no_grad():
            flat_ids = input_ids.reshape(-1)
            ids = packed_hash_ids(
                self.hasher, flat_ids, lengths, self.layer_hash_index, image_token_id=self.image_token_id
            )
            rows = self.top_engram_rows(self.lookup(ids))
        kv, _ = self.linear_wkv(rows.flatten(-2))
        x = hidden_states.view(tokens * bsz, self.hc_mult, self.dim)
        out = top_engram_gate(
            x,
            kv.view(tokens * bsz, -1),
            self.q_weight,
            self.k_weight,
            self.eps,
            self.clamp_value,
            None if self.image_token_id is None else flat_ids,
            self.image_token_id,
        )
        canon_dump("engram_out", out)
        return out.view(tokens, bsz, width)
