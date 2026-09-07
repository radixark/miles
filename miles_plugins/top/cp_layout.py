"""Ulysses CP: transform between megatron's local sequence shards and FA3's head shards.

Two conceptual transforms, implemented as four forward all-to-alls per attention call: one each
for Q, K, V, and output. The permutations preserve values; replicated KV gradients are summed
in backward, so this is not a claim of CP1-bitwise gradients. Forward parity also requires FA3's
per-row output is invariant to how many heads share the call, which is measured in
`tests/fast-gpu/test_top_attention_shape_invariance.py` (bitwise at cp 2/4/8 with num_splits pinned
to 1, including the KV-replicated case).

THE CP CEILING IS NOT `num_kv_heads`. Asserting `num_heads % cp_size == 0` here would carry that
ceiling: under GQA the KV tensors have `num_query_groups` heads, and at tp>1 the bound is
`num_query_groups / tp`, so a four-group model is capped at cp=1 whenever tp=4. Replicating the KV
heads lifts it to `num_q_heads / tp` and changes no arithmetic, which is what XoRL does
(`repeat_kv`, n_repeat = ulysses_size // num_kv_heads).
"""

from __future__ import annotations

import torch
from torch import Tensor

from megatron.core.tensor_parallel.mappings import all_to_all


def replicate_kv_heads_for_cp(x: Tensor, cp_size: int) -> Tensor:
    """Repeat KV heads so `cp_size` ranks can each own at least one.

    A no-op when the head count already divides cp_size. Repeats along the head dim so that head
    group g serves cp ranks [g*n, (g+1)*n) -- matching how FA3 maps a query head to its KV group,
    so each rank's post-a2a slice still pairs the right q heads with the right kv head.
    """
    num_heads = x.shape[-2]
    if num_heads >= cp_size:
        return x
    if cp_size % num_heads != 0:
        raise ValueError(
            f"[top] Ulysses CP needs cp_size ({cp_size}) to be a multiple of the KV head count "
            f"({num_heads}) to replicate; got a remainder. Pick a cp_size that divides or is "
            f"divisible by the KV head count."
        )
    return x.repeat_interleave(cp_size // num_heads, dim=-2)


class UlyssesCPLayout:
    """Sequence-shard <-> head-shard transforms over one CP group."""

    def __init__(self, cp_group, cp_size: int) -> None:
        self.cp_group = cp_group
        self.cp_size = cp_size

    def local_packed_lengths(self, cu_seqlens: Tensor, local_tokens: int) -> list[int]:
        """Per-packed-sequence LOCAL lengths, from the GLOBAL cu_seqlens megatron passes in."""
        global_lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        local_lengths = []
        for length in global_lengths:
            if length % self.cp_size != 0:
                raise ValueError(
                    f"[top] Ulysses CP requires padded sequence lengths divisible by cp_size; "
                    f"got length={length}, cp_size={self.cp_size}. The collator pads to "
                    f"2*cp_size for the zigzag layout -- check it ran."
                )
            local_lengths.append(length // self.cp_size)
        if sum(local_lengths) != local_tokens:
            raise ValueError(
                f"[top] packed cu_seqlens do not match the local CP shard: "
                f"sum(local_lengths)={sum(local_lengths)}, local_tokens={local_tokens}. The "
                f"cu_seqlens handed to attention must be GLOBAL, not per-rank."
            )
        return local_lengths

    def sequence_to_head_parallel(self, x: Tensor, cu_seqlens: Tensor) -> Tensor:
        """local zigzag sequence shard, all heads -> full sequence, head shard."""
        local_tokens, num_heads, head_dim = x.shape
        if num_heads % self.cp_size != 0:
            raise ValueError(
                f"[top] Ulysses CP needs num_heads ({num_heads}) divisible by cp_size "
                f"({self.cp_size}); replicate KV heads first via replicate_kv_heads_for_cp."
            )
        local_lengths = self.local_packed_lengths(cu_seqlens, local_tokens)

        x = x.reshape(local_tokens, 1, num_heads * head_dim)
        hidden_per_rank = x.shape[-1] // self.cp_size
        rank_ordered = torch.cat(
            torch.split(x.reshape(local_tokens, -1), hidden_per_rank, dim=1), dim=0
        )
        rank_ordered = all_to_all(self.cp_group, rank_ordered)
        rank_ordered = rank_ordered.reshape(local_tokens * self.cp_size, 1, hidden_per_rank)

        # Undo the zigzag: front-half chunks in rank order, back-half in REVERSE rank order, so the
        # result is in natural token order. FA3 then sees an ordinary full-length sequence.
        sequential = []
        for seq_index, local_length in enumerate(local_lengths):
            if local_length % 2 != 0:
                raise ValueError(
                    f"[top] Ulysses CP expects two equal zigzag chunks per rank; got "
                    f"local_length={local_length}."
                )
            chunk = local_length // 2
            seq_offset = sum(local_lengths[:seq_index])
            for source_rank in range(self.cp_size):
                start = source_rank * local_tokens + seq_offset
                sequential.append(rank_ordered[start : start + chunk])
            for source_rank in range(self.cp_size - 1, -1, -1):
                start = source_rank * local_tokens + seq_offset + chunk
                sequential.append(rank_ordered[start : start + chunk])

        x = torch.cat(sequential, dim=0)
        return x.view(local_tokens * self.cp_size, num_heads // self.cp_size, head_dim)

    def head_to_sequence_parallel(
        self, x: Tensor, cu_seqlens: Tensor, local_tokens: int, num_heads: int
    ) -> Tensor:
        """full sequence, head shard -> local zigzag sequence shard, all heads."""
        global_tokens, heads_per_cp_rank, head_dim = x.shape
        if global_tokens != local_tokens * self.cp_size:
            raise ValueError(
                f"[top] unexpected Ulysses global token count: {global_tokens} vs "
                f"{local_tokens * self.cp_size}"
            )
        if heads_per_cp_rank * self.cp_size != num_heads:
            raise ValueError(
                f"[top] unexpected Ulysses head shard: {heads_per_cp_rank} * {self.cp_size} "
                f"!= {num_heads}"
            )
        local_lengths = self.local_packed_lengths(cu_seqlens, local_tokens)

        x = x.reshape(global_tokens, 1, heads_per_cp_rank * head_dim)
        rank_ordered: list[list[Tensor]] = [[] for _ in range(self.cp_size)]
        seq_start = 0
        for local_length in local_lengths:
            chunk = local_length // 2
            chunks = torch.split(x[seq_start : seq_start + local_length * self.cp_size], chunk, dim=0)
            assert len(chunks) == 2 * self.cp_size
            for rank in range(self.cp_size):
                rank_ordered[rank].append(chunks[rank])
                rank_ordered[rank].append(chunks[2 * self.cp_size - rank - 1])
            seq_start += local_length * self.cp_size

        stacked = torch.cat([torch.cat(parts, dim=0) for parts in rank_ordered], dim=0)
        stacked = all_to_all(self.cp_group, stacked.reshape(global_tokens, -1))
        output = torch.cat(torch.split(stacked, local_tokens, dim=0), dim=-1)
        return output.view(local_tokens, num_heads, head_dim)
