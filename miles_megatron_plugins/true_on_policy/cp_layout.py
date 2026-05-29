from __future__ import annotations

import torch
from megatron.core.tensor_parallel.mappings import all_to_all
from torch import Tensor


class SGLangUlyssesCPLayout:
    """Ulysses CP transforms between Megatron local sequence shards and FA3 head shards."""

    def __init__(self, cp_group, cp_size: int) -> None:
        self.cp_group = cp_group
        self.cp_size = cp_size

    def local_packed_lengths(self, cu_seqlens: Tensor, local_tokens: int) -> list[int]:
        """Return per-packed-sequence local CP lengths from global cu_seqlens."""
        global_lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        local_lengths = []
        for length in global_lengths:
            assert length % self.cp_size == 0, (
                f"Ulysses CP requires padded sequence lengths divisible by cp_size; "
                f"got length={length}, cp_size={self.cp_size}"
            )
            local_lengths.append(length // self.cp_size)

        assert sum(local_lengths) == local_tokens, (
            f"Packed cu_seqlens do not match local CP shard length: "
            f"sum(local_lengths)={sum(local_lengths)}, local_tokens={local_tokens}"
        )
        return local_lengths

    def sequence_to_head_parallel(self, x: Tensor, cu_seqlens: Tensor) -> Tensor:
        """Ulysses CP all-to-all: local zigzag sequence shard -> full sequence, head shard."""
        local_tokens, num_heads, head_dim = x.shape
        assert num_heads % self.cp_size == 0, f"num_heads={num_heads} must be divisible by cp_size={self.cp_size}"
        local_lengths = self.local_packed_lengths(cu_seqlens, local_tokens)

        x = x.reshape(local_tokens, 1, num_heads * head_dim)
        hidden_per_rank = x.shape[-1] // self.cp_size
        rank_ordered = torch.cat(torch.split(x.reshape(local_tokens, -1), hidden_per_rank, dim=1), dim=0)
        rank_ordered = all_to_all(self.cp_group, rank_ordered)
        rank_ordered = rank_ordered.reshape(local_tokens * self.cp_size, 1, hidden_per_rank)

        per_source_offsets = []
        offset = 0
        for _ in range(self.cp_size):
            per_source_offsets.append(offset)
            offset += local_tokens

        sequential = []
        for seq_index, local_length in enumerate(local_lengths):
            assert (
                local_length % 2 == 0
            ), f"Ulysses CP expects two equal zigzag chunks per rank; got local_length={local_length}"
            chunk = local_length // 2
            seq_offset = sum(local_lengths[:seq_index])

            for source_rank in range(self.cp_size):
                start = per_source_offsets[source_rank] + seq_offset
                sequential.append(rank_ordered[start : start + chunk])
            for source_rank in range(self.cp_size - 1, -1, -1):
                start = per_source_offsets[source_rank] + seq_offset + chunk
                sequential.append(rank_ordered[start : start + chunk])

        x = torch.cat(sequential, dim=0)
        return x.view(local_tokens * self.cp_size, num_heads // self.cp_size, head_dim)

    def head_to_sequence_parallel(self, x: Tensor, cu_seqlens: Tensor, local_tokens: int, num_heads: int) -> Tensor:
        """Ulysses CP inverse all-to-all: full sequence, head shard -> local zigzag sequence shard."""
        global_tokens, heads_per_cp_rank, head_dim = x.shape
        assert (
            global_tokens == local_tokens * self.cp_size
        ), f"Unexpected Ulysses global token count: {global_tokens} vs {local_tokens * self.cp_size}"
        assert (
            heads_per_cp_rank * self.cp_size == num_heads
        ), f"Unexpected Ulysses head shard: {heads_per_cp_rank} * {self.cp_size} != {num_heads}"
        local_lengths = self.local_packed_lengths(cu_seqlens, local_tokens)

        x = x.reshape(global_tokens, 1, heads_per_cp_rank * head_dim)
        rank_ordered = [[] for _ in range(self.cp_size)]
        seq_start = 0
        for local_length in local_lengths:
            chunk = local_length // 2
            chunks = torch.split(x[seq_start : seq_start + local_length * self.cp_size], chunk, dim=0)
            assert len(chunks) == 2 * self.cp_size
            for rank in range(self.cp_size):
                rank_ordered[rank].append(chunks[rank])
                rank_ordered[rank].append(chunks[2 * self.cp_size - rank - 1])
            seq_start += local_length * self.cp_size

        rank_ordered = torch.cat([torch.cat(parts, dim=0) for parts in rank_ordered], dim=0)
        rank_ordered = all_to_all(self.cp_group, rank_ordered.reshape(global_tokens, -1))
        output = torch.cat(torch.split(rank_ordered, local_tokens, dim=0), dim=-1)
        return output.view(local_tokens, num_heads, head_dim)
