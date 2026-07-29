"""Stage-boundary packing for pipeline-parallel Kimi K3.

The attention-residual state (prefix_sum plus the snapshot bank) must cross
pipeline stage boundaries, but Megatron's p2p carries a single hidden-states
tensor. The stage-exit layer packs [prefix_sum, bank] along the hidden
dimension into one 3-D tensor and the stage-entry layer unpacks it. Miles
always runs the Megatron backend with variable_seq_lengths, so the receiver
allocates its buffer from the sender's actual shape and any row-count
disagreement fails loudly at unpack instead of corrupting silently.
"""

import torch


def bank_num_rows(layer_idx: int, block_size: int) -> int:
    """Snapshot rows present before global layer ``layer_idx`` executes."""
    assert layer_idx > 0
    return (layer_idx + block_size - 1) // block_size


def pack_stage_boundary(prefix_sum: torch.Tensor, block_residual: torch.Tensor) -> torch.Tensor:
    assert block_residual.shape[-2] > 0, "stage boundary before the first snapshot write"
    return torch.cat((prefix_sum, block_residual.flatten(-2)), dim=-1)


def unpack_stage_boundary(packed: torch.Tensor, hidden_size: int, num_rows: int) -> tuple[torch.Tensor, torch.Tensor]:
    expected = (1 + num_rows) * hidden_size
    assert (
        packed.shape[-1] == expected
    ), f"stage-boundary payload width {packed.shape[-1]} != (1 + {num_rows}) * {hidden_size}"
    prefix_sum = packed[..., :hidden_size].contiguous()
    block_residual = packed[..., hidden_size:].unflatten(-1, (num_rows, hidden_size)).contiguous()
    return prefix_sum, block_residual
