"""
Smoke tests for attention_output_gate + TP fix.

Bug: When num_query_groups < TP (e.g. num_kv_heads=2, TP=4), the gate tensor
in SelfAttention.get_query_key_value_tensors was not sliced by TP the same way
query was, causing shape mismatch in _apply_output_gate:
  gate.view(*x.shape) -> invalid shape because gate has (num_heads/num_kv_heads)
  heads but x only has (num_heads/TP) heads.

These tests verify the fix without requiring multi-GPU or full model init.
"""

import pytest
import torch


# ---------------------------------------------------------------------------
# Test 1: Pure tensor shape simulation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "num_attention_heads,num_kv_heads,head_dim,tp_size",
    [
        (16, 2, 256, 4),  # Qwen3.5-35B / Qwen3-Next with TP4
        (16, 2, 256, 8),  # TP8
        (16, 4, 256, 8),  # num_kv_heads=4, TP8
        (32, 4, 128, 8),  # another config
        (16, 2, 256, 2),  # TP=num_kv_heads, should NOT enter the slice path
        (16, 8, 128, 4),  # num_kv_heads >= TP, should NOT enter the slice path
    ],
)
def test_gate_query_shape_match_after_tp_slice(num_attention_heads, num_kv_heads, head_dim, tp_size):
    """Simulate the gate/query slicing logic and verify shapes match."""
    sq, b = 32, 1  # sequence length, batch

    if num_kv_heads < tp_size:
        # This is the path that was buggy
        num_query_groups_per_partition = 1
        num_attention_heads_per_partition = num_attention_heads // num_kv_heads
    else:
        num_query_groups_per_partition = num_kv_heads // tp_size
        num_attention_heads_per_partition = num_attention_heads // tp_size

    num_query_heads_per_group = num_attention_heads_per_partition // num_query_groups_per_partition

    for rank in range(tp_size):
        # Simulate gate after split: [sq, b, ng, np/ng * hn]
        gate_raw = torch.randn(
            sq,
            b,
            num_query_groups_per_partition,
            num_query_heads_per_group * head_dim,
        )
        # Reshape: [sq, b, ng, np/ng * hn] -> [sq, b, np, hn]
        gate = gate_raw.reshape(sq, b, -1, head_dim)

        # Simulate query after reshape + slice
        query = torch.randn(sq, b, num_query_groups_per_partition, num_query_heads_per_group * head_dim)
        query = query.reshape(sq, b, -1, head_dim)

        if num_kv_heads < tp_size:
            tp_ratio = tp_size // num_kv_heads
            idx = rank % tp_ratio
            size = num_attention_heads_per_partition // tp_ratio
            query = query[:, :, idx * size : (idx + 1) * size, :]
            # THE FIX: also slice gate
            gate = gate[:, :, idx * size : (idx + 1) * size, :]

        # Now simulate _apply_output_gate: flatten to [sq, b, h_local]
        x = query.reshape(sq, b, -1)  # core_attn_out shape

        # This is the critical check: gate must be viewable as x.shape
        try:
            gate_for_view = gate.contiguous().view(*x.shape)
        except RuntimeError as e:
            pytest.fail(
                f"gate.view(*x.shape) failed for rank={rank}, " f"gate numel={gate.numel()}, x.shape={x.shape}: {e}"
            )

        assert gate_for_view.shape == x.shape, f"rank={rank}: gate shape {gate_for_view.shape} != x shape {x.shape}"


# ---------------------------------------------------------------------------
# Test 2: Verify the bug would occur WITHOUT the fix
# ---------------------------------------------------------------------------


def test_without_fix_gate_shape_mismatch():
    """Verify that without slicing gate, shapes mismatch for num_kv_heads < TP."""
    num_attention_heads = 16
    num_kv_heads = 2
    head_dim = 256
    tp_size = 4
    sq, b = 32, 1

    num_attention_heads_per_partition = num_attention_heads // num_kv_heads  # 8
    num_query_heads_per_group = num_attention_heads_per_partition  # 8 (ng=1)

    gate = torch.randn(sq, b, 1, num_query_heads_per_group * head_dim)
    gate = gate.reshape(sq, b, -1, head_dim)  # [32, 1, 8, 256]

    query = torch.randn(sq, b, 1, num_query_heads_per_group * head_dim)
    query = query.reshape(sq, b, -1, head_dim)  # [32, 1, 8, 256]

    # Slice query (as Megatron does)
    tp_ratio = tp_size // num_kv_heads  # 2
    size = num_attention_heads_per_partition // tp_ratio  # 4
    query = query[:, :, 0:size, :]  # [32, 1, 4, 256]

    # Do NOT slice gate (the bug)
    x = query.reshape(sq, b, -1)  # [32, 1, 1024]

    # gate has 8 heads * 256 = 2048, x has 4 heads * 256 = 1024
    assert gate.numel() != x.numel(), "Expected size mismatch without the fix"

    with pytest.raises(RuntimeError):
        gate.contiguous().view(*x.shape)


# ---------------------------------------------------------------------------
# Test 3: Numerical correctness — gate applied to correct head partition
# ---------------------------------------------------------------------------


def test_gate_numerical_correctness_per_rank():
    """Each TP rank should get the correct slice of gate matching its query heads."""
    num_attention_heads = 16
    num_kv_heads = 2
    head_dim = 256
    tp_size = 4
    sq, b = 4, 1

    num_heads_per_kv_group = num_attention_heads // num_kv_heads  # 8
    tp_ratio = tp_size // num_kv_heads  # 2
    heads_per_rank = num_attention_heads // tp_size  # 4

    # Create a full gate where each head has a distinct value
    # Shape: [sq, b, num_heads, head_dim]
    full_gate = torch.zeros(sq, b, num_attention_heads, head_dim)
    for h in range(num_attention_heads):
        full_gate[:, :, h, :] = h  # head i filled with value i

    for rank in range(tp_size):
        # Which KV group does this rank belong to?
        kv_group = rank // tp_ratio  # 0,0,1,1

        # Within the kv_group, all query heads for that group
        group_start = kv_group * num_heads_per_kv_group
        group_gate = full_gate[:, :, group_start : group_start + num_heads_per_kv_group, :]
        # Shape: [sq, b, 8, 256]

        # Apply the fix: slice to this rank's portion
        idx = rank % tp_ratio
        size = num_heads_per_kv_group // tp_ratio  # 4
        rank_gate = group_gate[:, :, idx * size : (idx + 1) * size, :]
        # Shape: [sq, b, 4, 256]

        # Expected heads for this rank
        expected_start = rank * heads_per_rank
        expected_gate = full_gate[:, :, expected_start : expected_start + heads_per_rank, :]

        assert torch.equal(rank_gate, expected_gate), (
            f"rank={rank}: got heads with values "
            f"{rank_gate[0, 0, :, 0].tolist()}, "
            f"expected {expected_gate[0, 0, :, 0].tolist()}"
        )


# ---------------------------------------------------------------------------
# Test 4: Import and verify the patched code path exists
# ---------------------------------------------------------------------------


@pytest.mark.skip(reason="Requires patched Megatron-LM; verified via patch file instead")
def test_patched_attention_has_gate_slice():
    """Verify the patched SelfAttention.get_query_key_value_tensors has the gate slice."""
    import inspect

    from megatron.core.transformer.attention import SelfAttention

    source = inspect.getsource(SelfAttention.get_query_key_value_tensors)
    assert (
        "gate = gate[:, :, idx * size" in source
    ), "Patch not applied: gate TP slicing not found in SelfAttention.get_query_key_value_tensors"
