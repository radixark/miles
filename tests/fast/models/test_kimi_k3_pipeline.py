import pytest
import torch

from miles_plugins.models.kimi_k3.pipeline import bank_num_rows, pack_stage_boundary, unpack_stage_boundary


def test_pack_unpack_is_exact_in_both_directions() -> None:
    """Megatron's p2p carries one hidden-states tensor, so the attention-residual
    bank rides across the stage boundary flattened into the hidden dim. Values
    have to survive the round trip and gradients have to route back to the right
    side of the split -- a transposed unflatten would mix bank rows into
    prefix_sum with no shape error. The width assertion is the only thing
    standing between a sender/receiver row-count disagreement and silently
    reinterpreted activations.
    """
    torch.manual_seed(0)
    prefix_sum = torch.randn(5, 2, 16, dtype=torch.bfloat16)
    block_residual = torch.randn(5, 2, 3, 16, dtype=torch.bfloat16)

    packed = pack_stage_boundary(prefix_sum, block_residual)
    assert packed.shape == (5, 2, 4 * 16)

    prefix_out, bank_out = unpack_stage_boundary(packed, 16, 3)
    torch.testing.assert_close(prefix_out, prefix_sum, rtol=0, atol=0)
    torch.testing.assert_close(bank_out, block_residual, rtol=0, atol=0)

    grad_prefix_sum = torch.randn(4, 1, 8, requires_grad=True)
    grad_block_residual = torch.randn(4, 1, 2, 8, requires_grad=True)
    prefix_out, bank_out = unpack_stage_boundary(pack_stage_boundary(grad_prefix_sum, grad_block_residual), 8, 2)
    grad_prefix = torch.randn_like(prefix_out)
    grad_bank = torch.randn_like(bank_out)
    torch.autograd.backward([prefix_out, bank_out], [grad_prefix, grad_bank])

    torch.testing.assert_close(grad_prefix_sum.grad, grad_prefix, rtol=0, atol=0)
    torch.testing.assert_close(grad_block_residual.grad, grad_bank, rtol=0, atol=0)

    with pytest.raises(AssertionError, match="stage-boundary payload width"):
        unpack_stage_boundary(torch.zeros(2, 1, 3 * 16), 16, 3)


def test_bank_num_rows_matches_write_schedule() -> None:
    """The receiver sizes its unpack from ``bank_num_rows`` while the sender's
    row count comes from the write schedule (``layer_idx % block_size == 0``).
    The ceil-div has to agree with that schedule at every one of the 93 layers,
    from both sides of a stage boundary -- one row off and the payload is
    reinterpreted, not rejected.
    """
    block_size = 12
    for layer_idx in range(1, 93):
        rows_written_before = sum(1 for w in range(layer_idx) if w % block_size == 0)
        assert bank_num_rows(layer_idx, block_size) == rows_written_before

    for last_layer_of_stage in range(92):
        rows_after_exit = last_layer_of_stage // block_size + 1
        assert bank_num_rows(last_layer_of_stage + 1, block_size) == rows_after_exit
