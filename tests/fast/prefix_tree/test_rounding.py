"""A deterministic counterexample: merging BF16 adjoints moves a rounding point."""

import pytest
import torch
import torch.nn.functional as F

from miles.backends.training_utils.prefix_tree.reduction import OrderedRowGather


@pytest.fixture(autouse=True)
def _deterministic_algorithms():
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


@pytest.mark.parametrize("ordered", [False, True])
def test_bf16_gradient_difference_can_exist_without_nondeterminism(ordered):
    source = torch.tensor([[1.5]])
    coefficients = torch.tensor([[1.0], [1.0 / 256]])
    reference = torch.zeros(1, 1, requires_grad=True)
    compact = torch.zeros(1, 1, requires_grad=True)
    gather = OrderedRowGather.from_indices([0, 0], 1, device="cpu")
    with torch.autocast("cpu", dtype=torch.bfloat16):
        expanded_output = F.linear(source.repeat(2, 1), reference)
        unique_output = F.linear(source, compact)
        compact_output = gather.gather(unique_output) if ordered else unique_output[[0, 0]]
    (expanded_output.float() * coefficients).sum().backward()
    (compact_output.float() * coefficients).sum().backward()
    # Expanded GEMM rounds 1.5 * (1 + 1/256) only at the end. Shared GEMM
    # receives round_bf16(1 + 1/256) == 1, losing the small contribution first.
    assert reference.grad.item() == 1.5078125
    assert compact.grad.item() == 1.5


def test_the_same_shared_linear_gradient_is_exact_in_fp32():
    source = torch.tensor([[1.5]])
    coefficients = torch.tensor([[1.0], [1.0 / 256]])
    reference = torch.zeros(1, 1, requires_grad=True)
    compact = torch.zeros(1, 1, requires_grad=True)
    (F.linear(source.repeat(2, 1), reference) * coefficients).sum().backward()
    (F.linear(source, compact)[[0, 0]] * coefficients).sum().backward()
    assert reference.grad.item() == compact.grad.item() == 1.505859375
