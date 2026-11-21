import pytest
import torch
import torch.nn.functional as F

from miles.backends.training_utils.prefix_tree.attention import TreeAttentionLayout, tree_attention
from miles.backends.training_utils.prefix_tree.plan import build_prefix_tree


@pytest.mark.parametrize("deterministic_reduction", [False, True])
def test_float64_outputs_and_every_qkv_gradient_against_independent_expanded_attention(deterministic_reduction):
    torch.manual_seed(41)
    sequences = [[1, 2, 3, 4, 5], [1, 2, 3, 9], [1, 2], [1, 2, 3, 4, 5], [8, 7]]
    plan = build_prefix_tree(sequences)
    actual = [torch.randn(len(plan.tokens), h, 8, dtype=torch.float64, requires_grad=True) for h in [4, 2, 2]]
    expected = [tensor.detach().clone().requires_grad_() for tensor in actual]
    output = tree_attention(
        *actual,
        TreeAttentionLayout.from_plan(plan, "cpu", deterministic_reduction=deterministic_reduction),
        scaling=8**-0.5,
    )
    actual_terms, expected_terms = [], []
    for nodes in plan.sample_nodes:
        q, k, v = [tensor[list(nodes)].transpose(0, 1) for tensor in expected]
        expanded = F.scaled_dot_product_attention(
            q,
            k.repeat_interleave(2, 0),
            v.repeat_interleave(2, 0),
            is_causal=True,
        ).transpose(0, 1)
        torch.testing.assert_close(output[list(nodes)], expanded, atol=1e-12, rtol=1e-12)
        coefficients = torch.randn_like(expanded)
        actual_terms.append((output[list(nodes)] * coefficients).sum())
        expected_terms.append((expanded * coefficients).sum())
    sum(actual_terms).backward()
    sum(expected_terms).backward()
    for x, y in zip(actual, expected, strict=True):
        torch.testing.assert_close(x.grad, y.grad, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("enabled", [False, True])
def test_global_determinism_requirement_reaches_custom_attention(enabled):
    previous = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    observed = {}
    plan = build_prefix_tree([[1, 2], [1, 3]])
    layout = TreeAttentionLayout.from_plan(plan, "cpu")
    values = torch.zeros(len(plan.tokens), 2, 4)

    def kernel(q, k, v, **kwargs):
        observed.update(kwargs)
        return q

    try:
        torch.use_deterministic_algorithms(enabled)
        tree_attention(values, values, values, layout, scaling=0.5, varlen_func=kernel)
        assert observed.get("deterministic", False) == enabled
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=warn_only)
