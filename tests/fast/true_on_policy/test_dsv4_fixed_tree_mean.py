import pytest
import torch

from miles_plugins.models.deepseek_v4.ops.utils import fixed_tree_mean_last_dim


def test_fixed_tree_mean_has_explicit_association_and_keeps_dim():
    x = torch.tensor(
        [[1.0e20, 1.0, -1.0e20, 1.0], [1.0, 2.0, 3.0, 4.0]],
        dtype=torch.float32,
    )

    actual = fixed_tree_mean_last_dim(x)
    expected = ((x[..., 0] + x[..., 1]) + (x[..., 2] + x[..., 3])) / 4

    assert actual.shape == (2, 1)
    assert torch.equal(actual[..., 0], expected)


def test_fixed_tree_mean_is_outer_shape_invariant():
    torch.manual_seed(1234)
    compact = torch.randn(3, 5, 512, dtype=torch.float32)
    padded = torch.zeros(3, 17, 512, dtype=torch.float32)
    padded[:, :5].copy_(compact)

    compact_mean = fixed_tree_mean_last_dim(compact)
    padded_mean = fixed_tree_mean_last_dim(padded)[:, :5]

    assert torch.equal(compact_mean, padded_mean)


@pytest.mark.parametrize("width", [0, 3, 6])
def test_fixed_tree_mean_rejects_unsupported_width(width):
    with pytest.raises(RuntimeError, match="nonzero power-of-two"):
        fixed_tree_mean_last_dim(torch.zeros(2, width))


def test_fixed_tree_mean_preserves_autograd():
    x = torch.randn(2, 3, 8, dtype=torch.float64, requires_grad=True)

    assert torch.autograd.gradcheck(fixed_tree_mean_last_dim, (x,))


def test_fixed_tree_mean_rejects_scalar_input():
    with pytest.raises(RuntimeError, match="at least one dimension"):
        fixed_tree_mean_last_dim(torch.tensor(1.0))


def test_fixed_tree_mean_rejects_non_floating_input():
    with pytest.raises(RuntimeError, match="floating-point tensor"):
        fixed_tree_mean_last_dim(torch.ones(2, 4, dtype=torch.int64))
