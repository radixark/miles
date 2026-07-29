import pytest
import torch

from miles_plugins.models.deepseek_v4.ops import utils as dsv4_utils


def _manual_fixed_tree_mean_last_dim(x):
    width = x.shape[-1]
    reduced = x
    while reduced.shape[-1] > 1:
        reduced = reduced.reshape(
            *reduced.shape[:-1],
            reduced.shape[-1] // 2,
            2,
        )
        reduced = reduced[..., 0] + reduced[..., 1]
    return reduced / width


def _manual_query_rms_norm(x, eps):
    x_fp32 = x.float()
    square_mean = _manual_fixed_tree_mean_last_dim(x_fp32.square())
    return (x_fp32 * torch.rsqrt(square_mean + eps)).to(x.dtype)


def test_fixed_tree_mean_has_explicit_association_and_keeps_dim():
    x = torch.tensor(
        [[1.0e20, 1.0, -1.0e20, 1.0], [1.0, 2.0, 3.0, 4.0]],
        dtype=torch.float32,
    )

    actual = dsv4_utils.fixed_tree_mean_last_dim(x)
    expected = ((x[..., 0] + x[..., 1]) + (x[..., 2] + x[..., 3])) / 4

    assert actual.shape == (2, 1)
    assert torch.equal(actual[..., 0], expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_dsv4_query_rms_uses_fixed_tree_and_is_batch_invariant(
    dtype,
    monkeypatch,
):
    torch.manual_seed(1234)
    full = torch.randn(
        3,
        17,
        4,
        512,
        dtype=torch.float32,
    ).to(dtype)
    single = full[:, :1].detach().clone().requires_grad_()
    full = full.detach().requires_grad_()
    reference_input = single.detach().clone().requires_grad_()

    fixed_tree_calls = 0
    fixed_tree_mean = dsv4_utils.fixed_tree_mean_last_dim

    def traced_fixed_tree_mean(x):
        nonlocal fixed_tree_calls
        fixed_tree_calls += 1
        return fixed_tree_mean(x)

    monkeypatch.setattr(
        dsv4_utils,
        "fixed_tree_mean_last_dim",
        traced_fixed_tree_mean,
    )

    for _ in range(5):
        single_output = dsv4_utils.dsv4_query_rms_norm(
            single,
            eps=1.0e-6,
            batch_invariant=True,
        )
        full_output = dsv4_utils.dsv4_query_rms_norm(
            full,
            eps=1.0e-6,
            batch_invariant=True,
        )[:, :1]
        assert torch.equal(single_output, full_output)

    reference_output = _manual_query_rms_norm(reference_input, eps=1.0e-6)
    assert fixed_tree_calls == 10
    assert single_output.dtype == dtype
    assert torch.equal(single_output, reference_output)

    upstream = torch.randn_like(single_output)
    single_output.backward(upstream)
    full_output.backward(upstream)
    reference_output.backward(upstream)
    assert torch.equal(single.grad, full.grad[:, :1])
    assert torch.equal(single.grad, reference_input.grad)


def test_dsv4_query_rms_default_matches_legacy_formula():
    torch.manual_seed(5678)
    x = torch.randn(
        2,
        3,
        4,
        512,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    reference_input = x.detach().clone().requires_grad_()
    reference_fp32 = reference_input.float()
    expected = (
        reference_fp32
        * torch.rsqrt(
            reference_fp32.square().mean(-1, keepdim=True) + 1.0e-6
        )
    ).to(reference_input.dtype)

    actual = dsv4_utils.dsv4_query_rms_norm(
        x,
        eps=1.0e-6,
        batch_invariant=False,
    )

    assert actual.dtype == x.dtype
    assert torch.equal(actual, expected)

    upstream = torch.randn_like(actual)
    actual.backward(upstream)
    expected.backward(upstream)
    assert torch.equal(x.grad, reference_input.grad)


@pytest.mark.parametrize("width", [0, 3, 6])
def test_fixed_tree_mean_rejects_unsupported_width(width):
    with pytest.raises(RuntimeError, match="nonzero power-of-two"):
        dsv4_utils.fixed_tree_mean_last_dim(torch.zeros(2, width))


def test_fixed_tree_mean_preserves_autograd():
    x = torch.randn(2, 3, 8, dtype=torch.float64, requires_grad=True)

    assert torch.autograd.gradcheck(
        dsv4_utils.fixed_tree_mean_last_dim,
        (x,),
    )


def test_fixed_tree_mean_rejects_scalar_input():
    with pytest.raises(RuntimeError, match="at least one dimension"):
        dsv4_utils.fixed_tree_mean_last_dim(torch.tensor(1.0))


def test_fixed_tree_mean_rejects_non_floating_input():
    with pytest.raises(RuntimeError, match="floating-point tensor"):
        dsv4_utils.fixed_tree_mean_last_dim(
            torch.ones(2, 4, dtype=torch.int64)
        )
