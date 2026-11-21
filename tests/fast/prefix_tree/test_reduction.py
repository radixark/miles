import pytest
import torch

from miles.backends.training_utils.prefix_tree.reduction import OrderedRowGather


def test_ordered_gather_fp64_gradient_and_empty_rows():
    plan = OrderedRowGather.from_indices([2, 0, 2, 0, 2], 4, device="cpu")
    values = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(plan.gather, (values,))
    plan.gather(values).sum().backward()
    torch.testing.assert_close(values.grad[:, 0], torch.tensor([2, 0, 3, 0], dtype=torch.float64))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fixed_order_accumulates_before_casting_back(dtype):
    plan = OrderedRowGather.from_indices([0] * 32, 1, device="cpu")
    value = torch.zeros(1, 2, dtype=dtype, requires_grad=True)
    coefficients = torch.tensor([[256.0, 2.0]] + [[1.0, 0.01]] * 30 + [[-256.0, -2.0]], dtype=dtype)
    plan.gather(value).backward(coefficients)
    torch.testing.assert_close(value.grad, coefficients.float().sum(dim=0, keepdim=True).to(dtype), rtol=0, atol=0)
