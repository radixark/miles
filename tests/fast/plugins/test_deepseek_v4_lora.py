from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="stage-a-cpu", labels=[])

from types import SimpleNamespace

import pytest
import torch

from miles_plugins.models.deepseek_v4.lora import apply_grouped_wo_a


class _GroupedLoRA:
    _adapter_enabled = True
    adapter = object()

    def __init__(self, base_weight, lora_a, lora_b):
        self.to_wrap = SimpleNamespace(weight=base_weight)
        self.lora_a = lora_a
        self.lora_b = lora_b

    def adapter_forward(self, adapter, input_):
        assert adapter is self.adapter
        return (input_ @ self.lora_a.T) @ self.lora_b.T


def test_grouped_wo_a_uses_only_matching_lora_group():
    torch.manual_seed(0)
    batch, sequence, groups = 2, 3, 4
    input_dim, output_dim, adapter_rank = 5, 2, 3
    input_ = torch.randn(batch, sequence, groups, input_dim)
    base_weight = torch.randn(groups * output_dim, input_dim)
    lora_a = torch.randn(adapter_rank, input_dim, requires_grad=True)
    lora_b = torch.randn(groups * output_dim, adapter_rank, requires_grad=True)
    module = _GroupedLoRA(base_weight, lora_a, lora_b)

    output = apply_grouped_wo_a(
        module,
        input_,
        num_groups=groups,
        group_output_dim=output_dim,
    )

    effective_weight = base_weight.view(groups, output_dim, input_dim) + (lora_b @ lora_a).view(
        groups, output_dim, input_dim
    )
    expected = torch.einsum("...gd,grd->...gr", input_, effective_weight)
    torch.testing.assert_close(output, expected)

    output.square().sum().backward()
    assert lora_a.grad is not None
    assert lora_b.grad is not None


def test_grouped_wo_a_uses_base_projection_when_adapter_is_disabled():
    groups, input_dim, output_dim = 2, 3, 4
    input_ = torch.randn(5, groups, input_dim)
    base_weight = torch.randn(groups * output_dim, input_dim)
    module = _GroupedLoRA(
        base_weight,
        torch.randn(2, input_dim),
        torch.randn(groups * output_dim, 2),
    )
    module._adapter_enabled = False

    output = apply_grouped_wo_a(
        module,
        input_,
        num_groups=groups,
        group_output_dim=output_dim,
    )

    expected = torch.einsum(
        "...gd,grd->...gr",
        input_,
        base_weight.view(groups, output_dim, input_dim),
    )
    torch.testing.assert_close(output, expected)


def test_grouped_wo_a_rejects_incompatible_adapter_output():
    groups, input_dim, output_dim = 2, 3, 4
    input_ = torch.randn(5, groups, input_dim)
    module = _GroupedLoRA(
        torch.randn(groups * output_dim, input_dim),
        torch.randn(2, input_dim),
        # One output row short: a standard projection could silently broadcast
        # this, but the grouped diagonal would no longer describe wo_a.
        torch.randn(groups * output_dim - 1, 2),
    )

    with pytest.raises(RuntimeError, match="LoRA output mismatch"):
        apply_grouped_wo_a(
            module,
            input_,
            num_groups=groups,
            group_output_dim=output_dim,
        )
