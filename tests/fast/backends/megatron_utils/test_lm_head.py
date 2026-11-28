"""``--log-probs-backend fused_lm_head`` switches the output layer only when asked, and only on models with the hook."""

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

import miles.backends.megatron_utils.lm_head as lm_head_module
from miles.backends.megatron_utils.lm_head import (
    LMHeadContext,
    check_model_takes_output_processor,
    hidden_states_output_processor,
    lm_head_forward_kwargs,
)


class _ModelWithHook(torch.nn.Module):
    def forward(self, input_ids, *, output_processor=None, output_processor_context=None):
        raise NotImplementedError


class _OutputLayer(torch.nn.Module):  # ColumnParallelLinear's call signature, on one rank
    sequence_parallel = False

    def __init__(self, hidden, vocab):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(vocab, hidden))

    def forward(self, x, weight=None, runtime_gather_output=None):
        return x @ (self.weight if weight is None else weight).t(), None


class _ModelWithoutHook(torch.nn.Module):  # stands for HybridModel, whose forward has no output_processor
    def forward(self, input_ids):
        raise NotImplementedError


class _ModelWithDelegatedHook(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.language_model = _ModelWithHook()

    def forward(self, input_ids, **kwargs):
        return self.language_model(input_ids, **kwargs)


def _model(role):
    chunk = _ModelWithHook()
    chunk.role = role
    return [chunk]


@pytest.mark.parametrize(
    ("backend", "role"),
    [("fused", "actor"), ("torch", "actor"), ("fused_lm_head", "critic")],
    ids=["fused", "torch", "critic"],
)
def test_the_model_keeps_its_output_unless_fused_lm_head_on_the_actor(backend, role):
    assert lm_head_forward_kwargs(SimpleNamespace(log_probs_backend=backend), _model(role)) == ({}, None)


def test_other_backends_never_read_the_role():
    assert lm_head_forward_kwargs(SimpleNamespace(log_probs_backend="fused"), [_ModelWithHook()]) == ({}, None)


def test_fused_lm_head_on_the_actor_passes_the_processor_and_its_context():
    kwargs, context = lm_head_forward_kwargs(SimpleNamespace(log_probs_backend="fused_lm_head"), _model("actor"))
    assert kwargs["output_processor"] is hidden_states_output_processor
    assert kwargs["output_processor_context"] is context
    assert isinstance(context, LMHeadContext) and context.output_layer is None


@pytest.mark.parametrize("scale", [None, 0.5], ids=["megatron_without_mup", "mup"])
def test_project_scales_the_logits_as_the_model_would(scale):
    layer, context = _OutputLayer(hidden=8, vocab=16), LMHeadContext()
    hidden = hidden_states_output_processor(
        hidden_states=torch.randn(4, 1, 8),
        output_layer=layer,
        output_weight=None,
        context=context,
        **({} if scale is None else {"scale_logits": lambda logits: logits * scale}),
    )
    rows = hidden.squeeze(0)
    logits = rows @ layer.weight.t()
    torch.testing.assert_close(
        context.project(rows, torch.arange(rows.size(0))), logits if scale is None else logits * scale
    )


def test_sequence_parallel_gather_owns_head_gradient_reduction(monkeypatch):
    layer, context = _OutputLayer(hidden=4, vocab=7), LMHeadContext()
    layer.sequence_parallel = True
    layer.tp_group = object()

    def gather(tensor, tensor_parallel_output_grad, group):
        assert tensor_parallel_output_grad
        assert group is layer.tp_group
        return tensor

    monkeypatch.setattr(lm_head_module, "gather_from_sequence_parallel_region", gather)
    hidden = hidden_states_output_processor(
        hidden_states=torch.randn(5, 1, 4), output_layer=layer, output_weight=None, context=context
    ).squeeze(0)
    torch.testing.assert_close(context.project(hidden, torch.arange(5)), hidden @ layer.weight.t())
    assert layer.sequence_parallel


def test_multi_lora_output_projects_each_tile_with_its_own_adapter(monkeypatch):
    class RoutedOutputLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.to_wrap = _OutputLayer(hidden=4, vocab=7)
            self.adapter_weights = torch.nn.Parameter(torch.randn(2, 7, 4))
            self.tokens_per_adapter_splits = (2, 3)
            self.disable_sequence_parallel_comm = False
            self.to_wrap.allreduce_dgrad = True
            self.to_wrap.tp_group = object()

        def forward(self, x, weight=None, runtime_gather_output=None):
            assert not self.to_wrap.sequence_parallel
            assert not self.to_wrap.allreduce_dgrad
            assert self.disable_sequence_parallel_comm
            base, _ = self.to_wrap(x, weight=weight, runtime_gather_output=runtime_gather_output)
            parts = x.reshape(-1, 4).split(self.tokens_per_adapter_splits)
            adapted = torch.cat(
                [part @ adapter.t() for part, adapter in zip(parts, self.adapter_weights, strict=True)]
            )
            return base + adapted.reshape_as(base), None

    def set_slot_counts(layer, counts):
        layer.tokens_per_adapter_splits = tuple(counts.tolist())

    bridge_routing = ModuleType("megatron.bridge.peft.multi_lora_layers")
    bridge_routing.set_tokens_per_adapter_slot = set_slot_counts
    monkeypatch.setitem(sys.modules, "megatron.bridge.peft.multi_lora_layers", bridge_routing)
    layer, context = RoutedOutputLayer(), LMHeadContext()
    copied_groups = []

    def copy_to_tp(tensor, group):
        copied_groups.append(group)
        return tensor

    monkeypatch.setattr(lm_head_module, "copy_to_tensor_model_parallel_region", copy_to_tp)
    hidden = hidden_states_output_processor(
        hidden_states=torch.randn(5, 1, 4), output_layer=layer, output_weight=None, context=context
    ).squeeze(0)
    rows = torch.tensor([0, 2, 4])
    selected = hidden.index_select(0, rows)
    logits = context.project(selected, rows)
    expected = selected @ layer.to_wrap.weight.t() + torch.cat(
        [selected[:1] @ layer.adapter_weights[0].t(), selected[1:] @ layer.adapter_weights[1].t()]
    )
    torch.testing.assert_close(logits, expected)
    assert layer.tokens_per_adapter_splits == (1, 2)
    assert not layer.disable_sequence_parallel_comm
    assert layer.to_wrap.allreduce_dgrad
    assert copied_groups == [layer.to_wrap.tp_group]
    logits.sum().backward()
    assert layer.adapter_weights.grad is not None
    assert layer.adapter_weights.grad.abs().sum(dim=(1, 2)).gt(0).all()


def test_a_model_with_the_hook_is_accepted():
    check_model_takes_output_processor([_ModelWithHook(), _ModelWithHook()])


def test_a_bridge_model_whose_instance_forward_is_wrapped_is_accepted():
    model = _ModelWithHook()
    model.forward = lambda *args, **kwargs: None  # miles' bridge-mode logits-only wrapper
    check_model_takes_output_processor([model])


def test_a_bridge_model_that_delegates_to_gpt_model_is_accepted():
    check_model_takes_output_processor([_ModelWithDelegatedHook()])


def test_a_model_without_the_hook_is_rejected_before_the_forward():
    with pytest.raises(ValueError, match="_ModelWithoutHook does not"):
        check_model_takes_output_processor([_ModelWithHook(), _ModelWithoutHook()])
