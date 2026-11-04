"""``--log-probs-backend fused_lm_head`` switches the output layer only when asked, and only on models with the hook."""

from types import SimpleNamespace

import pytest
import torch

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
    torch.testing.assert_close(context.project(rows), logits if scale is None else logits * scale)


def test_a_model_with_the_hook_is_accepted():
    check_model_takes_output_processor([_ModelWithHook(), _ModelWithHook()])


def test_a_bridge_model_whose_instance_forward_is_wrapped_is_accepted():
    model = _ModelWithHook()
    model.forward = lambda *args, **kwargs: None  # miles' bridge-mode logits-only wrapper
    check_model_takes_output_processor([model])


def test_a_model_without_the_hook_is_rejected_before_the_forward():
    with pytest.raises(ValueError, match="_ModelWithoutHook does not"):
        check_model_takes_output_processor([_ModelWithHook(), _ModelWithoutHook()])
