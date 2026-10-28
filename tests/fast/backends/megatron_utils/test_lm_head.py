"""``--log-probs-backend fused_lm_head`` switches the output layer only when asked, and only on models with the hook."""

from types import SimpleNamespace

import pytest
import torch

from miles.backends.megatron_utils.lm_head import (
    LMHeadContext,
    check_model_takes_output_processor,
    lm_head_forward_kwargs,
    lm_head_hidden_states,
)


class _ModelWithHook(torch.nn.Module):
    def forward(self, input_ids, *, output_processor=None, output_processor_context=None):
        raise NotImplementedError


class _ModelWithoutHook(torch.nn.Module):  # stands for HybridModel, whose forward has no output_processor
    def forward(self, input_ids):
        raise NotImplementedError


def test_without_lm_head_input_the_model_keeps_its_output():
    assert lm_head_forward_kwargs(lm_head_input=False) == ({}, None)


def test_lm_head_input_passes_the_processor_and_its_context():
    kwargs, context = lm_head_forward_kwargs(lm_head_input=True)
    assert kwargs["output_processor"] is lm_head_hidden_states
    assert kwargs["output_processor_context"] is context
    assert isinstance(context, LMHeadContext) and context.output_layer is None


def test_the_processor_rejects_mup_which_scales_the_logits_after_the_output_layer():
    with pytest.raises(ValueError, match="MuP"):
        lm_head_hidden_states(
            hidden_states=torch.zeros(4, 1, 8),
            output_layer=torch.nn.Linear(8, 16),
            output_weight=None,
            context=LMHeadContext(),
            config=SimpleNamespace(use_mup=True),
        )


def test_a_model_with_the_hook_is_accepted():
    check_model_takes_output_processor([_ModelWithHook(), _ModelWithHook()])


def test_a_bridge_model_whose_instance_forward_is_wrapped_is_accepted():
    model = _ModelWithHook()
    model.forward = lambda *args, **kwargs: None  # miles' bridge-mode logits-only wrapper
    check_model_takes_output_processor([model])


def test_a_model_without_the_hook_is_rejected_before_the_forward():
    with pytest.raises(ValueError, match="_ModelWithoutHook does not"):
        check_model_takes_output_processor([_ModelWithHook(), _ModelWithoutHook()])
