
import pytest
import torch
from transformers.models.qwen3 import modeling_qwen3
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from miles.backends.experimental.fsdp_utils.models.qwen3 import (
    Qwen3FinalRMSNorm,
    apply_qwen3_dense_true_on_policy_patch,
    resolve_qwen3_dense_sync_dtype,
)


def test_qwen3_patch_changes_only_final_norm_and_is_idempotent():
    config = _tiny_config()
    model = modeling_qwen3.Qwen3ForCausalLM(config)
    final_norm = model.model.norm
    final_norm_weight = final_norm.weight

    assert apply_qwen3_dense_true_on_policy_patch(model)
    assert not apply_qwen3_dense_true_on_policy_patch(model)

    assert model.model.norm is final_norm
    assert model.model.norm.weight is final_norm_weight
    assert isinstance(model.model.norm, Qwen3FinalRMSNorm)
    assert type(model.model.layers[0].input_layernorm) is modeling_qwen3.Qwen3RMSNorm
    assert type(model.model.layers[0].post_attention_layernorm) is modeling_qwen3.Qwen3RMSNorm
    assert type(model.model.layers[0].self_attn.q_norm) is modeling_qwen3.Qwen3RMSNorm
    assert "model.norm.weight" in model.state_dict()

    later_nonformal_model = modeling_qwen3.Qwen3ForCausalLM(config)
    assert type(later_nonformal_model.model.norm) is modeling_qwen3.Qwen3RMSNorm


def _tiny_config():
    return Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
    )


def test_qwen3_final_norm_uses_contract_rounding_order():
    norm = Qwen3FinalRMSNorm(16, eps=1e-6).to(torch.float32)
    torch.manual_seed(1)
    norm.weight.data.copy_(torch.randn_like(norm.weight))
    hidden_states = torch.randn(4, 16, dtype=torch.float32)

    output = norm(hidden_states)

    normalized = hidden_states * torch.rsqrt(hidden_states.pow(2).mean(-1, keepdim=True) + norm.variance_epsilon)
    expected = norm.weight.to(torch.bfloat16) * normalized.to(torch.bfloat16)
    cast_after_fp32_multiply = (norm.weight * normalized).to(torch.bfloat16)
    assert output.dtype is torch.bfloat16
    assert torch.equal(output, expected)
    assert not torch.equal(output, cast_after_fp32_multiply)


@pytest.mark.parametrize(
    "name",
    [
        "model.embed_tokens.weight",
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.post_attention_layernorm.weight",
        "model.layers.0.self_attn.q_norm.weight",
        "model.layers.0.self_attn.k_norm.weight",
    ],
)
def test_qwen3_formal_sync_preserves_fp32_contract_parameters(name):
    assert resolve_qwen3_dense_sync_dtype(name, torch.bfloat16) is torch.float32


@pytest.mark.parametrize(
    "name",
    [
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.norm.weight",
        "lm_head.weight",
    ],
)
def test_qwen3_formal_sync_keeps_bf16_math_parameters_at_checkpoint_dtype(name):
    assert resolve_qwen3_dense_sync_dtype(name, torch.bfloat16) is torch.bfloat16
