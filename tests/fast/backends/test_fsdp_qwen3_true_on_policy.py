from types import SimpleNamespace

import pytest
import torch
from transformers.models.qwen3 import modeling_qwen3
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from miles.backends.experimental.fsdp_utils.adaptations.class_patches import (
    _MODEL_INSTANCE_PATCH_HOOKS,
    apply_model_instance_patches,
)
from miles.backends.experimental.fsdp_utils.adaptations.precision import apply_fp32_master, resolve_precision_policy
from miles.backends.experimental.fsdp_utils.models.qwen3 import (
    Qwen3FinalRMSNorm,
    apply_qwen3_dense_true_on_policy_patch,
    resolve_qwen3_dense_sync_dtype,
)
from miles.true_on_policy.contracts import QWEN3_DENSE_TRUE_ON_POLICY_V1


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


def test_qwen3_instance_patch_registry_is_contract_gated(monkeypatch):
    from miles.backends.experimental.fsdp_utils.models import qwen3 as qwen3_model

    hook = {hook.name: hook for hook in _MODEL_INSTANCE_PATCH_HOOKS}["qwen3_dense_true_on_policy"]
    calls = []
    monkeypatch.setattr(
        qwen3_model,
        "apply_qwen3_dense_true_on_policy_patch",
        lambda model: calls.append(model),
    )
    model = object()
    formal_contract = QWEN3_DENSE_TRUE_ON_POLICY_V1.name

    for model_type, true_on_policy_mode, contract, fp16 in [
        ("qwen3", False, formal_contract, False),
        ("qwen3", True, None, False),
        ("qwen3_moe", True, formal_contract, False),
        ("qwen3", True, formal_contract, True),
    ]:
        apply_model_instance_patches(
            model,
            SimpleNamespace(model_type=model_type),
            SimpleNamespace(
                true_on_policy_mode=true_on_policy_mode,
                sglang_true_on_policy_contract=contract,
                fp16=fp16,
            ),
        )
    assert calls == []

    args = SimpleNamespace(
        true_on_policy_mode=True,
        sglang_true_on_policy_contract=formal_contract,
        fp16=False,
    )
    config = SimpleNamespace(model_type="qwen3")
    assert hook.applies_to(config, args)
    apply_model_instance_patches(model, config, args)
    assert calls == [model]


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


def test_qwen3_formal_sync_preserves_post_update_fp32_values():
    model = modeling_qwen3.Qwen3ForCausalLM(_tiny_config()).to(torch.bfloat16)
    policy = resolve_precision_policy(
        model.config,
        SimpleNamespace(
            fp16=False,
            keep_fp32_master=True,
            true_on_policy_mode=True,
            sglang_true_on_policy_contract=QWEN3_DENSE_TRUE_ON_POLICY_V1.name,
        ),
    )
    model = apply_fp32_master(model, policy.sync_dtype_resolver)

    fp32_name = "model.layers.0.input_layernorm.weight"
    bf16_name = "model.norm.weight"
    params = dict(model.named_parameters())
    with torch.no_grad():
        params[fp32_name].add_(1e-6)
        params[bf16_name].add_(1e-6)

    sync_dtypes = model._fsdp_sync_dtypes
    fp32_synced = params[fp32_name].to(sync_dtypes[fp32_name])
    bf16_synced = params[bf16_name].to(sync_dtypes[bf16_name])

    assert torch.equal(fp32_synced, params[fp32_name])
    assert not torch.equal(bf16_synced.to(torch.float32), params[bf16_name])
