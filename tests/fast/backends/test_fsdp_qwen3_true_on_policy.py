from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from transformers.models.qwen3 import modeling_qwen3
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from miles.backends.experimental.fsdp_utils.adaptations.class_patches import (
    _MODEL_INSTANCE_PATCH_HOOKS,
    apply_model_instance_patches,
)
from miles.backends.experimental.fsdp_utils.adaptations.precision import apply_fp32_master
from miles.backends.experimental.fsdp_utils.models.qwen3 import (
    Qwen3Attention,
    Qwen3DecoderLayer,
    Qwen3FinalRMSNorm,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
    _fp32_rms_norm,
    apply_qwen3_dense_true_on_policy_patch,
)
from miles.backends.experimental.fsdp_utils.update_weight_utils import UpdateWeight
from miles.true_on_policy.contracts import QWEN3_DENSE_TRUE_ON_POLICY_V1


def _contract_fp32_param_names():
    return {
        "model.embed_tokens.weight",
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.post_attention_layernorm.weight",
        "model.layers.0.self_attn.q_norm.weight",
        "model.layers.0.self_attn.k_norm.weight",
    }


@pytest.mark.parametrize("tie_word_embeddings", [False, True])
def test_qwen3_patch_applies_contract_storage_dtypes_and_is_idempotent(tie_word_embeddings):
    config = _tiny_config(tie_word_embeddings=tie_word_embeddings)
    model = modeling_qwen3.Qwen3ForCausalLM(config).to(torch.bfloat16)
    final_norm = model.model.norm
    final_norm_weight = final_norm.weight

    assert apply_qwen3_dense_true_on_policy_patch(model)
    assert not apply_qwen3_dense_true_on_policy_patch(model)

    assert model.model.norm is final_norm
    assert model.model.norm.weight is final_norm_weight
    assert isinstance(model.model.norm, Qwen3FinalRMSNorm)
    assert isinstance(model.model.rotary_emb, Qwen3RotaryEmbedding)
    assert isinstance(model.model.layers[0], Qwen3DecoderLayer)
    assert isinstance(model.model.layers[0].self_attn, Qwen3Attention)
    assert isinstance(model.model.layers[0].input_layernorm, Qwen3RMSNorm)
    assert isinstance(model.model.layers[0].post_attention_layernorm, Qwen3RMSNorm)
    assert isinstance(model.model.layers[0].self_attn.q_norm, Qwen3RMSNorm)
    assert isinstance(model.model.layers[0].self_attn.k_norm, Qwen3RMSNorm)
    assert "model.norm.weight" in model.state_dict()
    assert (model.model.embed_tokens.weight is model.lm_head.weight) is tie_word_embeddings
    for name, param in model.named_parameters():
        expected_dtype = torch.float32 if name in _contract_fp32_param_names() else torch.bfloat16
        assert param.dtype is expected_dtype, name

    later_nonformal_model = modeling_qwen3.Qwen3ForCausalLM(config).to(torch.bfloat16)
    assert type(later_nonformal_model.model.norm) is modeling_qwen3.Qwen3RMSNorm
    assert type(later_nonformal_model.model.layers[0]) is modeling_qwen3.Qwen3DecoderLayer
    assert type(later_nonformal_model.model.layers[0].self_attn) is modeling_qwen3.Qwen3Attention
    assert {param.dtype for param in later_nonformal_model.parameters()} == {torch.bfloat16}


def _tiny_config(*, tie_word_embeddings=True):
    return Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        tie_word_embeddings=tie_word_embeddings,
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


def test_qwen3_contract_rms_norm_cast_order():
    norm = Qwen3RMSNorm(4, eps=1e-6).to(torch.float32)
    norm.weight.data.copy_(torch.tensor([0.5, 1.0, 1.5, 2.0]))
    bf16_hidden_states = torch.tensor([[0.25, -0.5, 1.0, -2.0]], dtype=torch.bfloat16)

    variance = bf16_hidden_states.float().pow(2).mean(-1, keepdim=True)
    normalized = bf16_hidden_states.float() * torch.rsqrt(variance + norm.variance_epsilon)
    expected_qk = norm.weight * normalized.to(torch.bfloat16)
    qk_output = norm(bf16_hidden_states)

    assert qk_output.dtype is torch.float32
    torch.testing.assert_close(qk_output, expected_qk, rtol=0, atol=0)

    fp32_hidden_states = bf16_hidden_states.float() + 0.0001
    decoder_output = _fp32_rms_norm(norm, fp32_hidden_states)
    variance = fp32_hidden_states.pow(2).mean(-1, keepdim=True)
    expected_decoder = norm.weight * (fp32_hidden_states * torch.rsqrt(variance + norm.variance_epsilon))

    assert decoder_output.dtype is torch.float32
    torch.testing.assert_close(decoder_output, expected_decoder, rtol=0, atol=0)


class _CapturingUpdater(UpdateWeight):
    def connect_rollout_engines(
        self,
        rollout_engines,
        rollout_engine_lock,
        engine_gpu_counts=None,
        engine_gpu_offsets=None,
    ):
        pass

    def update_bucket_weights(self, named_tensors, weight_version=None):
        self.sent = dict(named_tensors)


@pytest.mark.parametrize("tie_word_embeddings", [False, True])
def test_qwen3_fp32_master_sync_preserves_contract_weight_updates(tie_word_embeddings):
    model = modeling_qwen3.Qwen3ForCausalLM(_tiny_config(tie_word_embeddings=tie_word_embeddings)).to(torch.bfloat16)
    apply_qwen3_dense_true_on_policy_patch(model)
    apply_fp32_master(model)

    sync_dtypes = model._fsdp_sync_orig_dtypes
    for name in _contract_fp32_param_names():
        assert sync_dtypes[name] is torch.float32
    assert sync_dtypes["model.layers.0.self_attn.q_proj.weight"] is torch.bfloat16
    assert sync_dtypes["model.norm.weight"] is torch.bfloat16
    assert sync_dtypes["lm_head.weight"] is torch.bfloat16

    fp32_value = torch.tensor(0.9898309112)
    embed_weight = model.model.embed_tokens.weight
    norm_weight = model.model.layers[0].input_layernorm.weight
    dense_weight = model.model.layers[0].self_attn.q_proj.weight
    lm_head_weight = model.lm_head.weight
    with torch.no_grad():
        embed_weight.flatten()[0].copy_(fp32_value)
        norm_weight.flatten()[0].copy_(fp32_value)
        dense_weight.flatten()[0].copy_(fp32_value)
        lm_head_weight.flatten()[0].copy_(fp32_value)

    updater = _CapturingUpdater(SimpleNamespace(), model)
    updater.wait_and_update_bucket_weights(
        [
            ("model.embed_tokens.weight", embed_weight, sync_dtypes["model.embed_tokens.weight"]),
            (
                "model.layers.0.input_layernorm.weight",
                norm_weight,
                sync_dtypes["model.layers.0.input_layernorm.weight"],
            ),
            (
                "model.layers.0.self_attn.q_proj.weight",
                dense_weight,
                sync_dtypes["model.layers.0.self_attn.q_proj.weight"],
            ),
            ("lm_head.weight", lm_head_weight, sync_dtypes["lm_head.weight"]),
        ]
    )

    assert updater.sent["model.embed_tokens.weight"].dtype is torch.float32
    assert updater.sent["model.embed_tokens.weight"].flatten()[0] == fp32_value
    assert updater.sent["model.layers.0.input_layernorm.weight"].dtype is torch.float32
    assert updater.sent["model.layers.0.input_layernorm.weight"].flatten()[0] == fp32_value
    assert updater.sent["lm_head.weight"].dtype is torch.bfloat16
    assert updater.sent["lm_head.weight"].flatten()[0].float() != fp32_value
    assert updater.sent["model.layers.0.self_attn.q_proj.weight"].dtype is torch.bfloat16
    assert updater.sent["model.layers.0.self_attn.q_proj.weight"].flatten()[0].float() != fp32_value


def test_qwen3_ref_update_preserves_actor_master_weights():
    actor = modeling_qwen3.Qwen3ForCausalLM(_tiny_config()).to(torch.bfloat16)
    apply_qwen3_dense_true_on_policy_patch(actor)
    apply_fp32_master(actor)

    fp32_value = torch.tensor(0.9898309112)
    with torch.no_grad():
        actor.model.embed_tokens.weight.flatten()[0].copy_(fp32_value)
        actor.model.layers[0].input_layernorm.weight.flatten()[0].copy_(fp32_value)
        actor.model.layers[0].self_attn.q_proj.weight.flatten()[0].copy_(fp32_value)

    ref = modeling_qwen3.Qwen3ForCausalLM(_tiny_config()).to(torch.bfloat16)
    apply_qwen3_dense_true_on_policy_patch(ref)
    apply_fp32_master(ref)
    ref.load_state_dict(actor.state_dict())

    assert ref.model.embed_tokens.weight.dtype is torch.float32
    assert ref.model.embed_tokens.weight.flatten()[0] == fp32_value
    assert ref.model.layers[0].input_layernorm.weight.dtype is torch.float32
    assert ref.model.layers[0].input_layernorm.weight.flatten()[0] == fp32_value
    assert ref.model.layers[0].self_attn.q_proj.weight.dtype is torch.float32
    assert ref.model.layers[0].self_attn.q_proj.weight.flatten()[0] == fp32_value


def test_qwen3_ref_model_has_uniform_fp32_storage_before_fsdp(monkeypatch):
    from miles.backends.experimental.fsdp_utils import actor as actor_module

    config = _tiny_config()
    args = SimpleNamespace(
        attn_implementation="eager",
        fp16=False,
        keep_fp32_master=True,
        true_on_policy_mode=True,
        sglang_true_on_policy_contract=QWEN3_DENSE_TRUE_ON_POLICY_V1.name,
    )
    actor = object.__new__(actor_module.FSDPTrainRayActor)
    actor.args = args
    actor.hf_config = config
    actor.precision_policy = SimpleNamespace(
        keep_fp32_master=True,
        param_dtype=torch.float32,
        reduce_dtype=torch.float32,
    )
    actor._get_init_weight_context_manager = lambda: nullcontext
    actor.get_model_cls = lambda: SimpleNamespace(
        from_pretrained=lambda *args, **kwargs: modeling_qwen3.Qwen3ForCausalLM(config).to(torch.bfloat16)
    )
    actor._fsdp2_load_full_state_dict = lambda model, full_state, device_mesh, cpu_offload: model

    mesh = object()
    captured = {}

    def capture_apply_fsdp2(model, **kwargs):
        captured["dtypes"] = {param.dtype for param in model.parameters()}
        captured["sync_dtypes"] = model._fsdp_sync_orig_dtypes
        return model

    monkeypatch.setattr(actor_module.os.path, "isdir", lambda path: True)
    monkeypatch.setattr(actor_module.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(actor_module, "get_parallel_state", lambda: SimpleNamespace(dp_mesh=mesh))
    monkeypatch.setattr(actor_module, "apply_fsdp2", capture_apply_fsdp2)

    ref = actor._create_ref_model("/checkpoint")

    assert captured["dtypes"] == {torch.float32}
    assert captured["sync_dtypes"]["model.embed_tokens.weight"] is torch.float32
    assert captured["sync_dtypes"]["model.layers.0.self_attn.q_proj.weight"] is torch.bfloat16
    assert {param.dtype for param in ref.parameters()} == {torch.float32}
