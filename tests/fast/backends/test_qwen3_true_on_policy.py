from types import SimpleNamespace

import pytest
import torch

from miles.backends.experimental.fsdp_utils.actor import apply_fsdp2
from miles.backends.experimental.fsdp_utils.models.qwen3 import Qwen3DecoderLayer, Qwen3RMSNorm, _fp32_rms_norm


def test_qwen3_true_on_policy_registry_gate(monkeypatch):
    from miles.backends.experimental.fsdp_utils.adaptations.class_patches import _MODEL_PATCH_HOOKS
    from miles.backends.experimental.fsdp_utils.models import qwen3 as qwen3_model

    hook = {hook.name: hook for hook in _MODEL_PATCH_HOOKS}["qwen3_true_on_policy_precision"]
    assert hook.applies_to(SimpleNamespace(model_type="qwen3"))
    assert not hook.applies_to(SimpleNamespace(model_type="qwen3_moe"))

    calls = []
    monkeypatch.setattr(qwen3_model, "apply_true_on_policy_patch_for_qwen3", lambda: calls.append(True))
    hook.apply(
        SimpleNamespace(model_type="qwen3"),
        SimpleNamespace(
            true_on_policy_mode=False,
            sglang_true_on_policy_contract="qwen3_dense_true_on_policy_v1",
            fp16=False,
        ),
    )
    hook.apply(
        SimpleNamespace(model_type="qwen3"),
        SimpleNamespace(
            true_on_policy_mode=True,
            sglang_true_on_policy_contract=None,
            fp16=False,
        ),
    )
    assert calls == []
    hook.apply(
        SimpleNamespace(model_type="qwen3"),
        SimpleNamespace(
            true_on_policy_mode=True,
            sglang_true_on_policy_contract="qwen3_dense_true_on_policy_v1",
            fp16=False,
        ),
    )
    assert calls == [True]
    with pytest.raises(ValueError, match="requires FSDP bf16 compute"):
        hook.apply(
            SimpleNamespace(model_type="qwen3"),
            SimpleNamespace(
                true_on_policy_mode=True,
                sglang_true_on_policy_contract="qwen3_dense_true_on_policy_v1",
                fp16=True,
            ),
        )


def test_qwen3_contract_rms_norm_cast_order():
    norm = Qwen3RMSNorm(4, eps=1e-6).to(torch.bfloat16)
    norm.weight.data.copy_(torch.tensor([0.5, 1.0, 1.5, 2.0], dtype=torch.bfloat16))
    hidden_states = torch.tensor(
        [[0.25, -0.5, 1.0, -2.0]],
        dtype=torch.bfloat16,
    )

    variance = hidden_states.float().pow(2).mean(-1, keepdim=True)
    normalized = hidden_states.float() * torch.rsqrt(variance + norm.variance_epsilon)
    expected_qk = norm.weight.float() * normalized.to(torch.bfloat16)
    qk_output = norm(hidden_states)
    assert qk_output.dtype == torch.float32
    torch.testing.assert_close(qk_output, expected_qk, rtol=0, atol=0)

    fp32_hidden_states = hidden_states.float() + 0.0001
    variance = fp32_hidden_states.pow(2).mean(-1, keepdim=True)
    normalized = fp32_hidden_states * torch.rsqrt(variance + norm.variance_epsilon)
    expected_final = norm.weight * normalized.to(torch.bfloat16)
    final_output = norm(fp32_hidden_states)
    assert final_output.dtype == torch.bfloat16
    torch.testing.assert_close(final_output, expected_final, rtol=0, atol=0)

    decoder_norm = _fp32_rms_norm(norm, fp32_hidden_states)
    expected_decoder = norm.weight.float() * normalized
    assert decoder_norm.dtype == torch.float32
    torch.testing.assert_close(decoder_norm, expected_decoder, rtol=0, atol=0)


def test_qwen3_contract_preserves_fp32_residual_between_layers():
    from transformers import Qwen3Config
    from transformers.models.qwen3 import modeling_qwen3

    original_classes = {
        name: getattr(modeling_qwen3, name)
        for name in ("Qwen3RMSNorm", "Qwen3RotaryEmbedding", "Qwen3Attention", "Qwen3DecoderLayer")
    }
    try:
        from miles.backends.experimental.fsdp_utils.models.qwen3 import apply_true_on_policy_patch_for_qwen3

        apply_true_on_policy_patch_for_qwen3()
        config = Qwen3Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=4,
            max_position_embeddings=32,
        )
        config._attn_implementation = "eager"
        model = modeling_qwen3.Qwen3ForCausalLM(config).to(torch.bfloat16)

        dtypes = {}

        def record_input(name):
            def hook(module, args):
                dtypes[name] = args[0].dtype

            return hook

        def record_output(name):
            def hook(module, args, output):
                dtypes[name] = output.dtype

            return hook

        handles = [
            model.model.layers[0].register_forward_pre_hook(record_input("layer_0_input")),
            model.model.layers[1].register_forward_pre_hook(record_input("layer_1_input")),
            model.model.layers[0].self_attn.q_proj.register_forward_pre_hook(record_input("q_proj_input")),
            model.model.layers[0].self_attn.q_norm.register_forward_hook(record_output("q_norm_output")),
            model.model.norm.register_forward_pre_hook(record_input("final_norm_input")),
            model.lm_head.register_forward_pre_hook(record_input("lm_head_input")),
        ]
        try:
            output = model(input_ids=torch.tensor([[1, 2, 3, 4]]), use_cache=False)
        finally:
            for handle in handles:
                handle.remove()

        output.logits.float().sum().backward()
        assert dtypes == {
            "layer_0_input": torch.bfloat16,
            "q_proj_input": torch.bfloat16,
            "q_norm_output": torch.float32,
            "layer_1_input": torch.float32,
            "final_norm_input": torch.float32,
            "lm_head_input": torch.bfloat16,
        }
        assert output.logits.dtype == torch.bfloat16
        assert all(isinstance(layer, Qwen3DecoderLayer) for layer in model.model.layers)
        assert model.model.layers[1].self_attn.q_proj.weight.grad is not None
    finally:
        for name, cls in original_classes.items():
            setattr(modeling_qwen3, name, cls)


def test_apply_fsdp2_preserves_only_marked_module_inputs(monkeypatch):
    import torch.distributed.fsdp

    class MarkedDecoder(torch.nn.Module):
        _fsdp_preserve_forward_input_dtype = True

    class PlainDecoder(torch.nn.Module):
        pass

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.marked = MarkedDecoder()
            self.plain = PlainDecoder()
            self._no_split_modules = ["MarkedDecoder", "PlainDecoder"]
            self.config = SimpleNamespace(tie_word_embeddings=True)

    calls = []
    monkeypatch.setattr(
        torch.distributed.fsdp,
        "fully_shard",
        lambda module, **kwargs: calls.append((module, kwargs["mp_policy"])),
    )
    model = Model()
    apply_fsdp2(model, args=SimpleNamespace(fp16=False))

    by_module = {module: policy for module, policy in calls}
    assert by_module[model.marked].cast_forward_inputs is False
    assert by_module[model.plain].cast_forward_inputs is True
    assert by_module[model].cast_forward_inputs is True
    assert all(policy.param_dtype == torch.bfloat16 for policy in by_module.values())
    assert all(policy.reduce_dtype == torch.float32 for policy in by_module.values())
