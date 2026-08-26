from argparse import Namespace
from types import SimpleNamespace

import pytest

# TODO: add back after megatron bump
pytestmark = pytest.mark.skip(
    reason="true-on-policy megatron backend is not in miles-main-20260819 yet; re-enable in the true-on-policy follow-up PR."
)


def _make_args():
    return Namespace(
        custom_model_provider_path=None,
        megatron_to_hf_mode="raw",
        transformer_impl="local",
        spec=None,
        num_experts=None,
        moe_grouped_gemm=False,
        qk_layernorm=True,
        multi_latent_attention=False,
        moe_use_legacy_grouped_gemm=False,
        normalization="RMSNorm",
        fp8_param_gather=False,
        padded_vocab_size=151936,
        max_position_embeddings=4096,
        fp16_lm_cross_entropy=False,
        untie_embeddings_and_output_weights=False,
        position_embedding_type="rope",
        rotary_percent=1.0,
        rotary_base=1000000,
        use_rope_scaling=False,
        mtp_num_layers=0,
    )


def _patch_local_model_provider(monkeypatch):
    from miles.backends.megatron_utils import model_provider as provider_module

    captured = {}

    def fake_core_transformer_config_from_args(_args):
        return SimpleNamespace(
            hidden_size=2560,
            sequence_parallel=False,
            use_kitchen=False,
            true_on_policy_contract="qwen3_dense_true_on_policy_v1",
            use_kitchen_attention=False,
            kitchen_attention_backend="sdpa",
        )

    def fake_get_gpt_layer_local_spec(**kwargs):
        captured.update(kwargs)
        return "layer-spec"

    class FakeGPTModel:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.config = kwargs["config"]
            self.share_embeddings_and_output_weights = kwargs["share_embeddings_and_output_weights"]

    monkeypatch.setattr(
        provider_module,
        "core_transformer_config_from_args",
        fake_core_transformer_config_from_args,
    )
    monkeypatch.setattr(
        provider_module,
        "get_gpt_layer_local_spec",
        fake_get_gpt_layer_local_spec,
    )
    monkeypatch.setattr(provider_module, "GPTModel", FakeGPTModel)

    return provider_module, captured


def test_local_model_provider_passes_true_on_policy_spec_flags(monkeypatch):
    provider_module, captured = _patch_local_model_provider(monkeypatch)

    provider = provider_module.get_model_provider_func(_make_args())
    model = provider()

    assert model.kwargs["transformer_layer_spec"] == "layer-spec"
    assert model.share_embeddings_and_output_weights is True
    assert captured["normalization"] == "RMSNorm"
    assert captured["use_true_on_policy_backend"] is True
    assert captured["use_kitchen"] is False
    assert captured["use_kitchen_attention"] is False
    assert captured["kitchen_attention_backend"] == "sdpa"


def test_critic_provider_builds_untied_scalar_output_head(monkeypatch):
    provider_module, _ = _patch_local_model_provider(monkeypatch)

    provider = provider_module.get_model_provider_func(_make_args(), role="critic")
    model = provider(pre_process=False, post_process=True)

    assert model.share_embeddings_and_output_weights is False
    assert tuple(model.output_layer.weight.shape) == (1, 2560)
