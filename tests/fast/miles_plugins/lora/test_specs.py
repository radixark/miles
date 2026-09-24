"""Unit tests for the native-LoRA architecture specs and HF target contract — no GPU."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from miles.utils.lora.hf_lora_targets import resolve_hf_lora_targets
from miles_plugins.lora.config import LoRAConfig
from miles_plugins.lora.registry import MODEL_SPECS, resolve_adapter_targets
from miles_plugins.lora.spec.attention import GQAAttentionSpec, MLAAttentionSpec
from miles_plugins.lora.spec.base import AttachContext


def _assert_supported_architecture(config, tp_size: int = 1) -> None:
    spec = MLAAttentionSpec() if bool(getattr(config, "multi_latent_attention", False)) else GQAAttentionSpec()
    spec.validate(config, tp_size=tp_size)


def _config(*, mla=False, num_query_groups=8, q_lora_rank=1536):
    return SimpleNamespace(multi_latent_attention=mla, num_query_groups=num_query_groups, q_lora_rank=q_lora_rank)


class TestArchitectureGuards:
    def test_plain_gqa_passes(self):
        _assert_supported_architecture(_config())

    def test_mla_ignores_the_query_group_bound(self):
        _assert_supported_architecture(_config(mla=True, num_query_groups=2), tp_size=4)

    def test_query_groups_below_tp_size_rejected(self):
        with pytest.raises(AssertionError, match="num_query_groups.*--megatron-to-hf-mode bridge"):
            _assert_supported_architecture(_config(num_query_groups=2), tp_size=4)

    def test_query_groups_equal_to_tp_size_passes(self):
        _assert_supported_architecture(_config(num_query_groups=4), tp_size=4)

    def test_mla_without_q_lora_rank_rejected(self):
        """An uncompressed query path exports an unfused q_proj SGLang's qkv_a loader cannot ingest."""
        with pytest.raises(AssertionError, match="q_lora_rank"):
            _assert_supported_architecture(_config(mla=True, q_lora_rank=None))


def _qwen3_modules(num_layers=2):
    return [
        f"model.layers.{layer}.{module}"
        for layer in range(num_layers)
        for module in (
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        )
    ]


class TestResolveAdapterTargets:
    """Arg-time contract: HF targets from #3318's resolution, validated against the native spec."""

    def test_complete_fused_families_pass_through(self):
        targets = ["q_proj", "k_proj", "v_proj", "o_proj"]
        assert resolve_adapter_targets({"model_type": "qwen3"}, targets, hf_modules=_qwen3_modules()) == targets

    def test_partial_fused_family_adds_zero_filled_serving_siblings(self):
        targets = ["model.layers.*.self_attn.q_proj", "model.layers.*.mlp.up_proj"]
        assert resolve_adapter_targets({"model_type": "qwen3"}, targets, hf_modules=_qwen3_modules()) == [
            *targets,
            "model.layers.*.self_attn.k_proj",
            "model.layers.*.self_attn.v_proj",
            "model.layers.*.mlp.gate_proj",
        ]

    def test_mla_compressed_projections_share_a_serving_family(self):
        targets = ["model.layers.*.self_attn.q_a_proj"]
        assert resolve_adapter_targets({"model_type": "deepseek_v3"}, targets, hf_modules=[]) == [
            *targets,
            "model.layers.*.self_attn.kv_a_proj_with_mqa",
        ]

    def test_shared_experts_are_attachable(self):
        targets = [f"model.layers.*.mlp.shared_experts.{leaf}" for leaf in ("gate_proj", "up_proj", "down_proj")]
        assert resolve_adapter_targets({"model_type": "glm4_moe"}, targets, hf_modules=[]) == targets

    @pytest.mark.parametrize(
        "model_type,target",
        [
            ("qwen3_moe", "model.layers.*.mlp.experts.gate_up_proj"),
            ("kimi_k2", "model.layers.*.mlp.experts.*.gate_proj"),
            ("qwen3_5", "model.language_model.layers.*.linear_attn.in_proj_qkv"),
        ],
    )
    def test_unimplemented_projections_fail_closed(self, model_type, target):
        with pytest.raises(AssertionError, match="does not implement adapters"):
            resolve_adapter_targets({"model_type": model_type}, [target], hf_modules=[])

    def test_leaf_selectors_are_checked_against_the_modules_they_select(self):
        modules = [*_qwen3_modules(1), "model.layers.1.mlp.experts.3.gate_proj"]
        with pytest.raises(AssertionError, match=r"layers\.1\.mlp\.experts\.3\.gate_proj"):
            resolve_adapter_targets({"model_type": "qwen3"}, ["gate_proj", "up_proj"], hf_modules=modules)

    def test_unregistered_model_type_fails_closed(self):
        with pytest.raises(AssertionError, match="no spec registered"):
            resolve_adapter_targets({"model_type": "gpt_oss"}, ["q_proj"], hf_modules=[])


class TestInklingSpec:
    @pytest.mark.parametrize("multimodal", [False, True], ids=["text", "multimodal"])
    def test_complete_layout_is_served_through_sglang_auto_detection(self, multimodal):
        config = dict(model_type="inkling_text", mlp_layer_types=["dense", "sparse"], n_shared_experts=0)
        if multimodal:
            config = dict(model_type="inkling_mm_model", text_config=config)
        hf_targets = resolve_hf_lora_targets(config)
        assert resolve_adapter_targets(config, hf_targets, hf_modules=[]) == "all-linear"
        with pytest.raises(AssertionError, match="complete adapter layout"):
            resolve_adapter_targets(config, [t for t in hf_targets if not t.endswith(".up_proj")], hf_modules=[])

    def test_legacy_config_uses_the_same_layout(self):
        legacy = dict(model_type="inkling_model", dense_mlp_idx=1, num_hidden_layers=2, n_shared_experts=1)
        assert resolve_adapter_targets(legacy, resolve_hf_lora_targets(legacy), hf_modules=[]) == "all-linear"

    def test_spec_carries_tml_naming_and_custom_hooks(self):
        spec = MODEL_SPECS["inkling_mm_model"]
        assert spec.complete_layout
        assert spec.attention.hf_block == "attn"
        assert spec.attention.supported_targets == {"wq_du", "wk_dv", "wv_dv", "wr_du", "wo_ud"}
        assert spec.mlp.supported_targets == {"gate_up_proj", "down_proj"}
        assert spec.experts is not None and spec.lm_head is not None


class TestLoRAConfigSelection:
    def test_scoped_and_leaf_selectors(self):
        config = LoRAConfig(rank=2, alpha=4, dropout=0.0, target_modules=("model.layers.0.self_attn.o_proj", "q_proj"))
        assert config.selects("model.layers.0.self_attn.o_proj")
        assert not config.selects("model.layers.1.self_attn.o_proj")
        assert config.selects("model.layers.7.self_attn.q_proj")

    def test_complete_layout_selects_everything(self):
        assert LoRAConfig(rank=2, alpha=4, dropout=0.0, target_modules=None).selects(
            "language_model.layers.0.attn.wq_du"
        )


class _Linear(nn.Module):
    def __init__(self, out_features, in_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))

    def forward(self, x):
        return x @ self.weight.t(), None


def _gqa_attention(hidden=8, heads=2, head_dim=4):
    attention = nn.Module()
    attention.linear_qkv = _Linear(3 * heads * head_dim, hidden)
    attention.linear_proj = _Linear(hidden, heads * head_dim)
    attention.num_attention_heads_per_partition = heads
    attention.num_query_groups_per_partition = heads
    attention.hidden_size_per_attention_head = head_dim
    return attention


class TestAttachWalk:
    """The attach walk selects projections by their full HF path."""

    @staticmethod
    def _context(targets):
        return AttachContext(
            lora=LoRAConfig(rank=2, alpha=4, dropout=0.0, target_modules=targets),
            transformer_config=SimpleNamespace(hidden_size=8, sequence_parallel=False, layernorm_epsilon=1e-5),
            tp_size=1,
            tp_rank=0,
            layer_prefix="model.layers.",
            shared_expert="mlp.shared_experts.",
        )

    def test_layer_scoped_target_attaches_only_that_layer(self):
        context = self._context(("model.layers.0.self_attn.o_proj",))
        layer0, layer1 = _gqa_attention(), _gqa_attention()
        assert GQAAttentionSpec().attach(layer0, "model.layers.0.self_attn.", context) == 1
        assert GQAAttentionSpec().attach(layer1, "model.layers.1.self_attn.", context) == 0
        assert hasattr(layer0, "lora_o_adapter") and not hasattr(layer0, "lora_qkv_adapter")
        assert not hasattr(layer1, "lora_o_adapter")

    def test_partial_fused_selection_builds_a_split_adapter_for_the_selected_rows(self):
        attention = _gqa_attention()
        GQAAttentionSpec().attach(attention, "model.layers.0.self_attn.", self._context(("q_proj",)))
        assert [projection.hf for projection in attention.lora_qkv_adapter.projection_specs] == ["q_proj"]
