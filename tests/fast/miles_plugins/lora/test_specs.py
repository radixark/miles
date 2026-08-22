"""Unit tests for the native-LoRA architecture specs and target sets — no GPU."""

from types import SimpleNamespace

import pytest

from miles.backends.megatron_utils.lora_utils import convert_target_modules_to_hf
from miles_plugins.lora.spec.attention import GQAAttentionSpec, MLAAttentionSpec
from miles_plugins.lora.spec.mlp import FusedGatedMLPSpec

IMPLEMENTED_TARGETS = (
    GQAAttentionSpec().supported_targets | MLAAttentionSpec().supported_targets | FusedGatedMLPSpec().supported_targets
)


def _assert_supported_architecture(config, tp_size: int = 1) -> None:
    """Dispatch to the family's attention-spec validate the way the registry resolves it."""
    spec = MLAAttentionSpec() if bool(getattr(config, "multi_latent_attention", False)) else GQAAttentionSpec()
    spec.validate(config, tp_size=tp_size)


def _fake_model(num_layers=2, *, output_gate=False, mla=False, with_qkv=True, num_query_groups=8, q_lora_rank=1536):
    layers = []
    for i in range(num_layers):
        attn = SimpleNamespace(layer_number=i + 1)
        if with_qkv:
            attn.linear_qkv = object()
        layers.append(SimpleNamespace(layer_number=i + 1, self_attention=attn))
    cfg = SimpleNamespace(
        attention_output_gate=output_gate,
        multi_latent_attention=mla,
        num_query_groups=num_query_groups,
        q_lora_rank=q_lora_rank,
    )
    return SimpleNamespace(config=cfg, decoder=SimpleNamespace(layers=layers))


class TestArchitectureGuards:
    def test_plain_gqa_model_passes(self):
        model = _fake_model()
        _assert_supported_architecture(model.config)

    def test_output_gate_is_supported(self):
        """The gated query slice is handled by the permutation, not rejected."""
        model = _fake_model(output_gate=True)
        _assert_supported_architecture(model.config)

    def test_mla_is_supported(self):
        """MLA has its own projection set; the fused-qkv guards must not fire on it."""
        model = _fake_model(mla=True, with_qkv=False)
        _assert_supported_architecture(model.config, tp_size=2)

    def test_missing_linear_qkv_is_not_an_error(self):
        """Mixer-only layers carry no attention adapter; apply_native_lora reports them."""
        model = _fake_model(with_qkv=False)
        _assert_supported_architecture(model.config)

    def test_query_groups_below_tp_size_rejected(self):
        model = _fake_model(num_query_groups=2)
        with pytest.raises(AssertionError, match="num_query_groups"):
            _assert_supported_architecture(model.config, tp_size=4)

    def test_query_groups_equal_to_tp_size_passes(self):
        model = _fake_model(num_query_groups=4)
        _assert_supported_architecture(model.config, tp_size=4)

    def test_error_names_the_escape_hatch(self):
        model = _fake_model(num_query_groups=2)
        with pytest.raises(AssertionError, match="--lora-provider-path"):
            _assert_supported_architecture(model.config, tp_size=4)


class TestShippedRegistries:
    """Lock in which shipped registries (scripts/models/*.sh) the generic provider serves.

    Each may run --megatron-to-hf-mode raw, so a layout the generic path cannot
    slice must assert at startup naming --lora-provider-path rather than produce
    silently wrong gradients.
    """

    @pytest.mark.parametrize(
        "registry,kwargs",
        [
            ("glm4.7-flash", dict(mla=True)),
            ("kimi-k25_2layer", dict(mla=True)),
            ("glm5-744B-A40B_4layer", dict(mla=True)),
            # deepseek-v4-flash absent: wq_a/wq_b/wkv is not mcore MLA; registry fails it closed.
        ],
    )
    def test_mla_registries_are_accepted(self, registry, kwargs):
        """MLA is covered by MLAAttentionSpec.attach, including when TP exceeds the
        (meaningless for MLA) query-group count."""
        model = _fake_model(num_query_groups=2, **kwargs)
        _assert_supported_architecture(model.config, tp_size=4)

    def test_qwen3_5_gated_hybrid_is_accepted(self):
        """qwen3.5-35B-A3B.sh: --attention-output-gate plus GDN mixer layers.

        The gated query slice is permuted like any other, and a mixer layer simply
        carries no attention adapter, so TP <= num_query_groups is the only bar left.
        """
        model = _fake_model(output_gate=True, num_query_groups=2)
        _assert_supported_architecture(model.config, tp_size=2)

    def test_mla_without_q_lora_rank_is_rejected(self):
        """DeepSeek-V2-Lite / Moonlight: an uncompressed query path exports unfused
        q_proj + kv_a_proj_with_mqa, which SGLang's fused qkv_a loader cannot ingest.
        Every shipped MLA registry sets --q-lora-rank (scripts/models/*.sh), so only
        this uncovered layout is rejected.
        """
        model = _fake_model(mla=True, q_lora_rank=None)
        with pytest.raises(AssertionError) as excinfo:
            _assert_supported_architecture(model.config, tp_size=1)
        assert "q_lora_rank" in str(excinfo.value)
        assert "--lora-provider-path" in str(excinfo.value)

    def test_qwen3_5_above_query_group_count_still_rejected(self):
        model = _fake_model(output_gate=True, num_query_groups=2)
        with pytest.raises(AssertionError) as excinfo:
            _assert_supported_architecture(model.config, tp_size=4)
        message = str(excinfo.value)
        assert "num_query_groups" in message
        assert "--lora-provider-path" in message


class TestSupportedTargets:
    """`--target-modules` reaches the raw path in whichever spelling the user typed.

    Megatron-style names are what the bridge path accepts, so a run switched from
    bridge to raw keeps them. They used to fall through `_Spec.targets` unmatched,
    attaching nothing while SGLang still set up LoRA for the full list.
    """

    def test_architecture_specs_own_the_implemented_target_names(self):
        assert IMPLEMENTED_TARGETS == {
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "q_a_proj",
            "q_b_proj",
            "kv_a_proj_with_mqa",
            "kv_b_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        }

    def test_megatron_names_normalise_into_the_supported_set(self):
        for megatron_name in ("linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"):
            converted = set(convert_target_modules_to_hf([megatron_name]))
            assert converted
            assert converted <= IMPLEMENTED_TARGETS, (megatron_name, converted - IMPLEMENTED_TARGETS)

    def test_mla_megatron_names_normalise_too(self):
        assert set(convert_target_modules_to_hf(["linear_q_down_proj"])) <= IMPLEMENTED_TARGETS

    def test_hf_names_pass_through_unchanged(self):
        names = ["q_proj", "v_proj", "down_proj"]
        assert set(convert_target_modules_to_hf(names)) == set(names)

    def test_unimplemented_target_is_not_silently_accepted(self):
        assert "in_proj_qkvz" not in IMPLEMENTED_TARGETS


class TestInklingSpec:
    """Lock the Inkling registry entry's declared shape (attach is covered e2e)."""

    def _spec(self):
        from miles_plugins.lora.registry import MODEL_SPECS

        return MODEL_SPECS["inkling_mm_model"]

    def test_targets_cover_the_tml_projection_names(self):
        spec = self._spec()
        assert spec.attention.supported_targets == {"wq_du", "wk_dv", "wv_dv", "wr_du", "wo_ud"}
        assert spec.mlp.supported_targets == {"gate_up_proj", "down_proj"}

    def test_any_request_normalizes_to_the_full_native_set(self):
        spec = self._spec()
        normalized = spec.attention.normalize_targets(frozenset({"q_proj"}), expanded_from_all_linear=True)
        assert normalized == spec.attention.supported_targets | spec.mlp.supported_targets

    def test_block_prefixes_follow_tml_naming(self):
        spec = self._spec()
        assert spec.attention.layout.hf_block_prefix == "attn."
        assert spec.mlp.layout.hf_block_prefix == "mlp."

    def test_moe_and_lm_head_hooks_exist(self):
        spec = self._spec()
        assert callable(getattr(spec.moe, "attach", None))
        assert callable(getattr(spec.lm_head, "attach", None))
