"""Unit tests for native (raw-mode) LoRA helpers — no GPU, no distributed init.

Covers the qkv output permutation, the provider-protocol resolver, architecture
guards, the per-rank adapter shard naming, and the rollout gate.
"""

from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch

from miles.backends.megatron_utils.lora_utils import (
    _adapter_shard_name,
    _is_canonical_shard_writer,
    convert_target_modules_to_hf,
    reduce_marked_lora_grads,
    resolve_lora_provider,
)
from miles.utils.lora import lora_rollout_enabled
from miles_plugins.lora.distributed import rmsnorm
from miles_plugins.lora.hf_adapter import resolve_hf_naming
from miles_plugins.lora.lora import (
    _require_grad_on_first_activation,
    export_lora_hf_named,
    load_lora_adapter_hf,
    wrap_model_provider_with_lora,
)
from miles_plugins.lora.modules.linear import build_qkv_permutation
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


class TestBuildQkvPerm:
    def test_mha_single_group(self):
        perm = build_qkv_permutation(num_q_heads=1, num_groups=1, head_dim=2, device="cpu")
        assert perm.tolist() == [0, 1, 2, 3, 4, 5]

    def test_gqa_two_groups_matches_mcore_layout(self):
        perm = build_qkv_permutation(num_q_heads=4, num_groups=2, head_dim=1, device="cpu")
        assert perm.tolist() == [0, 1, 4, 6, 2, 3, 5, 7]

    def test_permutation_is_a_bijection(self):
        nq, ng, hd = 8, 4, 3
        perm = build_qkv_permutation(num_q_heads=nq, num_groups=ng, head_dim=hd, device="cpu")
        total = (nq + 2 * ng) * hd
        assert perm.numel() == total
        assert sorted(perm.tolist()) == list(range(total))

    def test_applied_to_delta_places_projections_per_group(self):
        nq, ng, hd = 4, 2, 1
        perm = build_qkv_permutation(num_q_heads=nq, num_groups=ng, head_dim=hd, device="cpu")
        plain = torch.tensor([[10.0, 11.0, 12.0, 13.0, 20.0, 21.0, 30.0, 31.0]])
        out = plain.index_select(-1, perm)
        assert out.tolist() == [[10.0, 11.0, 20.0, 30.0, 12.0, 13.0, 21.0, 31.0]]

    def test_output_gate_deinterleaves_the_query_slices(self):
        perm = build_qkv_permutation(num_q_heads=2, num_groups=1, head_dim=1, device="cpu", output_gate=True)
        assert perm.tolist() == [0, 2, 1, 3, 4, 5]

    def test_output_gate_permutation_is_a_bijection(self):
        nq, ng, hd = 8, 2, 3
        perm = build_qkv_permutation(num_q_heads=nq, num_groups=ng, head_dim=hd, device="cpu", output_gate=True)
        total = (2 * nq + 2 * ng) * hd
        assert perm.numel() == total
        assert sorted(perm.tolist()) == list(range(total))

    def test_output_gate_applied_to_delta(self):
        perm = build_qkv_permutation(num_q_heads=4, num_groups=2, head_dim=1, device="cpu", output_gate=True)
        plain = torch.tensor([[10.0, 40.0, 11.0, 41.0, 12.0, 42.0, 13.0, 43.0, 20.0, 21.0, 30.0, 31.0]])
        out = plain.index_select(-1, perm)
        assert out.tolist() == [[10.0, 11.0, 40.0, 41.0, 20.0, 30.0, 12.0, 13.0, 42.0, 43.0, 21.0, 31.0]]


class TestRmsNorm:
    def test_plain_gamma_scales_by_the_stored_weight(self):
        x = torch.tensor([[3.0, 4.0]])
        gamma = torch.tensor([2.0, 2.0])
        got = rmsnorm(x, gamma, eps=0.0)
        assert torch.allclose(got, torch.tensor([[3.0, 4.0]]) / 3.5355339 * 2.0, atol=1e-5)

    def test_zero_centered_gamma_adds_the_one_back(self):
        """--apply-layernorm-1p stores gamma - 1; the branch must see the same input
        the base GEMM does, or the adapter is fed a differently scaled activation."""
        x = torch.tensor([[3.0, 4.0]])
        stored = torch.tensor([1.0, 1.0])
        assert torch.allclose(
            rmsnorm(x, stored, eps=0.0, zero_centered_gamma=True),
            rmsnorm(x, stored + 1.0, eps=0.0),
        )


class TestFirstActivationGrad:
    """A frozen base plus recomputation is the case that silently trains nothing.

    Every adapter param sits inside a checkpointed block, so unless the block's
    input requires grad, autograd never enters the region and every adapter
    gradient comes back zero while all the sync checks still pass.
    """

    def test_a_frozen_embedding_output_has_no_graph_on_its_own(self):
        embedding = torch.nn.Embedding(4, 3)
        embedding.weight.requires_grad_(False)
        assert not embedding(torch.tensor([0, 1])).requires_grad

    def test_hook_makes_the_first_activation_require_grad(self):
        embedding = torch.nn.Embedding(4, 3)
        embedding.weight.requires_grad_(False)
        model = SimpleNamespace(embedding=embedding)
        assert _require_grad_on_first_activation(model) is embedding
        assert embedding(torch.tensor([0, 1])).requires_grad

    def test_stage_without_an_embedding_is_a_noop(self):
        assert _require_grad_on_first_activation(SimpleNamespace()) is None


def _write_index(tmp_path, keys):
    import json

    weight_map = {key: "a.safetensors" for key in keys}
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    return str(tmp_path)


class TestHfNaming:
    def test_deepseek_style_plural_shared_expert(self, tmp_path):
        path = _write_index(
            tmp_path,
            [
                "model.layers.0.self_attn.o_proj.weight",
                "model.layers.1.mlp.shared_experts.gate_proj.weight",
            ],
        )
        assert resolve_hf_naming(path) == ("model.layers.", "mlp.shared_experts.")

    def test_qwen3_5_nests_the_decoder_and_uses_singular(self, tmp_path):
        """The mtp block also has `layers.N.`; it must not win the prefix vote."""
        path = _write_index(
            tmp_path,
            [
                "model.language_model.layers.0.self_attn.q_proj.weight",
                "model.language_model.layers.1.mlp.shared_expert.up_proj.weight",
                "mtp.layers.0.mlp.shared_expert.gate_proj.weight",
                "vision_tower.encoder.blocks.0.wo.weight",
            ],
        )
        assert resolve_hf_naming(path) == ("model.language_model.layers.", "mlp.shared_expert.")

    def test_missing_index_falls_back_to_the_plain_layout(self, tmp_path):
        assert resolve_hf_naming(str(tmp_path)) == ("model.layers.", "mlp.shared_expert.")
        assert resolve_hf_naming(None) == ("model.layers.", "mlp.shared_expert.")


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


class TestResolveLoraProvider:
    def test_default_is_the_plugin(self):
        mod = resolve_lora_provider(Namespace())
        assert mod.wrap_model_provider_with_lora is wrap_model_provider_with_lora
        assert mod.export_lora_hf_named is export_lora_hf_named
        assert mod.load_lora_adapter_hf is load_lora_adapter_hf

    @pytest.mark.parametrize("path", ["miles_plugins.lora.lora", "miles.backends.megatron_utils.lora_native"])
    def test_native_provider_paths_are_imported(self, path):
        provider = resolve_lora_provider(Namespace(lora_provider_path=path))
        assert provider.wrap_model_provider_with_lora is wrap_model_provider_with_lora
        assert provider.export_lora_hf_named is export_lora_hf_named

    def test_module_without_protocol_is_rejected(self):
        args = Namespace(lora_provider_path="json")
        with pytest.raises(AssertionError, match="wrap_model_provider_with_lora"):
            resolve_lora_provider(args)


class TestWrapModelProvider:
    def test_provider_args_are_forwarded_and_result_wrapped(self):
        seen = {}

        def provider(pre_process, post_process, vp_stage=None):
            seen.update(pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)
            return _fake_model()

        calls = []
        wrapped = wrap_model_provider_with_lora(provider, Namespace(lora_rank=8))
        import miles_plugins.lora.lora as ln

        orig = ln.apply_native_lora
        ln.apply_native_lora = lambda m, a: calls.append((m, a)) or m
        try:
            out = wrapped(True, False, vp_stage=1)
        finally:
            ln.apply_native_lora = orig

        assert seen == {"pre_process": True, "post_process": False, "vp_stage": 1}
        assert out is calls[0][0]


class TestAdapterShardName:
    def test_native_name_is_ep_invariant(self):
        """Routed experts carry no native adapter and the shared expert shards over attention TP,
        so every EP rank holds identical state for a given (tp, pp) — one file serves them all."""
        names = {_adapter_shard_name(1, 2, ep, ep_sharded=False) for ep in range(4)}
        assert names == {"adapter_megatron_tp1_pp2.pt"}

    def test_bridge_name_keys_on_ep(self):
        """Bridge PEFT can attach genuinely expert-parallel adapters, so its shards differ per EP rank."""
        assert _adapter_shard_name(1, 2, 0, ep_sharded=True) == "adapter_megatron_tp1_pp2.pt"
        assert _adapter_shard_name(1, 2, 3, ep_sharded=True) == "adapter_megatron_tp1_pp2_ep3.pt"
        names = {_adapter_shard_name(0, 0, ep, ep_sharded=True) for ep in range(4)}
        assert len(names) == 4

    def test_writer_election_is_a_noop_without_distributed(self):
        assert _is_canonical_shard_writer("adapter_megatron_tp0_pp0.pt")


class TestReduceMarkedLoraGrads:
    def test_no_marked_params_is_a_noop_without_distributed(self):
        chunk = torch.nn.Linear(2, 2)
        reduce_marked_lora_grads([chunk])

    def test_empty_model_list_is_a_noop(self):
        reduce_marked_lora_grads([])


class TestLoraRolloutEnabled:
    def test_enabled_when_lora_on_and_not_train_only(self):
        assert lora_rollout_enabled(Namespace(lora_rank=16, debug_lora_train_only=False))

    def test_disabled_under_train_only(self):
        assert not lora_rollout_enabled(Namespace(lora_rank=16, debug_lora_train_only=True))

    def test_disabled_without_lora(self):
        assert not lora_rollout_enabled(Namespace(lora_rank=0, debug_lora_train_only=False))

    def test_missing_train_only_attr_defaults_to_enabled(self):
        assert lora_rollout_enabled(Namespace(lora_rank=8))


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
