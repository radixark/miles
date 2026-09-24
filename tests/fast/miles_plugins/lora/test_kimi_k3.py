"""Kimi K3 native LoRA on the plugin: export/import, merge, checkpoint keys, targets — no GPU."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from safetensors.torch import save_file

from miles.utils.lora.hf_lora_targets import resolve_hf_lora_targets
from miles.utils.lora.utils import get_adapter_target_modules, matches_lora_target
from miles_plugins.lora.config import LoRAConfig
from miles_plugins.lora.hf_adapter import export_lora_hf_named, load_lora_adapter_hf
from miles_plugins.lora.merge import merge_lora_into_weights
from miles_plugins.lora.modules import kimi_k3 as k3_modules
from miles_plugins.lora.modules.moe import _grouped_linear
from miles_plugins.lora.registry import resolve_adapter_targets
from miles_plugins.lora.sglang_adapter import export_lora_sglang_named
from miles_plugins.lora.spec.base import AttachContext
from miles_plugins.lora.spec.kimi_k3 import KimiK3AttentionSpec, KimiK3ExpertsSpec, KimiK3MLPSpec

HIDDEN, LATENT, RANK = 8, 6, 2
PREFIX = "language_model.model.layers."


class _Linear(nn.Module):
    def __init__(self, out_features, in_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))

    def forward(self, x):
        return x @ self.weight.t(), None


class _GroupedLinear(nn.Module):
    def __init__(self, num_experts, out_features, in_features):
        super().__init__()
        for index in range(num_experts):
            self.register_parameter(f"weight{index}", nn.Parameter(torch.randn(out_features, in_features)))

    def forward(self, x, tokens_per_expert):
        weights = torch.stack([getattr(self, f"weight{i}") for i in range(len(tokens_per_expert))])
        return _grouped_linear(x, weights, tokens_per_expert), None


def _attention(is_kda):
    attention = nn.Module()
    attention.is_kda = is_kda
    attention.tp_group = None
    attention.config = SimpleNamespace(hidden_size=HIDDEN)
    attention.o_proj = _Linear(HIDDEN, 12)
    if not is_kda:
        attention.q_lora_rank, attention.kv_lora_rank, attention.qk_extra_head_dim = 4, 5, 1
        attention.q_a_proj = _Linear(4, HIDDEN)
        attention.kv_a_proj_with_mqa = _Linear(6, HIDDEN)
    return attention


def _mlp(intermediate=5):
    mlp = nn.Module()
    mlp.config = SimpleNamespace(sequence_parallel=False, hidden_size=HIDDEN)
    mlp.tp_group = None
    mlp.linear_fc1 = _Linear(2 * intermediate, HIDDEN)
    mlp.linear_fc2 = _Linear(HIDDEN, intermediate)
    return mlp


def _moe(num_experts=3, intermediate=5):
    moe = nn.Module()
    moe.config = SimpleNamespace(
        moe_latent_size=LATENT, moe_ffn_hidden_size=intermediate, expert_tensor_parallel_size=1
    )
    moe.experts = nn.Module()
    moe.experts.num_local_experts = num_experts
    moe.experts.linear_fc1 = _GroupedLinear(num_experts, 2 * intermediate, LATENT)
    moe.experts.linear_fc2 = _GroupedLinear(num_experts, LATENT, intermediate)
    moe.shared_experts = _mlp()
    return moe


def _context(targets=None):
    return AttachContext(
        lora=LoRAConfig(rank=RANK, alpha=4, dropout=0.0, target_modules=targets),
        transformer_config=SimpleNamespace(hidden_size=HIDDEN, sequence_parallel=False),
        tp_size=1,
        tp_rank=0,
        layer_prefix=PREFIX,
        shared_expert="block_sparse_moe.shared_experts.",
    )


def _randomize(model):
    torch.manual_seed(0)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora" in name:
                parameter.normal_()


@pytest.fixture(autouse=True)
def _single_rank_comms(monkeypatch):
    from megatron.core import parallel_state
    from megatron.core.tensor_parallel import mappings

    monkeypatch.setattr(mappings, "reduce_from_tensor_model_parallel_region", lambda x, group=None: x)
    monkeypatch.setattr(parallel_state, "get_expert_model_parallel_rank", lambda: 0)


def _model(*, include_fc2=True):
    """Layer 3: MLA attention + dense MLP. Layer 4: KDA attention + routed and shared experts."""
    targets = [f"{PREFIX}*.block_sparse_moe.experts.*.w2"] if include_fc2 else []
    context = _context(tuple(targets) or ("unused",))
    model = nn.Module()
    model.mla, model.dense, model.kda, model.moe = _attention(False), _mlp(), _attention(True), _moe()
    KimiK3AttentionSpec().attach(model.mla, f"{PREFIX}3.self_attn.", context)
    KimiK3MLPSpec().attach(model.dense, f"{PREFIX}3.mlp.", context)
    KimiK3AttentionSpec().attach(model.kda, f"{PREFIX}4.self_attn.", context)
    KimiK3ExpertsSpec().attach(model.moe, f"{PREFIX}4.", context)
    KimiK3MLPSpec().attach(model.moe.shared_experts, f"{PREFIX}4.block_sparse_moe.shared_experts.", context)
    _randomize(model)
    return model


def _k3_hf_config():
    return dict(
        model_type="kimi_k3",
        text_config=dict(
            num_hidden_layers=5, first_k_dense_replace=4, moe_layer_freq=1, num_experts=3, num_shared_experts=1
        ),
    )


def _resolve(targets, **overrides):
    kwargs = dict(hf_modules=[], lora_type="lora", experts_shared_outer_loras=True) | overrides
    return resolve_adapter_targets(_k3_hf_config(), targets, **kwargs)


def test_parameter_names_match_the_original_integration():
    """Native Kimi K3 checkpoints key adapters by these names."""
    names = {name for name, _ in _model().named_parameters() if "lora" in name}
    assert {name for name in names if name.startswith("mla.")} == {
        f"mla.lora_adapter.{p}_lora_{f}" for p in ("o", "q_a", "kv_a") for f in "AB"
    }
    assert {name for name in names if name.startswith("dense.")} == {
        f"dense.lora_adapter.fc{i}_lora_{f}" for i in (1, 2) for f in "AB"
    }
    assert {name for name in names if name.startswith("moe.experts.")} == {
        f"moe.experts.lora_adapter.w{i}_lora_{f}" for i in (1, 2, 3) for f in "AB"
    }
    assert "moe.shared_experts.lora_adapter.fc1_lora_A" in names


@pytest.mark.parametrize("include_fc2", [True, False])
def test_export_names_and_shapes_match_sglang(include_fc2):
    """A wrong expert dim is a shape mismatch in SGLang's LoRA pool; a wrong HF name is silently dropped."""
    targets = resolve_hf_lora_targets(_k3_hf_config())
    if not include_fc2:
        targets = [target for target in targets if not target.endswith(".experts.*.w2")]
    exported = dict(export_lora_hf_named([_model(include_fc2=include_fc2)]))

    served = _resolve(targets)
    for module in get_adapter_target_modules(exported):
        assert any(matches_lora_target(module, target) for target in served), module
    assert exported[f"{PREFIX}3.self_attn.kv_a_proj_with_mqa.lora_B.weight"].shape == (6, RANK)
    assert exported[f"{PREFIX}4.self_attn.o_proj.lora_A.weight"].shape == (RANK, 12)
    assert f"{PREFIX}4.self_attn.q_a_proj.lora_A.weight" not in exported
    shared = f"{PREFIX}4.block_sparse_moe.shared_experts."
    assert torch.equal(exported[f"{shared}gate_proj.lora_A.weight"], exported[f"{shared}up_proj.lora_A.weight"])
    assert exported[f"{shared}gate_proj.lora_B.weight"].shape == (5, RANK)
    experts = f"{PREFIX}4.block_sparse_moe.experts."
    assert exported[f"{experts}w1.lora_A.weight"].shape == (1, RANK, LATENT)
    assert exported[f"{experts}w1.lora_B.weight"].shape == (3, 5, RANK)
    assert (f"{experts}w2.lora_B.weight" in exported) == include_fc2
    if include_fc2:
        assert exported[f"{experts}w2.lora_A.weight"].shape == (3, RANK, 5)
        assert exported[f"{experts}w2.lora_B.weight"].shape == (1, LATENT, RANK)


def test_serving_sync_exports_exactly_the_checkpoint_tensors():
    """Weight sync goes through the SGLang exporter; Kimi K3 has no fused serving siblings to zero-fill."""
    model = _model()
    assert dict(export_lora_sglang_named([model])).keys() == dict(export_lora_hf_named([model])).keys()


def test_hf_import_round_trips_the_export(tmp_path):
    source, target = _model(), _model()
    with torch.no_grad():
        for parameter in target.parameters():
            parameter.zero_()
    save_file(dict(export_lora_hf_named([source])), str(tmp_path / "adapter_model.safetensors"))

    load_lora_adapter_hf([target], str(tmp_path))

    source_params = dict(source.named_parameters())
    for name, parameter in target.named_parameters():
        if "lora" in name:
            exported_value = source_params[name].to(torch.bfloat16).to(parameter.dtype)
            torch.testing.assert_close(parameter, exported_value, msg=name)


def test_merged_weights_reproduce_the_adapter_forward():
    model = _model()
    weights = {name: parameter for name, parameter in model.named_parameters() if "lora" not in name}
    merged = merge_lora_into_weights([model], weights)
    for name, host in (
        ("mla.o_proj.weight", model.mla.o_proj),
        ("mla.q_a_proj.weight", model.mla.q_a_proj),
        ("dense.linear_fc1.weight", model.dense.linear_fc1),
        ("kda.o_proj.weight", model.kda.o_proj),
    ):
        inputs = torch.randn(4, host.weight.shape[1])
        torch.testing.assert_close(host(inputs)[0], inputs @ merged[name].t(), msg=name)
    tokens = [2, 1, 1]
    latent = torch.randn(4, LATENT)
    merged_fc1 = torch.stack([merged[f"moe.experts.linear_fc1.weight{i}"] for i in range(3)])
    torch.testing.assert_close(
        model.moe.experts.linear_fc1(latent, tokens)[0], _grouped_linear(latent, merged_fc1, tokens)
    )


@pytest.mark.parametrize("case", ["missing", "extra", "canonical", "per-expert"])
def test_target_contract_rejects_unverified_layouts(case):
    targets = resolve_hf_lora_targets(_k3_hf_config())
    overrides = {}
    if case == "missing":
        targets = [target for target in targets if not target.endswith(".q_a_proj")]
    elif case == "extra":
        targets.append(f"{PREFIX}*.self_attn.q_b_proj")
    elif case == "canonical":
        overrides["lora_type"] = "canonical_lora"
    else:
        overrides["experts_shared_outer_loras"] = False
    message = {
        "missing": r"missing=\[.*q_a_proj",
        "extra": r"unexpected=\[.*q_b_proj",
        "canonical": "canonical_lora",
        "per-expert": "shared-outer",
    }[case]
    with pytest.raises(AssertionError, match=message):
        _resolve(targets, **overrides)


def test_the_routed_expert_down_projection_is_optional():
    targets = [t for t in resolve_hf_lora_targets(_k3_hf_config()) if not t.endswith(".experts.*.w2")]
    served = _resolve(targets)
    assert f"{PREFIX}*.block_sparse_moe.experts.w1" in served
    assert not any(target.endswith(".w2") for target in served)


def test_grouped_linear_uses_expert_token_boundaries():
    """Wrong boundaries apply expert i's adapter to expert j's tokens without any error."""
    inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    weights = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])
    torch.testing.assert_close(_grouped_linear(inputs, weights, [1, 2]), torch.tensor([[1.0], [4.0], [6.0]]))


def test_k3_module_keeps_megatron_grad_flags():
    """TP-replicated partials are summed by Megatron; EP partials by reduce_marked_lora_grads."""
    model = _model()
    assert model.dense.lora_adapter.fc1_lora_A.sum_gradients_across_tp_domain
    assert model.moe.experts.lora_adapter.w1_lora_A._lora_grad_sum_group == "ep"
    assert model.moe.experts.lora_adapter.w1_lora_B.allreduce is False
    assert k3_modules.KimiK3ExpertsAdapter is type(model.moe.experts.lora_adapter)
