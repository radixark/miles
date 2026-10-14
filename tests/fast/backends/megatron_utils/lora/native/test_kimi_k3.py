"""Kimi K3 through the generic native-LoRA tables: export/import, merge, SP semantics, targets — no GPU."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from safetensors.torch import save_file
from tests.fast.backends.megatron_utils.lora.native.naming_stub import named

from miles.backends.megatron_utils.lora.native.config import LoRAConfig
from miles.backends.megatron_utils.lora.native.hf_adapter import export_lora_hf_named, load_lora_adapter_hf
from miles.backends.megatron_utils.lora.native.merge import merge_lora_into_weights
from miles.backends.megatron_utils.lora.native.modules.moe import _grouped_linear
from miles.backends.megatron_utils.lora.native.registry import MODEL_SPECS, resolve_adapter_targets
from miles.backends.megatron_utils.lora.native.spec.base import AttachContext
from miles.utils.lora.hf_lora_targets import resolve_hf_lora_targets
from miles.utils.lora.utils import get_adapter_target_modules, matches_lora_target

HIDDEN, LATENT, RANK = 8, 6, 2
PREFIX = "language_model.model.layers."
K3 = MODEL_SPECS["kimi_k3"]


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


def _attention(is_kda, *, tp_size=1):
    """Mirrors KimiK3Attention / KimiK3KDAAttention: KDA's o_proj is the linear-attention layer's
    head-sharded plain ``nn.Linear`` ``out_proj``, returning a bare TP partial sum."""
    attention = nn.Module()
    attention.is_kda = is_kda
    if is_kda:
        attention.linear_attn = nn.Module()
        attention.linear_attn.out_proj = nn.Linear(12 // tp_size, HIDDEN, bias=False)
    else:
        attention.o_proj = _Linear(HIDDEN, 12)
        attention.q_a_proj = _Linear(4, HIDDEN)
        attention.kv_a_proj_with_mqa = _Linear(6, HIDDEN)
    return attention


def _mlp(intermediate=5):
    mlp = nn.Module()
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


def _context(targets=None, *, sequence_parallel=False, tp_size=1, tp_rank=0):
    return AttachContext(
        lora=LoRAConfig(rank=RANK, alpha=4, dropout=0.0, target_modules=targets),
        transformer_config=SimpleNamespace(
            hidden_size=HIDDEN,
            sequence_parallel=sequence_parallel,
            q_lora_rank=4,
            kv_lora_rank=5,
            qk_pos_emb_head_dim=1,
        ),
        tp_size=tp_size,
        tp_rank=tp_rank,
        naming=None,
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
    monkeypatch.setattr(parallel_state, "get_expert_model_parallel_world_size", lambda: 2)


def _model(*, include_fc2=True, sequence_parallel=False):
    """Layer 3: MLA attention + dense MLP. Layer 4: KDA attention + routed and shared experts."""
    leaves = ("w1", "w3", "w2") if include_fc2 else ("w1", "w3")
    targets = [f"{PREFIX}*.block_sparse_moe.experts.*.{leaf}" for leaf in leaves]
    targets += [f"{PREFIX}*.{block}.{leaf}" for block, leaf in _NON_EXPERT_TARGETS]
    context = _context(tuple(targets), sequence_parallel=sequence_parallel)
    model = nn.Module()
    model.mla, model.dense, model.kda, model.moe = _attention(False), _mlp(), _attention(True), _moe()
    K3.attention.attach(model.mla, None, named(context, f"{PREFIX}3.self_attn."))
    K3.mlp.attach(model.dense, None, named(context, f"{PREFIX}3.mlp."))
    K3.attention.attach(model.kda, None, named(context, f"{PREFIX}4.self_attn."))
    K3.experts.attach(model.moe, None, named(context, f"{PREFIX}4.block_sparse_moe.experts."))
    K3.mlp.attach(model.moe.shared_experts, None, named(context, f"{PREFIX}4.block_sparse_moe.shared_experts."))
    _randomize(model)
    return model


_NON_EXPERT_TARGETS = [("self_attn", leaf) for leaf in ("q_a_proj", "kv_a_proj_with_mqa", "o_proj")] + [
    (block, leaf)
    for block in ("mlp", "block_sparse_moe.shared_experts")
    for leaf in ("gate_proj", "up_proj", "down_proj")
]


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


def test_k3_is_built_from_the_shared_tables():
    """Kimi K3 adds no modules of its own: every adapter is one the other architectures use."""
    adapters = {type(module).__name__ for module in _model().modules() if hasattr(module, "hf_prefix")}
    assert adapters == {"LoRALinear", "LoRAUnreducedRowLinear", "LoRASplitFC1", "LoRAGroupedFC1", "LoRAGroupedFC2"}


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
    assert exported[f"{shared}gate_proj.lora_B.weight"].shape == (5, RANK)
    experts = f"{PREFIX}4.block_sparse_moe.experts."
    assert exported[f"{experts}w1.lora_A.weight"].shape == (1, RANK, LATENT)
    assert exported[f"{experts}w1.lora_B.weight"].shape == (3, 5, RANK)
    assert (f"{experts}w2.lora_B.weight" in exported) == include_fc2
    if include_fc2:
        assert exported[f"{experts}w2.lora_A.weight"].shape == (3, RANK, 5)
        assert exported[f"{experts}w2.lora_B.weight"].shape == (1, LATENT, RANK)


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
        ("kda.linear_attn.out_proj.weight", model.kda.linear_attn.out_proj),
    ):
        inputs = torch.randn(4, host.weight.shape[1])
        output = host(inputs)
        output = output[0] if isinstance(output, tuple) else output
        torch.testing.assert_close(output, inputs @ merged[name].t(), msg=name)
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


def test_attention_runs_its_adapters_without_sequence_parallel():
    """K3 attention gathers the SP sequence itself: its row adapter all-reduces and its duplicated
    down projections see identical tokens on every rank, so neither reduce-scatters nor TP-sums grads."""
    model = _model(sequence_parallel=True)
    for adapter in (model.mla.lora_o_adapter, model.mla.lora_mla_q_a_adapter, model.mla.lora_mla_kv_a_adapter):
        assert not adapter.context.sequence_parallel
    assert not hasattr(model.mla.lora_mla_q_a_adapter.q_a_A, "_lora_grad_sum_group")
    assert not hasattr(model.mla.lora_o_adapter.o_B, "_lora_grad_sum_group")
    assert model.dense.lora_fc1_adapter.context.sequence_parallel
    assert model.dense.lora_fc2_adapter.down_B._lora_grad_sum_group == "tp"


def test_expert_grad_flags():
    """EP-replicated partials are summed by reduce_marked_lora_grads; expert params skip the DP all-reduce."""
    experts = _model().moe.experts
    assert experts.lora_fc1_adapter.w1_A._lora_grad_sum_group == "ep"
    assert experts.lora_fc2_adapter.w2_B._lora_grad_sum_group == "ep"
    assert experts.lora_fc1_adapter.w1_B.allreduce is False


def test_kda_o_proj_delta_stays_a_partial_sum(monkeypatch):
    """KDA's out_proj output is reduced by the layer after the adapter adds its delta: a reduce inside
    the adapter would apply it tp_size times. Summing the per-rank outputs must give W + s*B@A."""
    from megatron.core.tensor_parallel import mappings

    def _no_collective(*_args, **_kwargs):
        raise AssertionError("KDA o_proj LoRA must not reduce; the linear-attention layer does")

    monkeypatch.setattr(mappings, "reduce_from_tensor_model_parallel_region", _no_collective)
    monkeypatch.setattr(mappings, "reduce_scatter_to_sequence_parallel_region", _no_collective)
    tp_size, scale = 2, 4 / RANK
    torch.manual_seed(0)
    weight, lora_a, lora_b = torch.randn(HIDDEN, 12), torch.randn(RANK, 12), torch.randn(HIDDEN, RANK)
    inputs = torch.randn(4, 12)

    output = 0
    for tp_rank in range(tp_size):
        columns = slice(tp_rank * 12 // tp_size, (tp_rank + 1) * 12 // tp_size)
        attention = _attention(True, tp_size=tp_size)
        context = _context((f"{PREFIX}*.self_attn.o_proj",), tp_size=tp_size, tp_rank=tp_rank)
        assert K3.attention.attach(attention, None, named(context, f"{PREFIX}4.self_attn.")) == 1
        adapter = attention.linear_attn.lora_o_adapter
        assert adapter.o_B._lora_grad_sum_group == "tp"
        with torch.no_grad():
            attention.linear_attn.out_proj.weight.copy_(weight[:, columns])
            adapter.o_A.copy_(lora_a[:, columns])
            adapter.o_B.copy_(lora_b)
        output = output + attention.linear_attn.out_proj(inputs[:, columns])

    torch.testing.assert_close(output, inputs @ (weight + scale * lora_b @ lora_a).t())
