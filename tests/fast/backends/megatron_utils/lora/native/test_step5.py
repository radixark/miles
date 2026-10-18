"""Step5 native LoRA: the head-wise-gated ``linear_qkv`` and the shared-outer routed experts — shapes,
names, merge math, targets — no GPU."""

import json
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
from miles.backends.megatron_utils.lora.native.naming import ConverterNaming
from miles.backends.megatron_utils.lora.native.registry import (
    MODEL_SPECS,
    resolve_adapter_targets,
    resolve_config_spec,
)
from miles.backends.megatron_utils.lora.native.spec.base import AttachContext
from miles.utils.hf_utils.weight_mapping import build_hf_namespace_predicate
from miles.utils.lora.hf_lora_targets import exclude_hf_lora_targets, resolve_hf_lora_targets
from miles_plugins.models.step5.qkv_layout import fuse_qkv_and_gate, split_qkv_and_gate

HIDDEN, HEADS, GROUPS, HEAD_DIM, RANK, ALPHA = 16, 8, 4, 3, 2, 6
PREFIX = "model.layers.1.self_attn."
STEP5 = MODEL_SPECS["step5"]
SCALE = ALPHA / RANK


class _Linear(nn.Module):
    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.weight = nn.Parameter(weight.clone())

    def forward(self, x):
        return x @ self.weight.t(), None


@pytest.fixture(autouse=True)
def _single_process_tp(monkeypatch):
    """Each TP rank is built in this process; the column branch's TP-region copy is the identity forward."""
    from megatron.core.tensor_parallel import mappings

    monkeypatch.setattr(mappings, "copy_to_tensor_model_parallel_region", lambda x, group=None: x)


def _base_weights():
    torch.manual_seed(0)
    return {
        "q": torch.randn(HEADS * HEAD_DIM, HIDDEN),
        "k": torch.randn(GROUPS * HEAD_DIM, HIDDEN),
        "v": torch.randn(GROUPS * HEAD_DIM, HIDDEN),
        "g": torch.randn(HEADS, HIDDEN),
        "o": torch.randn(HIDDEN, HEADS * HEAD_DIM),
    }


def _hf_adapter():
    """Full HF adapter tensors as SGLang receives them: q/k/v/o only, B nonzero."""
    torch.manual_seed(1)
    rows = {"q_proj": HEADS * HEAD_DIM, "k_proj": GROUPS * HEAD_DIM, "v_proj": GROUPS * HEAD_DIM, "o_proj": HIDDEN}
    tensors = {}
    for leaf, out_rows in rows.items():
        in_features = HEADS * HEAD_DIM if leaf == "o_proj" else HIDDEN
        tensors[f"{PREFIX}{leaf}.lora_A.weight"] = torch.randn(RANK, in_features)
        tensors[f"{PREFIX}{leaf}.lora_B.weight"] = torch.randn(out_rows, RANK)
    return tensors


def _delta(adapter, leaf):
    return SCALE * adapter[f"{PREFIX}{leaf}.lora_B.weight"] @ adapter[f"{PREFIX}{leaf}.lora_A.weight"]


def _context(tp_size, tp_rank):
    return AttachContext(
        lora=LoRAConfig(rank=RANK, alpha=ALPHA, dropout=0.0, target_modules=None),
        transformer_config=SimpleNamespace(
            hidden_size=HIDDEN,
            sequence_parallel=False,
            num_query_groups=GROUPS,
            gated_linear_unit=True,
            head_wise_attn_gate=True,
            attention_output_gate=False,
        ),
        tp_size=tp_size,
        tp_rank=tp_rank,
        naming=None,
    )


def _rank_attention(base, tp_size, tp_rank):
    """One TP rank's Step5 attention: its block of the fused [q;k;v;gate] and its o_proj input columns."""
    fused = fuse_qkv_and_gate(
        query=base["q"], key=base["k"], value=base["v"], gate=base["g"], num_query_groups=GROUPS, tp_size=tp_size
    )
    attention = nn.Module()
    attention.linear_qkv = _Linear(fused.chunk(tp_size)[tp_rank])
    attention.linear_proj = _Linear(base["o"].chunk(tp_size, dim=1)[tp_rank])
    attention.num_attention_heads_per_partition = HEADS // tp_size
    attention.num_query_groups_per_partition = GROUPS // tp_size
    attention.hidden_size_per_attention_head = HEAD_DIM
    model = nn.Module()
    model.attn = attention
    context = _context(tp_size, tp_rank)
    STEP5.validate(context)
    assert STEP5.attention.attach(attention, None, named(context, PREFIX)) == 2
    return model


@pytest.mark.parametrize("tp_size", [1, 2, 4])
def test_adapters_cover_qkv_and_leave_the_gate_frozen(tmp_path, tp_size):
    base, adapter = _base_weights(), _hf_adapter()
    save_file(adapter, str(tmp_path / "adapter_model.safetensors"))
    ranks = [_rank_attention(base, tp_size, rank) for rank in range(tp_size)]

    merged_qkv, merged_o = [], []
    for rank, model in enumerate(ranks):
        qkv_adapter = model.attn.lora_qkv_adapter
        assert {name for name, _ in qkv_adapter.named_parameters()} == {
            f"{p}_{m}" for p in "qkv" for m in "AB"
        }, "the gate rows must carry no adapter"
        assert qkv_adapter.q_B.shape == (HEADS // tp_size * HEAD_DIM, RANK)
        assert qkv_adapter.k_B.shape == qkv_adapter.v_B.shape == (GROUPS // tp_size * HEAD_DIM, RANK)
        assert model.attn.lora_o_adapter.o_A.shape == (RANK, HEADS // tp_size * HEAD_DIM)
        assert load_lora_adapter_hf([model], str(tmp_path)) == 8

        weights = {name: param for name, param in model.named_parameters() if "lora" not in name}
        merged = merge_lora_into_weights([model], weights)
        merged_qkv.append(merged["attn.linear_qkv.weight"])
        merged_o.append(merged["attn.linear_proj.weight"])

        x = torch.randn(5, HIDDEN)
        with torch.no_grad():
            torch.testing.assert_close(
                model.attn.linear_qkv(x)[0], x @ merged_qkv[-1].t(), rtol=1e-5, atol=1e-5, msg=f"rank {rank}"
            )

    # The converter's split of the gathered merged weight is base + delta under each HF name.
    query, key, value, gate = split_qkv_and_gate(
        torch.cat(merged_qkv),
        num_query_groups=GROUPS,
        num_attention_heads=HEADS,
        head_dim=HEAD_DIM,
        tp_size=tp_size,
    )
    torch.testing.assert_close(query, base["q"] + _delta(adapter, "q_proj"))
    torch.testing.assert_close(key, base["k"] + _delta(adapter, "k_proj"))
    torch.testing.assert_close(value, base["v"] + _delta(adapter, "v_proj"))
    assert torch.equal(gate, base["g"])
    torch.testing.assert_close(torch.cat(merged_o, dim=1), base["o"] + _delta(adapter, "o_proj"))


def test_export_names_match_the_hf_projections(tmp_path):
    base, adapter = _base_weights(), _hf_adapter()
    save_file(adapter, str(tmp_path / "adapter_model.safetensors"))
    model = _rank_attention(base, tp_size=1, tp_rank=0)
    load_lora_adapter_hf([model], str(tmp_path))

    exported = dict(export_lora_hf_named([model]))
    assert set(exported) == set(adapter)
    for name, tensor in exported.items():
        torch.testing.assert_close(tensor.float(), adapter[name], rtol=1e-2, atol=1e-2)


def test_the_host_layout_is_checked():
    """A linear_qkv without the gate rows (an ungated GQA) is not Step5's layout."""
    attention = nn.Module()
    attention.linear_qkv = _Linear(torch.randn((HEADS + 2 * GROUPS) * HEAD_DIM, HIDDEN))
    attention.linear_proj = _Linear(torch.randn(HIDDEN, HEADS * HEAD_DIM))
    attention.num_attention_heads_per_partition = HEADS
    attention.num_query_groups_per_partition = GROUPS
    attention.hidden_size_per_attention_head = HEAD_DIM
    with pytest.raises(AssertionError, match="local rows"):
        STEP5.attention.attach(attention, None, named(_context(1, 0), PREFIX))


def _layer(layer_number, **weights):
    layer = nn.Module()
    layer.layer_number = layer_number
    for path, (shape, partition_dim) in weights.items():
        *modules, leaf = path.split("__")
        owner = layer
        for name in modules:
            if not hasattr(owner, name):
                owner.add_module(name, nn.Module())
            owner = getattr(owner, name)
        weight = nn.Parameter(torch.empty(shape))
        weight.tensor_model_parallel = partition_dim is not None
        weight.partition_dim = -1 if partition_dim is None else partition_dim
        owner.register_parameter(leaf, weight)
    return layer


def test_converter_names_qkv_gate_dense_and_shared_expert_hosts():
    args = SimpleNamespace(
        num_query_groups=GROUPS, num_attention_heads=HEADS, kv_channels=HEAD_DIM, tensor_model_parallel_size=2
    )
    qkv_rows = (HEADS + 2 * GROUPS) * HEAD_DIM + HEADS
    layer = _layer(
        2,
        self_attention__linear_qkv__weight=((qkv_rows // 2, HIDDEN), 0),
        self_attention__linear_proj__weight=((HIDDEN, HEADS * HEAD_DIM // 2), 1),
        mlp__shared_experts__linear_fc1__weight=((6, HIDDEN), 0),
        mlp__shared_experts__linear_fc2__weight=((HIDDEN, 3), 1),
    )
    dense = _layer(1, mlp__linear_fc1__weight=((10, HIDDEN), 0))
    naming = ConverterNaming(args, "step5", tp_size=2, is_hf_name=lambda _name: True)

    def prefix(layer, path, leaves):
        owner = layer
        for name in path.split("__"):
            owner = getattr(owner, name)
        return naming.block_prefix(layer, owner, leaves)

    leaves = [p.hf_leaf for p in STEP5.attention.layout.fused[0].projections]
    qkv_leaves = [*leaves, *STEP5.attention.layout.fused[0].unadapted_leaves]
    assert prefix(layer, "self_attention__linear_qkv__weight", qkv_leaves) == "model.layers.1.self_attn."
    with pytest.raises(AssertionError, match="g_proj"):
        prefix(layer, "self_attention__linear_qkv__weight", leaves)
    assert prefix(layer, "self_attention__linear_proj__weight", ["o_proj"]) == "model.layers.1.self_attn."
    assert prefix(layer, "mlp__shared_experts__linear_fc1__weight", ["gate_proj", "up_proj"]) == (
        "model.layers.1.share_expert."
    )
    assert prefix(layer, "mlp__shared_experts__linear_fc2__weight", ["down_proj"]) == "model.layers.1.share_expert."
    assert prefix(dense, "mlp__linear_fc1__weight", ["gate_proj", "up_proj"]) == "model.layers.0.mlp."


def _step5_vl_config(moe_layer_list=(1, 2, 3)):
    return {
        "model_type": "step5_vl",
        "text_config": {"model_type": "step5", "num_hidden_layers": 5, "moe_layer_list": list(moe_layer_list)},
    }


def _resolve(hf_config, targets, *, experts_shared_outer_loras=True):
    return resolve_adapter_targets(
        hf_config, targets, hf_modules=[], lora_type="lora", experts_shared_outer_loras=experts_shared_outer_loras
    )


_LEAVES = ("gate_proj", "up_proj", "down_proj")
_ROUTED_TARGETS = [f"model.layers.*.moe.experts.*.{leaf}" for leaf in _LEAVES]


def test_all_linear_adds_the_routed_experts_to_attention_dense_mlp_and_shared_expert():
    config = _step5_vl_config()
    assert resolve_config_spec(config) == ("step5", STEP5)
    targets = resolve_hf_lora_targets(config, target_modules=["all-linear"])
    non_expert = [
        *(f"model.layers.*.self_attn.{leaf}" for leaf in ("q_proj", "k_proj", "v_proj", "o_proj")),
        *(f"model.layers.*.mlp.{leaf}" for leaf in _LEAVES),
        *(f"model.layers.*.share_expert.{leaf}" for leaf in _LEAVES),
    ]
    assert targets == [*non_expert[:7], *_ROUTED_TARGETS, *non_expert[7:]]
    # SGLang addresses the shared-outer expert tensors without an expert index
    assert _resolve(config, targets) == [
        *non_expert[:7],
        *(f"model.layers.*.moe.experts.{leaf}" for leaf in _LEAVES),
        *non_expert[7:],
    ]
    assert resolve_hf_lora_targets(_step5_vl_config(moe_layer_list=range(5)), target_modules=["mlp"]) == [
        *_ROUTED_TARGETS,
        *(f"model.layers.*.share_expert.{leaf}" for leaf in _LEAVES),
    ]


def test_excluding_the_routed_experts_keeps_the_frozen_expert_layout():
    targets = exclude_hf_lora_targets(
        resolve_hf_lora_targets(_step5_vl_config(), target_modules=["all-linear"]), ["model.layers.*.moe.experts.*"]
    )
    assert not any(".experts." in target for target in targets) and len(targets) == 10
    assert _resolve(_step5_vl_config(), targets, experts_shared_outer_loras=False) == targets


def test_routed_experts_need_the_shared_outer_layout():
    with pytest.raises(AssertionError, match="experts-shared-outer-loras"):
        _resolve(_step5_vl_config(), _ROUTED_TARGETS, experts_shared_outer_loras=False)


def test_routed_gate_and_up_are_selected_together():
    with pytest.raises(AssertionError, match="select both or neither"):
        _resolve(_step5_vl_config(), ["model.layers.*.moe.experts.*.gate_proj"])
    assert _resolve(_step5_vl_config(), ["model.layers.*.moe.experts.*.down_proj"]) == [
        "model.layers.*.moe.experts.down_proj"
    ]


def test_the_gate_is_rejected():
    with pytest.raises(AssertionError, match="does not implement adapters"):
        _resolve(_step5_vl_config(), ["model.layers.*.self_attn.g_proj"])


def test_converter_names_the_grouped_expert_hosts(tmp_path):
    """The converter names each local expert ``moe.experts.E.*``; the packed checkpoint is ``moe.*_proj``."""
    layer = nn.Module()
    layer.layer_number = 4
    layer.mlp = nn.Module()
    layer.mlp.experts = nn.Module()
    for host, shape in (("linear_fc1", (6, HIDDEN)), ("linear_fc2", (HIDDEN, 3))):
        module = nn.Module()
        weight = nn.Parameter(torch.empty(shape))
        weight.tensor_model_parallel, weight.partition_dim, weight.allreduce = True, 0, False
        module.register_parameter("weight0", weight)
        layer.mlp.experts.add_module(host, module)

    weight_map = {f"model.layers.3.moe.{leaf}.weight": "model.safetensors" for leaf in _LEAVES}
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "step5", "num_hidden_layers": 5}))
    is_hf_name = build_hf_namespace_predicate(str(tmp_path))
    assert is_hf_name("model.layers.3.moe.experts.7.down_proj.weight")
    assert not is_hf_name("model.layers.3.moe.experts.7.w2.weight")

    args = SimpleNamespace(tensor_model_parallel_size=2)
    naming = ConverterNaming(args, "step5", tp_size=2, is_hf_name=is_hf_name)
    experts = layer.mlp.experts
    assert naming.block_prefix(layer, experts.linear_fc1.weight0, ["gate_proj", "up_proj"]) == (
        "model.layers.3.moe.experts."
    )
    assert naming.block_prefix(layer, experts.linear_fc2.weight0, ["down_proj"]) == "model.layers.3.moe.experts."


NUM_EXPERTS, MOE_INTER, EXPERT_PREFIX = 4, 3, "model.layers.1.moe.experts."
SWIGLU_LIMIT = 7.0


class _GroupedLinear(nn.Module):
    """TEGroupedLinear's interface: ``weight<i>`` per local expert, tokens grouped by expert."""

    def __init__(self, weights):
        super().__init__()
        for index, weight in enumerate(weights):
            self.register_parameter(f"weight{index}", nn.Parameter(weight.clone()))

    def forward(self, x, tokens_per_expert):
        weights = torch.stack([getattr(self, f"weight{i}") for i in range(len(tokens_per_expert))])
        return _grouped_linear(x, weights, tokens_per_expert), None


def _expert_base():
    torch.manual_seed(2)
    return {
        "fc1": torch.randn(NUM_EXPERTS, 2 * MOE_INTER, HIDDEN),
        "fc2": torch.randn(NUM_EXPERTS, HIDDEN, MOE_INTER),
    }


def _expert_adapter(b_std=1.0):
    """The shared-outer HF tensors SGLang loads: one A per gate/up and one B for down, shared by all experts."""
    torch.manual_seed(3)
    tensors = {}
    for leaf in ("gate_proj", "up_proj"):
        tensors[f"{EXPERT_PREFIX}{leaf}.lora_A.weight"] = torch.randn(1, RANK, HIDDEN)
        tensors[f"{EXPERT_PREFIX}{leaf}.lora_B.weight"] = b_std * torch.randn(NUM_EXPERTS, MOE_INTER, RANK)
    tensors[f"{EXPERT_PREFIX}down_proj.lora_A.weight"] = torch.randn(NUM_EXPERTS, RANK, MOE_INTER)
    tensors[f"{EXPERT_PREFIX}down_proj.lora_B.weight"] = b_std * torch.randn(1, HIDDEN, RANK)
    return tensors


def _expert_delta(adapter, leaf, expert):
    a, b = adapter[f"{EXPERT_PREFIX}{leaf}.lora_A.weight"], adapter[f"{EXPERT_PREFIX}{leaf}.lora_B.weight"]
    return SCALE * b[expert if b.shape[0] > 1 else 0] @ a[expert if a.shape[0] > 1 else 0]


@pytest.fixture
def _ep(monkeypatch):
    """Set the expert-parallel rank and size each simulated rank sees."""
    from megatron.core import parallel_state

    def set_ep(ep_rank, ep_size):
        monkeypatch.setattr(parallel_state, "get_expert_model_parallel_rank", lambda: ep_rank)
        monkeypatch.setattr(parallel_state, "get_expert_model_parallel_world_size", lambda: ep_size)

    return set_ep


def _rank_moe(base, ep_rank, ep_size):
    """One EP rank's Step5 MoE block: its contiguous slice of the routed experts."""
    local = slice(ep_rank * NUM_EXPERTS // ep_size, (ep_rank + 1) * NUM_EXPERTS // ep_size)
    moe = nn.Module()
    moe.config = SimpleNamespace(moe_ffn_hidden_size=MOE_INTER)
    moe.experts = nn.Module()
    moe.experts.num_local_experts = NUM_EXPERTS // ep_size
    moe.experts.linear_fc1 = _GroupedLinear(base["fc1"][local])
    moe.experts.linear_fc2 = _GroupedLinear(base["fc2"][local])
    assert STEP5.experts.attach(moe, None, named(_context(1, 0), EXPERT_PREFIX)) == 2
    return moe, local


def _clamped_swiglu(fc1_output):
    """Step5's expert activation: ``min(silu(gate), 7) * clamp(up, -7, 7)``."""
    gate, up = fc1_output.chunk(2, dim=-1)
    return torch.nn.functional.silu(gate).clamp(max=SWIGLU_LIMIT) * up.clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT)


@pytest.mark.parametrize("ep_size", [1, 2, 4])
def test_expert_adapters_shapes_load_and_merge_per_ep_rank(tmp_path, _ep, ep_size):
    base, adapter = _expert_base(), _expert_adapter()
    save_file(adapter, str(tmp_path / "adapter_model.safetensors"))
    for ep_rank in range(ep_size):
        _ep(ep_rank, ep_size)
        moe, local = _rank_moe(base, ep_rank, ep_size)
        fc1, fc2 = moe.experts.lora_fc1_adapter, moe.experts.lora_fc2_adapter
        num_local = NUM_EXPERTS // ep_size
        assert fc1.w1_A.shape == fc1.w3_A.shape == (RANK, HIDDEN)
        assert fc1.w1_B.shape == fc1.w3_B.shape == (num_local, MOE_INTER, RANK)
        assert fc2.w2_A.shape == (num_local, RANK, MOE_INTER) and fc2.w2_B.shape == (HIDDEN, RANK)
        # under EP the shared A sums its grads over EP and every adapter skips the DP all-reduce
        is_ep = ep_size > 1
        assert (getattr(fc1.w1_A, "_lora_grad_sum_group", None) == "ep") == is_ep
        assert (getattr(fc1.w1_B, "allreduce", True) is False) == is_ep
        assert load_lora_adapter_hf([moe], str(tmp_path)) == 6

        weights = {name: param for name, param in moe.named_parameters() if "lora" not in name}
        merged = merge_lora_into_weights([moe], weights)
        for local_index, expert in enumerate(range(NUM_EXPERTS)[local]):
            # the converter's gate/up split of linear_fc1 is base + delta under each HF name
            gate, up = merged[f"experts.linear_fc1.weight{local_index}"].chunk(2, dim=0)
            base_gate, base_up = base["fc1"][expert].chunk(2, dim=0)
            torch.testing.assert_close(gate, base_gate + _expert_delta(adapter, "gate_proj", expert))
            torch.testing.assert_close(up, base_up + _expert_delta(adapter, "up_proj", expert))
            torch.testing.assert_close(
                merged[f"experts.linear_fc2.weight{local_index}"],
                base["fc2"][expert] + _expert_delta(adapter, "down_proj", expert),
            )


def test_expert_export_round_trips_under_hf_names(tmp_path, _ep):
    _ep(0, 1)
    adapter = _expert_adapter()
    save_file(adapter, str(tmp_path / "adapter_model.safetensors"))
    moe, _ = _rank_moe(_expert_base(), 0, 1)
    load_lora_adapter_hf([moe], str(tmp_path))

    exported = dict(export_lora_hf_named([moe]))
    assert set(exported) == set(adapter)
    for name, tensor in exported.items():
        assert tensor.shape == adapter[name].shape, name
        torch.testing.assert_close(tensor.float(), adapter[name], rtol=1e-2, atol=1e-2)


def test_the_gate_up_delta_lands_before_the_clamped_swiglu(tmp_path, _ep):
    """The trainer's experts compute fc2(swiglu_clamp(fc1(x) + delta1)) + delta2, as SGLang's MoE-LoRA hooks
    do; with the clamp active, adding the gate/up delta after the activation would give a different output."""
    _ep(0, 1)
    base, adapter = _expert_base(), _expert_adapter(b_std=4.0)
    save_file(adapter, str(tmp_path / "adapter_model.safetensors"))
    moe, _ = _rank_moe(base, 0, 1)
    load_lora_adapter_hf([moe], str(tmp_path))
    tokens = [3, 1, 0, 2]
    x = 3 * torch.randn(sum(tokens), HIDDEN)

    with torch.no_grad():
        fc1_output = moe.experts.linear_fc1(x, tokens)[0]
        output = moe.experts.linear_fc2(_clamped_swiglu(fc1_output), tokens)[0]

    expected, wrong_order, row = [], [], 0
    for expert, count in enumerate(tokens):
        rows = x[row : row + count]
        row += count
        w1 = base["fc1"][expert]
        delta1 = torch.cat([_expert_delta(adapter, leaf, expert) for leaf in ("gate_proj", "up_proj")])
        w2 = base["fc2"][expert] + _expert_delta(adapter, "down_proj", expert)
        expected.append(_clamped_swiglu(rows @ (w1 + delta1).t()) @ w2.t())
        wrong_order.append((_clamped_swiglu(rows @ w1.t()) + rows @ delta1[:MOE_INTER].t()) @ w2.t())
    assert (fc1_output.abs() > SWIGLU_LIMIT).any(), "the clamp must be active for this test to mean anything"
    torch.testing.assert_close(output, torch.cat(expected), rtol=1e-4, atol=1e-4)
    assert not torch.allclose(output, torch.cat(wrong_order), rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("expert", [0, 7], ids=["first-expert", "other-expert"])
def test_shared_outer_adapters_reject_single_expert_targets(expert):
    targets = [f"model.layers.3.moe.experts.{expert}.{leaf}" for leaf in _LEAVES]
    with pytest.raises(AssertionError, match="covers all routed experts"):
        _resolve(_step5_vl_config(), targets)


def test_routed_gate_and_up_must_pair_within_the_same_layer():
    targets = ["model.layers.3.moe.experts.*.gate_proj", "model.layers.4.moe.experts.*.up_proj"]
    with pytest.raises(AssertionError, match="select both or neither"):
        _resolve(_step5_vl_config(), targets)
