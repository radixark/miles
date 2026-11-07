import inspect
from dataclasses import replace
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from megatron.core import dist_checkpointing
from megatron.core.dist_checkpointing.mapping import ShardedTensor
from safetensors.torch import save_file
from tests.fast.dist_utils import init_gloo, run_multiprocess

from miles.backends.megatron_utils.lora import bridge_checkpoint_loader
from miles.backends.megatron_utils.lora.bridge_adapter_slots import BridgeAdapterSlots, probe_bridge_row_mapping
from miles.backends.megatron_utils.lora.bridge_checkpoint_loader import (
    _BRIDGE_BASE_REMEDY,
    assert_bridge_run_config_matches,
    assert_same_base_as_torch_dist,
    bridge_base_model_candidates,
    torch_dist_base_dir,
)
from miles.backends.training_utils.checkpoint.adapter.base_model_check import assert_same_base_model
from miles.backends.training_utils.checkpoint.adapter.parametrization import AdapterParametrization
from miles.backends.training_utils.checkpoint.adapter.probed_slots import HfPiece, ProbedAdapterSlots, RowMapping
from miles.tinker.core.utils import build_parametrization_fields
from miles.utils import distributed_utils
from miles.utils.distributed_utils import init_gloo_group
from miles.utils.lora.hf_lora_targets import get_hf_lora_targets

Q, K = "model.layers.0.self_attn.q_proj", "model.layers.0.self_attn.k_proj"
TARGETS = get_hf_lora_targets({"model_type": "qwen3", "num_hidden_layers": 1})


def _export(params):
    """Bridge-like: B gathered over the TP split on rows, one A exported under both names."""
    a, b_q, b_k = params

    def gathered(local):
        parts = [torch.empty_like(local) for _ in range(dist.get_world_size())]
        dist.all_gather(parts, local.data)
        return torch.cat(parts)

    yield f"{Q}.lora_A.weight", a.data.clone()
    yield f"{Q}.lora_B.weight", gathered(b_q)
    yield f"{K}.lora_A.weight", a.data.clone()
    yield f"{K}.lora_B.weight", gathered(b_k)


def _model(params):
    """The probe swaps parameters where modules register them."""
    holder = torch.nn.Module()
    for index, param in enumerate(params):
        holder.register_parameter(f"p{index}", param)
    return [holder]


def _registered(model):
    """What an export reads: the probe's stand-ins while it runs."""
    return list(model[0].parameters())


class _ProbedSlots(ProbedAdapterSlots):
    def slot_parameters(self, slot):
        raise AssertionError("global_shapes reads only the row mapping")


def _probe(rank, world_size, port):
    init_gloo(rank, world_size, port=port)
    init_gloo_group()
    params = [
        torch.nn.Parameter(torch.randn(3, 5)),
        torch.nn.Parameter(torch.randn(2, 3)),
        torch.nn.Parameter(torch.randn(2, 3)),
    ]
    model = _model(params)
    before = [p.data.clone() for p in params]
    # Megatron DDP fills main_grad from a hook on each parameter's grad accumulator, which it keeps alive
    hooked, accumulators = [], [param.expand_as(param).grad_fn.next_functions[0][0] for param in params]
    for accumulator in accumulators:
        accumulator.register_hook(lambda *_: hooked.append(1))
    mapping = probe_bridge_row_mapping(model, params, lambda: _export(_registered(model)), TARGETS)
    sum(p.sum() for p in params).backward()
    assert len(hooked) == len(params), "the probe keeps each parameter's grad accumulator"

    assert mapping.shared_lora_a_groups == ((K, Q),)
    assert all(torch.equal(p.data, b) for p, b in zip(params, before, strict=True)), "the probe restores the params"
    pieces = {p.key: p for p in mapping.pieces}
    assert set(pieces) == {f"{K}.lora_A", f"{Q}.lora_B", f"{K}.lora_B"}, "a shared A is stored once"
    assert pieces[f"{Q}.lora_B"].offsets == (2 * rank, 0) and pieces[f"{Q}.lora_B"].global_shape == (4, 3)
    flat = torch.cat([p.data.reshape(-1) for p in params])
    assert torch.equal(flat[pieces[f"{K}.lora_B"].local_elements], params[2].data)
    assert torch.equal(flat[pieces[f"{K}.lora_A"].local_elements], params[0].data)
    slots = _ProbedSlots(mapping, TARGETS, iterator=None)
    assert slots.global_shapes(2) == {f"{K}.lora_A": (2, 5), f"{Q}.lora_B": (4, 2), f"{K}.lora_B": (4, 2)}
    # one index per local element stays resident; int64 would double it
    assert all(p.local_elements.dtype == torch.int32 for p in mapping.pieces)
    dist.destroy_process_group()


def test_probe_recovers_pieces_and_shared_a_from_a_tp_split_export():
    run_multiprocess(_probe)


def _pp_broadcast_export(params):
    """Bridge-like at PP=2: each stage holds one layer, and every stage's adapters reach every stage."""
    a, b_q, b_k = params

    def broadcast(local):
        parts = [torch.empty_like(local) for _ in range(dist.get_world_size())]
        dist.all_gather(parts, local.data)
        return parts

    for layer, (stage_a, stage_b_q, stage_b_k) in enumerate(
        zip(broadcast(a), broadcast(b_q), broadcast(b_k), strict=True)
    ):
        q, k = Q.replace(".0.", f".{layer}."), K.replace(".0.", f".{layer}.")
        yield f"{q}.lora_A.weight", stage_a.clone()
        yield f"{q}.lora_B.weight", stage_b_q
        yield f"{k}.lora_A.weight", stage_a.clone()
        yield f"{k}.lora_B.weight", stage_b_k


def _probe_pp(rank, world_size, port):
    init_gloo(rank, world_size, port=port)
    init_gloo_group()
    params = [torch.nn.Parameter(torch.randn(3, 5)) for _ in range(3)]
    model = _model(params)
    targets = get_hf_lora_targets({"model_type": "qwen3", "num_hidden_layers": 2})
    mapping = probe_bridge_row_mapping(model, params, lambda: _pp_broadcast_export(_registered(model)), targets)
    k, q = (name.replace(".0.", f".{rank}.") for name in (K, Q))
    assert {p.key for p in mapping.pieces} == {f"{k}.lora_A", f"{q}.lora_B", f"{k}.lora_B"}
    assert mapping.shared_lora_a_groups == tuple(
        (K.replace(".0.", f".{s}."), Q.replace(".0.", f".{s}.")) for s in range(2)
    )
    dist.destroy_process_group()


def test_probe_skips_the_shared_a_of_another_pipeline_stage():
    run_multiprocess(_probe_pp)


GATE, UP = "model.layers.0.mlp.experts.{}.gate_proj", "model.layers.0.mlp.experts.{}.up_proj"


def _expert_export(params):
    """Fused fc1 on 2 local experts per EP rank: one A per expert exported under gate and up; B rows split gate|up."""
    a, b = params

    def gathered(local):
        parts = [torch.empty_like(local) for _ in range(dist.get_world_size())]
        dist.all_gather(parts, local.data)
        return torch.cat(parts)

    all_a, all_b = gathered(a), gathered(b)
    for expert in range(all_a.shape[0]):
        yield f"{GATE.format(expert)}.lora_A.weight", all_a[expert].clone()
        yield f"{UP.format(expert)}.lora_A.weight", all_a[expert].clone()
        yield f"{GATE.format(expert)}.lora_B.weight", all_b[expert, :3].clone()
        yield f"{UP.format(expert)}.lora_B.weight", all_b[expert, 3:].clone()


def _probe_experts(rank, world_size, port):
    init_gloo(rank, world_size, port=port)
    init_gloo_group()
    params = [torch.nn.Parameter(torch.randn(2, 4, 5)), torch.nn.Parameter(torch.randn(2, 6, 4))]
    model = _model(params)
    mapping = probe_bridge_row_mapping(model, params, lambda: _expert_export(_registered(model)), TARGETS)
    gate, up = GATE.format("x").replace(".x.", "."), UP.format("x").replace(".x.", ".")
    assert mapping.shared_lora_a_groups == ((gate, up),)
    pieces = {p.key: p for p in mapping.pieces}
    assert set(pieces) == {f"{gate}.lora_A", f"{gate}.lora_B", f"{up}.lora_B"}, "the up copy of a shared A is dropped"
    assert pieces[f"{up}.lora_B"].offsets == (2 * rank, 0, 0) and pieces[f"{up}.lora_B"].global_shape == (4, 3, 4)
    flat = torch.cat([p.data.reshape(-1) for p in params])
    assert torch.equal(flat[pieces[f"{up}.lora_B"].local_elements], params[1].data[:, 3:])
    dist.destroy_process_group()


def test_probe_stacks_ep_split_experts_and_stores_a_shared_expert_a_once():
    run_multiprocess(_probe_experts)


HEAD = "lm_head"


def _trimmed_head_export(params, served_rows):
    """Native-like: B gathered over the vocab split, then trimmed to the served vocab."""
    a, b = params
    parts = [torch.empty_like(b) for _ in range(dist.get_world_size())]
    dist.all_gather(parts, b.data)
    yield f"{HEAD}.lora_A.weight", a.data.clone()
    yield f"{HEAD}.lora_B.weight", torch.cat(parts)[:served_rows]


def _probe_trimmed_head(rank, world_size, port):
    init_gloo(rank, world_size, port=port)
    init_gloo_group()
    # vocab 7 padded to 8: rank 1's last row of B is padding
    params = [torch.nn.Parameter(torch.randn(2, 5)), torch.nn.Parameter(torch.randn(4, 2))]
    model = _model(params)
    padding = torch.arange(10 + 3 * 2, 10 + 4 * 2) if rank == 1 else torch.empty(0, dtype=torch.long)
    with pytest.raises(RuntimeError, match="rank 1: AssertionError: .*does not reach 2 of this rank's elements"):
        probe_bridge_row_mapping(model, params, lambda: _trimmed_head_export(_registered(model), 7), TARGETS)

    mapping = probe_bridge_row_mapping(
        model, params, lambda: _trimmed_head_export(_registered(model), 7), TARGETS, unexported=padding
    )

    head_b = next(p for p in mapping.pieces if p.key == f"{HEAD}.lora_B")
    assert head_b.global_shape == (7, 2) and head_b.offsets == (4 * rank, 0)
    flat = torch.cat([p.data.reshape(-1) for p in params])
    assert torch.equal(flat[head_b.local_elements], params[1].data[: 4 - rank])
    with pytest.raises(RuntimeError, match="the export reaches elements declared unexported"):
        probe_bridge_row_mapping(
            model, params, lambda: _trimmed_head_export(_registered(model), 8), TARGETS, unexported=padding
        )
    dist.destroy_process_group()


def test_probe_exempts_only_the_declared_unexported_elements_from_coverage():
    run_multiprocess(_probe_trimmed_head)


GATE_UP = "model.layers.0.mlp.gate_up_proj"


def _fused_gate_up_export(params):
    """Native's fused gate_up at TP: gate and up gathered separately, then concatenated."""
    (b,) = params

    def gathered(local):
        parts = [torch.empty_like(local) for _ in range(dist.get_world_size())]
        dist.all_gather(parts, local.contiguous())
        return torch.cat(parts)

    half = b.shape[0] // 2
    yield f"{GATE_UP}.lora_B.weight", torch.cat([gathered(b.data[:half]), gathered(b.data[half:])])


def _probe_fused_gate_up(rank, world_size, port):
    init_gloo(rank, world_size, port=port)
    init_gloo_group()
    params = [torch.nn.Parameter(torch.randn(4, 3))]
    model = _model(params)
    mapping = probe_bridge_row_mapping(model, params, lambda: _fused_gate_up_export(_registered(model)), TARGETS)

    assert [(p.key, p.global_shape, p.offsets) for p in mapping.pieces] == [
        (f"{GATE_UP}.lora_B", (8, 3), (2 * rank, 0)),
        (f"{GATE_UP}.lora_B", (8, 3), (4 + 2 * rank, 0)),
    ]
    flat = params[0].data.reshape(-1)
    assert torch.equal(torch.cat([flat[p.local_elements] for p in mapping.pieces]), params[0].data)
    dist.destroy_process_group()


def test_probe_splits_a_ranks_bands_of_one_key_into_rectangles():
    run_multiprocess(_probe_fused_gate_up)


# process -> slot -> parameter indices it holds; process 0 holds the same in both slots, its peers do not
_HOLDS = [{0: {0}, 1: {0}}, {0: {1}, 1: {2}}, {0: {2}, 1: {1}}]


def _specs_of_two_slots(rank, world_size, port):
    dist.init_process_group(
        "gloo", init_method=f"tcp://localhost:{port}", rank=rank, world_size=world_size, timeout=timedelta(seconds=10)
    )
    init_gloo_group()
    slots = BridgeAdapterSlots.__new__(BridgeAdapterSlots)
    slots._specs, slots._multi_lora, slots.slot_parameters = {}, False, lambda slot: []
    slots.row_mapping = RowMapping(pieces=[], shared_lora_a_groups=(), target_modules=())
    for slot in (0, 1):
        assert slots.local_shard_specs(slot, frozenset(_HOLDS[rank][slot])) == []
    dist.destroy_process_group()


def test_every_process_gathers_shard_specs_for_each_slot():
    run_multiprocess(_specs_of_two_slots, world_size=3)


def _run(**overrides) -> AdapterParametrization:
    fields = dict(
        base_model="base",
        lora_rank=8,
        lora_alpha=16.0,
        lora_type="lora",
        target_modules=("model.layers.0.self_attn.q_proj", "model.layers.0.mlp.experts.down_proj"),
        experts_shared_outer_loras=False,
    )
    return AdapterParametrization(**(fields | overrides))


FP32 = {"_target_": "torch.float32", "_call_": False}


# the peft fields each Bridge class writes: CanonicalLoRA has no lora_dtype or a2a_experimental
_BRIDGE_PEFT_BY_CLASS = {
    "lora": {"_target_": "megatron.bridge.peft.lora.LoRA", "lora_dtype": None, "a2a_experimental": False},
    "canonical_lora": {"_target_": "megatron.bridge.peft.canonical_lora.CanonicalLoRA"},
}


def _bridge_config(peft_class: str = "lora", **peft) -> dict:
    return {
        "peft": {"dim": 8, "alpha": 32, "dropout": 0.0, "exclude_modules": [], "share_expert_adapters": False}
        | _BRIDGE_PEFT_BY_CLASS[peft_class]
        | peft,
        "optimizer": {
            "optimizer": "adam",
            "use_distributed_optimizer": True,
            "use_precision_aware_optimizer": False,
        }
        | {field: FP32 for field in ("main_params_dtype", "exp_avg_dtype", "exp_avg_sq_dtype")},
        "checkpoint": {"save_optim": True},
        "model": {"moe_grouped_gemm": True},
    }


@pytest.mark.parametrize("lora_type", ["lora", "canonical_lora"])
def test_a_bridge_run_config_of_the_same_lora_is_accepted(lora_type):
    assert_bridge_run_config_matches(
        _bridge_config(lora_type), _run(lora_type=lora_type), params_dtype=torch.bfloat16, load_optimizer=True
    )


def test_a_canonical_lora_source_is_refused_by_a_lora_run():
    with pytest.raises(AssertionError, match="lora_type='canonical_lora'"):
        assert_bridge_run_config_matches(
            _bridge_config("canonical_lora"), _run(), params_dtype=torch.bfloat16, load_optimizer=True
        )


@pytest.mark.parametrize(
    "section, fields, match",
    [
        ("peft", {"_target_": "megatron.bridge.peft.dora.DoRA"}, "neither LoRA nor CanonicalLoRA"),
        ("peft", {"dim": 16}, "lora_rank=16"),
        ("peft", {"share_expert_adapters": True}, "share_expert_adapters"),
        ("peft", {"lora_dtype": {"_target_": "torch.float32", "_call_": False}}, "lora_dtype"),
        ("peft", {"normalize_moe_lora": True}, "normalize_moe_lora"),
        ("optimizer", {"main_params_dtype": {"_target_": "torch.float16", "_call_": False}}, "main_params_dtype"),
    ],
    ids=[
        "dora",
        "rank",
        "shared_expert_adapters",
        "lora_dtype",
        "normalize_moe_lora",
        "main_params_dtype",
    ],
)
def test_an_unsupported_bridge_run_config_is_refused(section, fields, match):
    config = _bridge_config()
    config[section] |= fields
    with pytest.raises(AssertionError, match=match):
        assert_bridge_run_config_matches(config, _run(), params_dtype=torch.bfloat16, load_optimizer=True)


def test_share_expert_adapters_defaults_to_bridge_behaviour_and_matters_only_for_per_expert_adapters():
    config = _bridge_config()
    del config["peft"]["share_expert_adapters"]
    with pytest.raises(AssertionError, match="share_expert_adapters"):
        assert_bridge_run_config_matches(config, _run(), params_dtype=torch.bfloat16, load_optimizer=False)
    dense = _run(target_modules=("model.layers.0.self_attn.q_proj",))
    assert_bridge_run_config_matches(config, dense, params_dtype=torch.bfloat16, load_optimizer=False)


def test_share_expert_adapters_is_accepted_for_sequential_experts():
    # Bridge gives each SequentialMLP local expert its own adapter whatever share_expert_adapters says
    config = _bridge_config(share_expert_adapters=True)
    config["model"]["moe_grouped_gemm"] = False
    assert_bridge_run_config_matches(config, _run(), params_dtype=torch.bfloat16, load_optimizer=False)


def test_adam_state_needs_a_distributed_fp32_adam_source():
    config = _bridge_config()
    config["optimizer"]["use_distributed_optimizer"] = False
    assert_bridge_run_config_matches(config, _run(), params_dtype=torch.bfloat16, load_optimizer=False)
    with pytest.raises(AssertionError, match="--no-load-optim"):
        assert_bridge_run_config_matches(config, _run(), params_dtype=torch.bfloat16, load_optimizer=True)
    config = _bridge_config()
    config["optimizer"]["exp_avg_dtype"] = {"_target_": "torch.bfloat16", "_call_": False}
    with pytest.raises(AssertionError, match="exp_avg_dtype must be fp32"):
        assert_bridge_run_config_matches(config, _run(), params_dtype=torch.bfloat16, load_optimizer=True)


def test_the_base_is_bridge_pretrained_checkpoint_not_the_hf_template(tmp_path):
    def hf_dir(name, norm):
        (tmp_path / name).mkdir()
        save_file({"model.norm.weight": norm}, tmp_path / name / "model.safetensors")
        return str(tmp_path / name)

    template, sft = hf_dir("template", torch.ones(4)), hf_dir("sft", torch.ones(4) * 2)
    config = {
        "checkpoint": {"pretrained_checkpoint": sft, "hf_source_path": template},
        "model": {},
        "tokenizer": {"tokenizer_model": None},
    }
    with pytest.raises(AssertionError, match="another base model"):
        assert_same_base_model("src", bridge_base_model_candidates(config), template, remedy=_BRIDGE_BASE_REMEDY)
    # a torch_dist base has no HF files: its HF template stands in
    config["checkpoint"]["pretrained_checkpoint"] = str(tmp_path)
    assert_same_base_model("src", bridge_base_model_candidates(config), template, remedy=_BRIDGE_BASE_REMEDY)
    # Bridge's last fallback
    config["checkpoint"]["hf_source_path"] = None
    config["tokenizer"]["tokenizer_model"] = template
    assert_same_base_model("src", bridge_base_model_candidates(config), template, remedy=_BRIDGE_BASE_REMEDY)
    # Bridge's template is hf_model_id, absent here: a later candidate that resolves must not stand in for it
    config["model"]["hf_model_id"] = str(tmp_path / "missing")
    with pytest.raises(
        AssertionError, match="has no local HF files.*set checkpoint.hf_source_path in the source's run_config.yaml"
    ):
        assert_same_base_model("src", bridge_base_model_candidates(config), template, remedy=_BRIDGE_BASE_REMEDY)


NORM_KEY, QKV_KEY = "decoder.final_layernorm.weight", "decoder.layers.self_attention.linear_qkv.weight"


class _Chunk(torch.nn.Module):
    """A model chunk: one norm, one projection and one adapter weight, keyed as Megatron shards them."""

    def __init__(self, norm: torch.Tensor, qkv: torch.Tensor | None = None):
        super().__init__()
        qkv = torch.ones(2, 2) if qkv is None else qkv
        self.norm, self.qkv, self.lora = (torch.nn.Parameter(t) for t in (norm, qkv, torch.ones(2)))

    def sharded_state_dict(self, metadata=None) -> dict:
        lora_key = "decoder.layers.self_attention.linear_qkv.adapter.linear_in.weight"
        return {
            NORM_KEY: ShardedTensor.from_rank_offsets(NORM_KEY, self.norm.data),
            "decoder.layers.0.self_attention.linear_qkv.weight": ShardedTensor.from_rank_offsets(
                QKV_KEY, self.qkv.data
            ),
            lora_key: ShardedTensor.from_rank_offsets(lora_key, self.lora.data),
        }


@pytest.fixture()
def single_rank_gloo(tmp_path, monkeypatch):
    dist.init_process_group(backend="gloo", init_method=f"file://{tmp_path / 'rendezvous'}", rank=0, world_size=1)
    monkeypatch.setattr(distributed_utils, "GLOO_GROUP", None)
    # torch_dist assumes CUDA: it syncs D2H staging copies and broadcasts its status flag on the current device
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args: None)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")
    monkeypatch.setattr(
        bridge_checkpoint_loader,
        "get_parallel_state",
        lambda: SimpleNamespace(intra_dp_cp=SimpleNamespace(group=None)),
    )
    init_gloo_group()
    try:
        yield
    finally:
        dist.destroy_process_group()


def _torch_dist_base(path, norm: torch.Tensor, *, with_norm: bool = True, qkv: torch.Tensor | None = None):
    path.mkdir()
    state = _Chunk(norm, qkv).sharded_state_dict()
    if not with_norm:
        del state[NORM_KEY]
    dist_checkpointing.save({"model": state}, str(path))
    return path


def test_a_torch_dist_base_fine_tuned_after_its_hf_conversion_is_refused(single_rank_gloo, tmp_path):
    base = torch.randn(4)
    chunk = _Chunk(base.clone())
    assert_same_base_as_torch_dist("src", [chunk], _torch_dist_base(tmp_path / "same", base))
    assert torch.equal(chunk.norm.data, base), "the comparison never writes the model's own weights"
    with pytest.raises(
        RuntimeError, match="trained on another base model .* differs in e.g. \\['decoder.final_layernorm"
    ):
        assert_same_base_as_torch_dist("src", [chunk], _torch_dist_base(tmp_path / "sft", base + 1))
    with pytest.raises(RuntimeError, match="cannot be compared with this run's model: it lacks"):
        assert_same_base_as_torch_dist("src", [chunk], _torch_dist_base(tmp_path / "bare", base, with_norm=False))


def test_a_torch_dist_base_with_a_merged_lora_is_refused(single_rank_gloo, tmp_path):
    norm = torch.randn(4)
    with pytest.raises(
        RuntimeError, match="trained on another base model .* differs in e.g. \\['decoder.layers.0.self"
    ):
        assert_same_base_as_torch_dist(
            "src", [_Chunk(norm.clone())], _torch_dist_base(tmp_path / "merged", norm, qkv=torch.ones(2, 2) + 1)
        )


def test_a_pretrained_checkpoint_resolves_to_its_tracked_torch_dist_iteration(single_rank_gloo, tmp_path):
    _torch_dist_base(tmp_path / "iter_0000007", torch.ones(4))
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("7")
    assert torch_dist_base_dir(str(tmp_path)) == tmp_path / "iter_0000007"
    assert torch_dist_base_dir(str(tmp_path / "iter_0000007")) == tmp_path / "iter_0000007"
    assert torch_dist_base_dir(str(tmp_path / "missing")) is None
    assert torch_dist_base_dir(None) is None


GDN = "model.layers.0.linear_attn"
GDN_TARGETS = get_hf_lora_targets(
    {
        "model_type": "qwen3_5_text",
        "num_hidden_layers": 1,
        "layer_types": ["linear_attention"],
        "linear_num_key_heads": 1,
        "linear_key_head_dim": 2,
        "linear_num_value_heads": 1,
        "linear_value_head_dim": 3,
    }
)


def _gdn_export(params):
    """Bridge-like GDN in_proj: the fused in_proj_qkv and in_proj_z export one A."""
    a, b_qkv, b_z = params
    yield f"{GDN}.in_proj_qkv.lora_A.weight", a.data.clone()
    yield f"{GDN}.in_proj_qkv.lora_B.weight", b_qkv.data.clone()
    yield f"{GDN}.in_proj_z.lora_A.weight", a.data.clone()
    yield f"{GDN}.in_proj_z.lora_B.weight", b_z.data.clone()


def _probe_gdn(rank, world_size, port):
    init_gloo(rank, world_size, port=port)
    init_gloo_group()
    params = [torch.nn.Parameter(torch.randn(4, 6)), torch.nn.Parameter(torch.randn(7, 4))]
    params.append(torch.nn.Parameter(torch.randn(3, 4)))
    model = _model(params)
    mapping = probe_bridge_row_mapping(model, params, lambda: _gdn_export(_registered(model)), GDN_TARGETS)
    group = tuple(f"{GDN}.in_proj_{m}" for m in "kqvz")
    assert mapping.shared_lora_a_groups == (group,)
    pieces = {p.key: p for p in mapping.pieces}
    assert set(pieces) == {f"{group[0]}.lora_A"} | {f"{m}.lora_B" for m in group}
    flat = torch.cat([p.data.reshape(-1) for p in params])
    assert torch.equal(flat[pieces[f"{group[0]}.lora_A"].local_elements], params[0].data)
    assert torch.equal(flat[pieces[f"{GDN}.in_proj_k.lora_B"].local_elements], params[1].data[2:4])
    dist.destroy_process_group()


def test_probe_stores_a_fused_gdn_modules_shared_a_under_the_sorted_first_member():
    run_multiprocess(_probe_gdn)


def _export_fails_on_rank_1(params):
    yield f"{Q}.lora_A.weight", params[0].data.clone()
    if dist.get_rank() == 1:
        raise ValueError("rank 1 cannot export")


def _probe_export_fails(rank, world_size, port):
    init_gloo(rank, world_size, port=port)
    init_gloo_group()
    params = [torch.nn.Parameter(torch.randn(3, 5))]
    model = _model(params)
    with pytest.raises(RuntimeError, match="adapter checkpoint failed: rank 1: ValueError: rank 1 cannot export"):
        probe_bridge_row_mapping(model, params, lambda: _export_fails_on_rank_1(_registered(model)), TARGETS)
    dist.destroy_process_group()


def test_an_export_failure_on_one_rank_fails_every_rank():
    """The fake export has no collectives; a rank failing inside one of Bridge's still hangs the others."""
    run_multiprocess(_probe_export_fails)


def _slots_of(pieces, real_rank):
    """Multi-LoRA slots whose probe found ``pieces``, padded to max rank 4; the slot's real rank is ``real_rank``."""
    slots = BridgeAdapterSlots.__new__(BridgeAdapterSlots)
    slots.row_mapping, slots._specs = RowMapping(pieces, (), ()), {}
    slots.lora_rank = lambda slot: real_rank
    return slots


def _trim_a_tp_split_rank(rank, world_size, port):
    init_gloo(rank, world_size, port=port)
    init_gloo_group()
    # max rank 4 split over 2 TP ranks: each rank holds 2 rows of A ([rank, in]) and 2 columns of B ([out, rank])
    a = HfPiece(f"{Q}.lora_A", (4, 3), (2 * rank, 0), torch.arange(6).view(2, 3))
    b = HfPiece(f"{Q}.lora_B", (5, 4), (0, 2 * rank), torch.arange(6, 16).view(5, 2))
    flat = torch.arange(16.0)

    slots = _slots_of([a, b], real_rank=3)
    specs = [(s.key, s.global_shape, s.offsets, s.local_shape) for s in slots.get_local_shard_specs(1)]
    kept = 2 if rank == 0 else 1
    assert specs == [(a.key, (3, 3), (2 * rank, 0), (kept, 3)), (b.key, (5, 3), (0, 2 * rank), (5, kept))]
    piece_a, piece_b = slots.local_pieces(1, [flat])
    assert torch.equal(piece_a, flat[a.local_elements[:kept]]) and torch.equal(
        piece_b, flat[b.local_elements[:, :kept]]
    )

    # real rank 2: rank 1's rows and columns are all padding, so it holds no piece of the slot
    slots = _slots_of([a, b], real_rank=2)
    assert (slots.get_local_shard_specs(1) == []) == (rank == 1)
    assert (slots.local_pieces(1, [flat]) == []) == (rank == 1)
    dist.destroy_process_group()


def test_a_multi_lora_slot_keeps_only_its_real_rank_of_a_tp_split_rank_axis():
    run_multiprocess(_trim_a_tp_split_rank)


def _offloaded_optimizer():
    """One DistributedOptimizer over an empty HybridDeviceOptimizer state; dummy steps land in the returned list."""
    from megatron.core.optimizer import ChainedOptimizer, DistributedOptimizer
    from megatron.core.optimizer.cpu_offloading.hybrid_optimizer import HybridDeviceOptimizer

    param = torch.nn.Parameter(torch.zeros(2))
    hdo = HybridDeviceOptimizer.__new__(HybridDeviceOptimizer)
    hdo.state, hdo.param_groups, hdo.param_to_inner_param = {}, [{"params": [param]}], {param: torch.ones(2)}
    stepped = []
    hdo.dummy_step = lambda: stepped.append(True)
    sub = DistributedOptimizer.__new__(DistributedOptimizer)
    sub.optimizer = hdo

    class Chained(ChainedOptimizer):
        param_groups = []

        def state_dict(self):
            return {}

        def load_state_dict(self, state):
            # what DistributedOptimizer.load_state_dict does before reading the state
            if len(hdo.state) == 0:
                hdo.dummy_step()

        def sharded_state_dict(self, model_sharded_state_dict, is_loading=False, metadata=None):
            # what DistributedOptimizer.sharded_state_dict does when it builds the state to load
            if is_loading:
                self.load_state_dict(self.state_dict())
            return {}

    chained = Chained.__new__(Chained)
    chained.chained_optimizers = [sub]
    return chained, hdo, param, stepped


def _checked_loader(optimizer, members, monkeypatch, *, load_optimizer=True, keys=()):
    """A loader over an empty model with a stubbed source read."""
    run = AdapterParametrization("base", 4, 8.0, "lora", ("model.layers.0.self_attn.q_proj",), False)
    loader = bridge_checkpoint_loader.MegatronBridgeCheckpointLoader([], optimizer, members, run, load_optimizer)
    monkeypatch.setattr(bridge_checkpoint_loader, "unwrap_model", lambda model: [])
    monkeypatch.setattr(
        loader, "_read_and_check_source", lambda path, chunks: ({"peft": {"alpha": 16}}, {}, [], [], None)
    )
    metadata = SimpleNamespace(state_dict_metadata=dict.fromkeys(keys))
    monkeypatch.setattr(
        bridge_checkpoint_loader.dcp, "FileSystemReader", lambda path: SimpleNamespace(read_metadata=lambda: metadata)
    )
    monkeypatch.setattr(bridge_checkpoint_loader.dist_checkpointing, "load", lambda state, path: {"optimizer": {}})
    return loader, run


def test_an_offloaded_optimizer_is_never_dummy_stepped_from_parametrization_to_load(
    single_rank_gloo, tmp_path, monkeypatch
):
    """Building the optimizer's state to load runs Megatron's load_state_dict, which dummy-steps an empty
    HybridDeviceOptimizer state: a real Adam step on random gradients, before any source state lands."""
    optimizer, hdo, param, stepped = _offloaded_optimizer()
    loader, _ = _checked_loader(optimizer, SimpleNamespace(export_adapter_state=lambda slot: None), monkeypatch)

    loader.parametrization(tmp_path)
    assert not stepped and not hdo.state, "checking the source must not touch the optimizer"
    loader.apply(loader.prepare(tmp_path))

    assert not stepped, "the empty HDO state must be zero-initialized, not dummy-stepped"
    assert torch.equal(hdo.state[param]["master_param"], torch.ones(2)) and int(hdo.state[param]["step"]) == 0


@pytest.mark.parametrize(
    "keys, has_optimizer_state",
    [(["optimizer.distributed.exp_avg"], True), (["chained_0.optimizer.step"], True), ([], False)],
    ids=["optimizer", "chained-optimizer", "weights-only"],
)
def test_a_bridge_checkpoint_reports_whether_it_holds_optimizer_state(
    single_rank_gloo, tmp_path, monkeypatch, keys, has_optimizer_state
):
    """load_adapter_source refuses a source without it unless --no-load-optim."""
    loader, run = _checked_loader(None, None, monkeypatch, keys=keys)
    assert loader.parametrization(tmp_path) == (replace(run, lora_alpha=16.0), has_optimizer_state)


def test_the_gateways_parametrization_fields_are_the_slots_keyword_arguments():
    """The stdlib gateway builds these fields; the trainer spreads them into the slots across Ray."""
    record = SimpleNamespace(base_model="m", lora_rank=8, lora_alpha=16.0)
    config = SimpleNamespace(lora_type="lora", experts_shared_outer_loras=False, base_model_fingerprint="f")
    keyword_only = {
        name
        for name, parameter in inspect.signature(BridgeAdapterSlots.parametrization).parameters.items()
        if parameter.kind is inspect.Parameter.KEYWORD_ONLY
    }
    assert set(build_parametrization_fields(record, config)) == keyword_only
