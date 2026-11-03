"""Bridge row-mapping probe and the run_config.yaml match."""

from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
from safetensors.torch import save_file
from tests.fast.dist_utils import init_gloo, run_multiprocess

from miles.backends.megatron_utils.lora.bridge_checkpoint import (
    BridgeAdapterSlots,
    assert_bridge_run_config_matches,
    bridge_base_model_candidates,
    probe_bridge_row_mapping,
)
from miles.backends.training_utils.checkpoint.adapter.base_model import assert_same_base_model
from miles.backends.training_utils.checkpoint.adapter.parametrization import AdapterParametrization
from miles.backends.training_utils.checkpoint.adapter.probed_slots import RowMapping
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


def _probe(rank, world_size, port):
    init_gloo(rank, world_size, port=port)
    init_gloo_group()
    params = [
        torch.nn.Parameter(torch.randn(3, 5)),
        torch.nn.Parameter(torch.randn(2, 3)),
        torch.nn.Parameter(torch.randn(2, 3)),
    ]
    before = [p.data.clone() for p in params]
    mapping = probe_bridge_row_mapping(params, lambda: _export(params), TARGETS)

    assert mapping.shared_lora_a_groups == ((K, Q),)
    assert all(torch.equal(p.data, b) for p, b in zip(params, before, strict=True)), "the probe restores the params"
    pieces = {p.key: p for p in mapping.pieces}
    assert set(pieces) == {f"{K}.lora_A", f"{Q}.lora_B", f"{K}.lora_B"}, "a shared A is stored once"
    assert pieces[f"{Q}.lora_B"].offsets == (2 * rank, 0) and pieces[f"{Q}.lora_B"].global_shape == (4, 3)
    flat = torch.cat([p.data.reshape(-1) for p in params])
    assert torch.equal(flat[pieces[f"{K}.lora_B"].local_elements], params[2].data)
    assert torch.equal(flat[pieces[f"{K}.lora_A"].local_elements], params[0].data)
    dist.destroy_process_group()


def test_probe_recovers_pieces_and_shared_a_from_a_tp_split_export():
    run_multiprocess(_probe)


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
    mapping = probe_bridge_row_mapping(params, lambda: _expert_export(params), TARGETS)
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
    padding = torch.arange(10 + 3 * 2, 10 + 4 * 2) if rank == 1 else torch.empty(0, dtype=torch.long)
    with pytest.raises(RuntimeError, match="rank 1: AssertionError: .*does not reach 2 of this rank's elements"):
        probe_bridge_row_mapping(params, lambda: _trimmed_head_export(params, 7), TARGETS)

    mapping = probe_bridge_row_mapping(params, lambda: _trimmed_head_export(params, 7), TARGETS, unexported=padding)

    head_b = next(p for p in mapping.pieces if p.key == f"{HEAD}.lora_B")
    assert head_b.global_shape == (7, 2) and head_b.offsets == (4 * rank, 0)
    flat = torch.cat([p.data.reshape(-1) for p in params])
    assert torch.equal(flat[head_b.local_elements], params[1].data[: 4 - rank])
    with pytest.raises(RuntimeError, match="the export reaches elements declared unexported"):
        probe_bridge_row_mapping(params, lambda: _trimmed_head_export(params, 8), TARGETS, unexported=padding)
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
    mapping = probe_bridge_row_mapping(params, lambda: _fused_gate_up_export(params), TARGETS)

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
        ("peft", {"dropout": 0.1}, "dropout"),
        ("peft", {"share_expert_adapters": True}, "share_expert_adapters"),
        ("peft", {"lora_dtype": {"_target_": "torch.float32", "_call_": False}}, "lora_dtype"),
        ("peft", {"normalize_moe_lora": True}, "normalize_moe_lora"),
        ("peft", {"exclude_modules": ["linear_fc1"]}, "exclude_modules"),
        ("peft", {"a2a_experimental": True}, "a2a_experimental"),
        ("optimizer", {"use_precision_aware_optimizer": True}, "precision-aware"),
        ("optimizer", {"main_params_dtype": {"_target_": "torch.float16", "_call_": False}}, "main_params_dtype"),
        ("checkpoint", {"save_optim": False}, "saved no optimizer state"),
    ],
    ids=[
        "dora",
        "rank",
        "dropout",
        "shared_expert_adapters",
        "lora_dtype",
        "normalize_moe_lora",
        "exclude_modules",
        "a2a_experimental",
        "precision_aware_optimizer",
        "main_params_dtype",
        "no_saved_optimizer",
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
        assert_same_base_model("src", bridge_base_model_candidates(config), template)
    # a torch_dist base has no HF files: its HF template stands in
    config["checkpoint"]["pretrained_checkpoint"] = str(tmp_path)
    assert_same_base_model("src", bridge_base_model_candidates(config), template)
    # Bridge's last fallback
    config["checkpoint"]["hf_source_path"] = None
    config["tokenizer"]["tokenizer_model"] = template
    assert_same_base_model("src", bridge_base_model_candidates(config), template)


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
    mapping = probe_bridge_row_mapping(params, lambda: _gdn_export(params), GDN_TARGETS)
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
    with pytest.raises(RuntimeError, match="adapter checkpoint failed: rank 1: ValueError: rank 1 cannot export"):
        probe_bridge_row_mapping(params, lambda: _export_fails_on_rank_1(params), TARGETS)
    dist.destroy_process_group()


def test_an_export_failure_on_one_rank_fails_every_rank():
    """The fake export has no collectives; a rank failing inside one of Bridge's still hangs the others."""
    run_multiprocess(_probe_export_fails)
