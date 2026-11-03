"""``save_adapter_dir`` -> commit -> ``load_adapter_dir`` restores weights, moments and step under a new split."""

import tempfile
from dataclasses import replace
from functools import partial
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint
from tests.fast.dist_utils import init_gloo, run_multiprocess

from miles.backends.training_utils.checkpoint.adapter.adapter_dir import (
    load_adapter_dir,
    save_adapter_dir,
    write_peft_export,
)
from miles.backends.training_utils.checkpoint.adapter.adapter_exports import write_peft_adapter
from miles.backends.training_utils.checkpoint.adapter.adapter_state import AdapterState, LocalShardSpec
from miles.backends.training_utils.checkpoint.adapter.commit import (
    commit_adapter_dir,
    resolve_adapter_dir,
    tenant_digest,
)
from miles.backends.training_utils.checkpoint.adapter.parametrization import AdapterParametrization
from miles.backends.training_utils.checkpoint.adapter.readers import AdapterSourceKind
from miles.utils.distributed_utils import init_gloo_group
from miles.utils.lora.hf_lora_targets import get_hf_lora_targets

MODULE = "model.layers.0.self_attn.q_proj"
RUN = AdapterParametrization("m", 2, 4.0, "canonical_lora", (MODULE,), False)
TARGETS = get_hf_lora_targets({"model_type": "qwen3", "num_hidden_layers": 1})
FULL = {
    "lora_A": torch.arange(2 * 6, dtype=torch.float32).reshape(2, 6),
    "lora_B": torch.arange(4 * 2, dtype=torch.float32).reshape(4, 2),
}


class FakeMembers:
    """B split by rows across ranks (column-parallel), A replicated with rank 0 primary."""

    def __init__(self, scale: float = 1.0):
        self.rank, self.world = (dist.get_rank(), dist.get_world_size()) if dist.is_initialized() else (0, 1)
        self.scale, self.imported = scale, None

    def get_local_shard_specs(self, slot):
        rows = 4 // self.world
        return [
            LocalShardSpec(f"{MODULE}.lora_A", (2, 6), (0, 0), (2, 6), is_primary=self.rank == 0),
            LocalShardSpec(f"{MODULE}.lora_B", (4, 2), (self.rank * rows, 0), (rows, 2)),
        ]

    def export_adapter_state(self, slot):
        specs = self.get_local_shard_specs(slot)
        pieces = [
            FULL[s.key.rsplit(".", 1)[1]][s.offsets[0] : s.offsets[0] + s.local_shape[0]] * self.scale for s in specs
        ]
        return AdapterState(specs, pieces, [p * 2 for p in pieces], [p * 3 for p in pieces], step=5)

    def import_adapter_state(self, slot, state):
        self.imported = state

    def export_bf16_adapter(self, slot):
        raise NotImplementedError


def _save(rank, world_size, port, *, root):
    init_gloo(rank, world_size, port=port)
    init_gloo_group()
    written = save_adapter_dir(
        Path(root), "w", FakeMembers(), 0, RUN, tenant_digest=tenant_digest("alice"), training_state={"iteration": 3}
    )
    if rank == 0:
        commit_adapter_dir(Path(root), "w", written, overwrite=False)
    dist.destroy_process_group()


def test_round_trip_with_optimizer_under_a_different_world_size():
    root = tempfile.mkdtemp()
    run_multiprocess(partial(_save, root=root))
    committed = resolve_adapter_dir(Path(root), "w", tenant="alice")
    members = FakeMembers()  # one process now owns everything
    source = load_adapter_dir(committed.path, members, 0, RUN, TARGETS, load_optimizer=True)
    assert source == RUN and members.imported.step == 5
    for spec, weight, exp_avg in zip(
        members.imported.shard_specs, members.imported.weights, members.imported.exp_avgs, strict=True
    ):
        assert torch.equal(weight, FULL[spec.key.rsplit(".", 1)[1]])
        assert torch.equal(exp_avg, weight * 2)
    assert (committed.path / "training_state.json").read_text().count('"iteration": 3') == 1


def test_weights_only_from_peft_upcasts_and_zeroes_moments(tmp_path):
    write_peft_adapter(tmp_path, {f"{MODULE}.{k}": v for k, v in FULL.items()}, RUN, TARGETS)
    members = FakeMembers()
    load_adapter_dir(tmp_path, members, 0, RUN, TARGETS, load_optimizer=False)
    state = members.imported
    assert state.step == 0 and all(not m.any() for m in state.exp_avgs)
    assert all(w.dtype == torch.float32 for w in state.weights)
    assert torch.equal(state.weights[1], FULL["lora_B"].to(torch.bfloat16).float())


def test_a_different_source_alpha_is_returned_for_the_caller_to_adopt(tmp_path):
    write_peft_adapter(tmp_path, {f"{MODULE}.{k}": v for k, v in FULL.items()}, RUN, TARGETS)
    run = replace(RUN, lora_alpha=16.0)  # a Tinker session adopts the source's alpha after loading
    assert load_adapter_dir(tmp_path, FakeMembers(), 0, run, TARGETS, load_optimizer=False).lora_alpha == 4.0


def test_weights_only_from_the_same_parametrization_reads_only_local_pieces(monkeypatch):
    root = tempfile.mkdtemp()
    run_multiprocess(partial(_save, root=root))
    committed = resolve_adapter_dir(Path(root), "w", tenant="alice")

    def whole_tensor_read(*args, **kwargs):
        raise AssertionError("a same-parametrization load must not read whole tensors")

    monkeypatch.setattr(
        "miles.backends.training_utils.checkpoint.adapter.readers.MilesAdapterDirReader.read_weights",
        whole_tensor_read,
    )
    members = FakeMembers()
    load_adapter_dir(committed.path, members, 0, RUN, TARGETS, load_optimizer=False)
    assert members.imported.step == 0 and not members.imported.exp_avgs[0].any()
    assert torch.equal(members.imported.weights[1], FULL["lora_B"])


def _rank0_cannot_create(rank, world_size, port, *, root):
    init_gloo(rank, world_size, port=port)
    init_gloo_group()
    if rank == 0:
        Path(root, "blocker").write_text("")
        # a file where the root directory should be makes rank 0's mkdir fail
        root = str(Path(root, "blocker"))
    else:
        root = str(Path(root, "blocker"))
    with pytest.raises(RuntimeError, match="adapter checkpoint failed"):
        save_adapter_dir(Path(root), "w", FakeMembers(), 0, RUN, tenant_digest=None)
    dist.destroy_process_group()


def test_a_rank0_prepare_failure_fails_every_rank_instead_of_hanging():
    root = tempfile.mkdtemp()
    process = (
        __import__("multiprocessing")
        .get_context("spawn")
        .Process(target=run_multiprocess, args=(partial(_rank0_cannot_create, root=root),))
    )
    process.start()
    process.join(timeout=60)
    hung = process.is_alive()
    process.kill()
    assert not hung and process.exitcode == 0


def test_an_in_model_source_without_alpha_takes_the_runs(tmp_path):
    torch.distributed.checkpoint.save(
        {"model.layers.0.attention.wq.lora_a.weight": torch.zeros(2)}, checkpoint_id=tmp_path, no_dist=True
    )
    source = load_adapter_dir(
        tmp_path,
        FakeMembers(),
        0,
        RUN,
        TARGETS,
        load_optimizer=True,
        in_model_loaders_by_kind={AdapterSourceKind.TORCHTITAN_CHECKPOINT: lambda path: replace(RUN, lora_alpha=None)},
    )
    assert source.lora_alpha == RUN.lora_alpha


class WeightsOnlyMembers(FakeMembers):
    """A run saved with --no-save-optim."""

    def export_adapter_state(self, slot):
        state = super().export_adapter_state(slot)
        return AdapterState(state.shard_specs, state.weights, [], [], step=5, has_optimizer_state=False)


def test_a_weights_only_dir_loads_without_optimizer_and_refuses_with_it(tmp_path):
    written = save_adapter_dir(tmp_path, "w", WeightsOnlyMembers(), 0, RUN, tenant_digest=None)
    members = FakeMembers()
    load_adapter_dir(written.path, members, 0, RUN, TARGETS, load_optimizer=False)
    assert torch.equal(members.imported.weights[1], FULL["lora_B"]) and members.imported.step == 0
    with pytest.raises(RuntimeError, match="no optimizer state"):
        load_adapter_dir(written.path, FakeMembers(), 0, RUN, TARGETS, load_optimizer=True)


class ExtraKeyMembers(FakeMembers):
    """Writes one tensor the run does not train, e.g. a second copy of a shared A."""

    def export_adapter_state(self, slot):
        state = super().export_adapter_state(slot)
        extra = LocalShardSpec("model.layers.0.self_attn.k_proj.lora_A", (2, 6), (0, 0), (2, 6))
        w = FULL["lora_A"].clone()
        return AdapterState(
            [*state.shard_specs, extra], [*state.weights, w], [*state.exp_avgs, w], [*state.exp_avg_sqs, w], step=5
        )


def test_a_backend_state_that_does_not_match_the_parametrization_is_refused_at_save(tmp_path):
    with pytest.raises(RuntimeError, match="unexpected .*k_proj"):
        save_adapter_dir(tmp_path, "w", ExtraKeyMembers(), 0, RUN, tenant_digest=None)
    assert resolve_adapter_dir(tmp_path, "w", tenant=None) is None


def _one_rank_cannot_read(rank, world_size, port, *, root):
    init_gloo(rank, world_size, port=port)
    init_gloo_group()
    path = Path(root) if rank == 0 else Path(root, "missing")
    with pytest.raises(RuntimeError, match="adapter checkpoint failed: rank 1"):
        load_adapter_dir(path, FakeMembers(), 0, RUN, TARGETS, load_optimizer=False)
    dist.destroy_process_group()


def test_a_load_failure_on_one_rank_fails_every_rank_instead_of_hanging(tmp_path):
    write_peft_adapter(tmp_path, {f"{MODULE}.{k}": v for k, v in FULL.items()}, RUN, TARGETS)
    process = (
        __import__("multiprocessing")
        .get_context("spawn")
        .Process(target=run_multiprocess, args=(partial(_one_rank_cannot_read, root=str(tmp_path)),))
    )
    process.start()
    process.join(timeout=60)
    hung = process.is_alive()
    process.kill()
    assert not hung and process.exitcode == 0


def test_an_in_model_source_with_another_parametrization_is_refused(tmp_path):
    torch.distributed.checkpoint.save(
        {"model.layers.0.attention.wq.lora_a.weight": torch.zeros(2)}, checkpoint_id=tmp_path, no_dist=True
    )
    shared = replace(RUN, target_modules=(MODULE, "m.k_proj"), shared_lora_a_groups=((MODULE, "m.k_proj"),))
    with pytest.raises(RuntimeError, match="cannot do"):
        load_adapter_dir(
            tmp_path,
            FakeMembers(),
            0,
            replace(shared, shared_lora_a_groups=()),
            TARGETS,
            load_optimizer=False,
            in_model_loaders_by_kind={AdapterSourceKind.TORCHTITAN_CHECKPOINT: lambda path: shared},
        )


class NoPrimaryAMembers(FakeMembers):
    """Every rank marks lora_A a replica, so no rank writes it."""

    def get_local_shard_specs(self, slot):
        a, b = super().get_local_shard_specs(slot)
        return [replace(a, is_primary=False), b]


def _save_without_a_primary(rank, world_size, port, *, root):
    init_gloo(rank, world_size, port=port)
    init_gloo_group()
    with pytest.raises(RuntimeError, match="missing .*lora_A"):
        save_adapter_dir(Path(root), "w", NoPrimaryAMembers(), 0, RUN, tenant_digest=None)
    dist.destroy_process_group()


def test_a_key_without_a_primary_is_refused_at_save():
    run_multiprocess(partial(_save_without_a_primary, root=tempfile.mkdtemp()))


class SpecsFailOnRank1(FakeMembers):
    def get_local_shard_specs(self, slot):
        assert self.rank != 1, "rank 1 cannot describe its pieces"
        return super().get_local_shard_specs(slot)


def _specs_fail_on_one_rank(rank, world_size, port, *, root):
    init_gloo(rank, world_size, port=port)
    init_gloo_group()
    with pytest.raises(RuntimeError, match="rank 1 cannot describe"):
        load_adapter_dir(Path(root), SpecsFailOnRank1(), 0, RUN, TARGETS, load_optimizer=False)
    dist.destroy_process_group()


def test_a_shard_spec_failure_on_one_rank_fails_every_rank(tmp_path):
    write_peft_adapter(tmp_path, {f"{MODULE}.{k}": v for k, v in FULL.items()}, RUN, TARGETS)
    process = (
        __import__("multiprocessing")
        .get_context("spawn")
        .Process(target=run_multiprocess, args=(partial(_specs_fail_on_one_rank, root=str(tmp_path)),))
    )
    process.start()
    process.join(timeout=60)
    hung = process.is_alive()
    process.kill()
    assert not hung and process.exitcode == 0


class PaddedRankMembers(FakeMembers):
    """A multi-LoRA backend that forgot to cut a max-rank slot to its real rank."""

    def get_local_shard_specs(self, slot):
        a, b = super().get_local_shard_specs(slot)
        return [a, replace(b, global_shape=(4, 3), local_shape=(4 // self.world, 3))]

    def export_adapter_state(self, slot):
        specs = self.get_local_shard_specs(slot)
        pieces = [FULL["lora_A"].clone(), torch.zeros(specs[1].local_shape)]
        return AdapterState(specs, pieces, pieces, pieces, step=1)


def test_a_padded_rank_is_refused_at_save(tmp_path):
    with pytest.raises(RuntimeError, match="rank 3 != 2"):
        save_adapter_dir(tmp_path, "w", PaddedRankMembers(), 0, RUN, tenant_digest=None)


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
GDN_ROWS = {"in_proj_qkv": 7, "in_proj_z": 3, "in_proj_b": 1, "in_proj_a": 1}


class GdnBridgeMembers:
    """Bridge's PEFT-named export: one A under every HF module of the group, in_proj_qkv fused."""

    bf16_adapter_key_style = "peft"

    def __init__(self, modules):
        self.a = torch.randn(4, 6)
        self.b = {m: torch.randn(GDN_ROWS[m], 4) for m in modules}

    def export_bf16_adapter(self, slot):
        for module, b in self.b.items():
            yield f"{GDN}.{module}.lora_A.weight", self.a.clone()
            yield f"{GDN}.{module}.lora_B.weight", b.clone()


@pytest.mark.parametrize(
    "modules", [("in_proj_qkv", "in_proj_z"), ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a")]
)
def test_peft_export_of_a_fused_gdn_group_writes_one_a_per_exported_module(tmp_path, modules):
    from safetensors.torch import load_file

    members = GdnBridgeMembers(modules)
    components = [
        c for m in modules for c in (("in_proj_q", "in_proj_k", "in_proj_v") if m == "in_proj_qkv" else (m,))
    ]
    group = tuple(sorted(f"{GDN}.{c}" for c in components))
    run = AdapterParametrization("m", 4, 8.0, "lora", group, False, shared_lora_a_groups=(group,))
    write_peft_export(tmp_path, members, 0, run, GDN_TARGETS)
    tensors = load_file(str(tmp_path / "adapter_model.safetensors"))
    assert set(tensors) == {f"{GDN}.{m}.lora_{f}.weight" for m in modules for f in "AB"}
    for module, b in members.b.items():
        assert torch.equal(tensors[f"{GDN}.{module}.lora_A.weight"], members.a.to(torch.bfloat16))
        assert torch.equal(tensors[f"{GDN}.{module}.lora_B.weight"], b.to(torch.bfloat16))
