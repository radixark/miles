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

from miles.backends.training_utils.checkpoint.adapter.adapter_dir import load_adapter_dir, save_adapter_dir
from miles.backends.training_utils.checkpoint.adapter.adapter_exports import write_peft_adapter
from miles.backends.training_utils.checkpoint.adapter.adapter_state import AdapterState, LocalShardSpec
from miles.backends.training_utils.checkpoint.adapter.commit import commit_adapter_dir, resolve_adapter_dir
from miles.backends.training_utils.checkpoint.adapter.parametrization import AdapterParametrization
from miles.backends.training_utils.checkpoint.adapter.readers import AdapterSourceKind
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
    written = save_adapter_dir(Path(root), "w", FakeMembers(), 0, RUN, tenant="alice", training_state={"iteration": 3})
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
    write_peft_adapter(tmp_path, {f"{MODULE}.{k}": v for k, v in FULL.items()}, RUN)
    members = FakeMembers()
    load_adapter_dir(tmp_path, members, 0, RUN, TARGETS, load_optimizer=False)
    state = members.imported
    assert state.step == 0 and all(not m.any() for m in state.exp_avgs)
    assert all(w.dtype == torch.float32 for w in state.weights)
    assert torch.equal(state.weights[1], FULL["lora_B"].to(torch.bfloat16).float())


def test_a_different_source_alpha_is_returned_for_the_caller_to_adopt(tmp_path):
    write_peft_adapter(tmp_path, {f"{MODULE}.{k}": v for k, v in FULL.items()}, RUN)
    run = replace(RUN, lora_alpha=16.0)  # a Tinker session adopts the source's alpha after loading
    assert load_adapter_dir(tmp_path, FakeMembers(), 0, run, TARGETS, load_optimizer=False).lora_alpha == 4.0


def test_weights_only_from_the_same_parametrization_reads_only_local_pieces(monkeypatch):
    root = tempfile.mkdtemp()
    run_multiprocess(partial(_save, root=root))
    committed = resolve_adapter_dir(Path(root), "w", tenant="alice")

    def whole_tensor_read(*args, **kwargs):
        raise AssertionError("a same-parametrization load must not read whole tensors")

    monkeypatch.setattr("torch.distributed.checkpoint.state_dict_loader._load_state_dict_from_keys", whole_tensor_read)
    members = FakeMembers()
    load_adapter_dir(committed.path, members, 0, RUN, TARGETS, load_optimizer=False)
    assert members.imported.step == 0 and not members.imported.exp_avgs[0].any()
    assert torch.equal(members.imported.weights[1], FULL["lora_B"])


def _rank0_cannot_create(rank, world_size, port, *, root):
    init_gloo(rank, world_size, port=port)
    if rank == 0:
        Path(root, "blocker").write_text("")
        # a file where the root directory should be makes rank 0's mkdir fail
        root = str(Path(root, "blocker"))
    else:
        root = str(Path(root, "blocker"))
    with pytest.raises(RuntimeError, match="adapter checkpoint failed"):
        save_adapter_dir(Path(root), "w", FakeMembers(), 0, RUN)
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
    written = save_adapter_dir(tmp_path, "w", WeightsOnlyMembers(), 0, RUN)
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


def test_an_unexpected_adapter_state_key_is_refused(tmp_path):
    written = save_adapter_dir(tmp_path, "w", ExtraKeyMembers(), 0, RUN)
    with pytest.raises(RuntimeError, match="unexpected .*k_proj"):
        load_adapter_dir(written.path, FakeMembers(), 0, RUN, TARGETS, load_optimizer=True)


def _one_rank_cannot_read(rank, world_size, port, *, root):
    init_gloo(rank, world_size, port=port)
    path = Path(root) if rank == 0 else Path(root, "missing")
    with pytest.raises(RuntimeError, match="adapter checkpoint failed: rank 1"):
        load_adapter_dir(path, FakeMembers(), 0, RUN, TARGETS, load_optimizer=False)
    dist.destroy_process_group()


def test_a_load_failure_on_one_rank_fails_every_rank_instead_of_hanging(tmp_path):
    write_peft_adapter(tmp_path, {f"{MODULE}.{k}": v for k, v in FULL.items()}, RUN)
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
