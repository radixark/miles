"""``save_adapter_dir`` -> commit -> ``load_adapter_dir`` restores weights, moments and step under a new split."""

import tempfile
from functools import partial
from pathlib import Path

import torch
import torch.distributed as dist
from tests.fast.dist_utils import init_gloo, run_multiprocess

from miles.backends.training_utils.checkpoint.adapter.adapter_dir import load_adapter_dir, save_adapter_dir
from miles.backends.training_utils.checkpoint.adapter.adapter_exports import write_peft_adapter
from miles.backends.training_utils.checkpoint.adapter.adapter_state import AdapterState, LocalShardSpec
from miles.backends.training_utils.checkpoint.adapter.commit import commit_adapter_dir, resolve_adapter_dir
from miles.backends.training_utils.checkpoint.adapter.parametrization import AdapterParametrization
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
