"""Reloading a slot restores adapter weights that megatron loads through a ShardedTensorFactory."""

from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from megatron.core.dist_checkpointing.mapping import ShardedTensor, ShardedTensorFactory

from miles.backends.megatron_utils.lora import checkpoint
from miles.utils import distributed_utils


@pytest.fixture
def single_rank_gloo(tmp_path, monkeypatch):
    dist.init_process_group(backend="gloo", init_method=f"file://{tmp_path / 'rendezvous'}", rank=0, world_size=1)
    monkeypatch.setattr(distributed_utils, "GLOO_GROUP", dist.group.WORLD)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")
    try:
        yield
    finally:
        dist.destroy_process_group()


def _build_swiglu_halves(key, tensor, replica_id, flattened_range):
    halves = torch.chunk(tensor, 2, dim=1)
    return [
        ShardedTensor.from_rank_offsets(key, half, (0, 0, 1), (2, index, 2), replica_id=replica_id, prepend_axis_num=1)
        for index, half in enumerate(halves)
    ]


def _merge_swiglu_halves(halves):
    return torch.cat(halves, dim=1)


class _GroupedExpertFc1LoraB(torch.nn.Module):
    """A packed [experts, 2 * ffn, rank] LoRA B saved as two SwiGLU halves, like grouped-expert fc1."""

    def __init__(self, slot: int, weight: torch.Tensor) -> None:
        super().__init__()
        self.key = f"decoder.layers.0.mlp.experts.linear_fc1.adapters.{slot}.linear_out.weight"
        self.weight = torch.nn.Parameter(weight.clone())

    def sharded_state_dict(self) -> dict:
        factory = ShardedTensorFactory(self.key, self.weight, _build_swiglu_halves, _merge_swiglu_halves, replica_id=0)
        return {self.key: factory}


def _slot_optimizer(slot: int, reload_masters=None) -> SimpleNamespace:
    return SimpleNamespace(
        slot=slot, sharded_state=lambda weights, is_loading: {}, reload_masters=reload_masters or (lambda: None)
    )


def test_weights_only_load_restores_factory_weights_before_reloading_masters(single_rank_gloo, tmp_path):
    trained = torch.randn(4, 8, 2)
    path = str(tmp_path / "slot")
    checkpoint.save_slot([_GroupedExpertFc1LoraB(1, trained)], _slot_optimizer(1), path)

    reloaded = _GroupedExpertFc1LoraB(0, torch.zeros(4, 8, 2))
    masters_source = []
    optimizer = _slot_optimizer(0, reload_masters=lambda: masters_source.append(reloaded.weight.detach().clone()))
    checkpoint.load_slot([reloaded], optimizer, path, load_optimizer=False)

    torch.testing.assert_close(reloaded.weight.detach(), trained)
    torch.testing.assert_close(masters_source, [trained])
