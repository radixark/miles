import multiprocessing
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from tests.fast.dist_utils import init_gloo, run_multiprocess

from miles.backends.megatron_utils.lora.checkpoint import LayerWiseSlotMembers
from miles.backends.training_utils.checkpoint.adapter.adapter_state import (
    AdapterState,
    LocalShardSpec,
    raise_if_any_rank_failed,
)
from miles.utils.distributed_utils import init_gloo_group


def _members(param: torch.Tensor, master: torch.Tensor) -> LayerWiseSlotMembers:
    slots = SimpleNamespace(
        slot_parameters=lambda slot: [param],
        local_shard_specs=lambda slot, holds: [SimpleNamespace(local_shape=param.shape)],
        piece_parameter=lambda slot, holds: [(0, torch.arange(param.numel()).view(param.shape))],
    )
    child = SimpleNamespace(
        optimizer=SimpleNamespace(param_groups=[{"step": 1}], state={master: {}}),
        float16_groups=[[param]],
        fp32_from_float16_groups=[[master]],
        fp32_from_fp32_groups=[[]],
    )
    return LayerWiseSlotMembers(slots, {0: SimpleNamespace(chained_optimizers=[child])})


def test_a_slot_whose_master_holds_nan_still_exports():
    master = torch.tensor([[1.0, float("nan")]])
    state = _members(master.to(torch.bfloat16), master).export_adapter_state(0)
    assert torch.equal(state.weights[0].isnan(), master.isnan())


def test_a_bf16_parameter_that_is_not_its_master_is_refused():
    master = torch.tensor([[1.0, 2.0]])
    with pytest.raises(AssertionError, match="is not its master"):
        _members(torch.tensor([[1.0, 3.0]], dtype=torch.bfloat16), master).export_adapter_state(0)


class _NotHeld(torch.Tensor):
    """A parameter another rank holds: exporting this rank's pieces must never read or convert it."""

    def detach(self):
        raise AssertionError("a parameter this rank does not hold was read")

    def float(self):
        raise AssertionError("a parameter this rank does not hold was converted")


def test_an_export_reads_only_the_parameters_this_rank_holds():
    master = torch.tensor([[1.0, 2.0]])
    held, elsewhere = master.to(torch.bfloat16), torch.zeros(3, 4).as_subclass(_NotHeld)
    slots = SimpleNamespace(
        slot_parameters=lambda slot: [held, elsewhere],
        local_shard_specs=lambda slot, holds: [SimpleNamespace(local_shape=held.shape)],
        piece_parameter=lambda slot, holds: [(0, torch.arange(2).view(1, 2))],
    )
    child = SimpleNamespace(
        optimizer=SimpleNamespace(param_groups=[{"step": 1}], state={master: {}}),
        float16_groups=[[held]],
        fp32_from_float16_groups=[[master]],
        fp32_from_fp32_groups=[[]],
    )
    state = LayerWiseSlotMembers(slots, {0: SimpleNamespace(chained_optimizers=[child])}).export_adapter_state(0)
    assert torch.equal(state.weights[0], master)


def _importing_members(master: torch.Tensor, adam: dict, group: dict, offsets: torch.Tensor, calls: list):
    param = master.to(torch.bfloat16)
    slots = SimpleNamespace(slot_parameters=lambda slot: [param], piece_parameter=lambda slot, holds: [(0, offsets)])
    child = SimpleNamespace(
        optimizer=SimpleNamespace(param_groups=[group], state={master: adam}),
        float16_groups=[[param]],
        fp32_from_float16_groups=[[master]],
        fp32_from_fp32_groups=[[]],
        _copy_main_params_to_model_params=lambda: calls.append("copy"),
    )
    optimizer = SimpleNamespace(chained_optimizers=[child], allgather_params=lambda: calls.append("allgather"))
    return LayerWiseSlotMembers(slots, {0: optimizer})


def _state(step, *pieces):
    weight, exp_avg, exp_avg_sq = pieces
    spec = LocalShardSpec("m.lora_A", tuple(weight.shape), (0,), tuple(weight.shape))
    return AdapterState([spec], [weight], [exp_avg], [exp_avg_sq], step=step)


def test_an_import_writes_the_pieces_at_their_offsets_and_builds_missing_moments():
    master, adam, group, calls = torch.zeros(2, 3), {}, {}, []
    offsets = torch.tensor([1, 4])
    members = _importing_members(master, adam, group, offsets, calls)
    members.import_adapter_state(
        0, _state(7, torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]), torch.tensor([5.0, 6.0]))
    )
    assert calls == ["copy"], "the import writes only this rank's pieces"
    members.finish_import_adapter_state(0)

    # Adam state appears at the first step; an import before it creates the moments
    assert torch.equal(master.view(-1), torch.tensor([0.0, 1.0, 0.0, 0.0, 2.0, 0.0]))
    assert torch.equal(adam["exp_avg"].view(-1), torch.tensor([0.0, 3.0, 0.0, 0.0, 4.0, 0.0]))
    assert torch.equal(adam["exp_avg_sq"].view(-1), torch.tensor([0.0, 5.0, 0.0, 0.0, 6.0, 0.0]))
    assert group["step"] == 7
    assert calls == ["copy", "allgather"], "the bf16 parameters are refreshed from the masters on every rank"


def test_an_import_without_a_step_drops_the_groups_step():
    master, adam, group, calls = torch.zeros(4), {}, {"step": 3}, []
    members = _importing_members(master, adam, group, torch.arange(4), calls)
    members.import_adapter_state(0, _state(None, torch.ones(4), torch.zeros(4), torch.zeros(4)))
    assert "step" not in group, "a source saved before any step restarts Adam's bias correction"


def _one_rank_cannot_import(rank, world_size, port):
    init_gloo(rank, world_size, port=port)
    init_gloo_group()
    master, calls = torch.zeros(2), []
    members = _importing_members(master, {}, {}, torch.arange(2), calls)
    # allgather_params is collective: here a barrier on the default group
    members._optimizers[0].allgather_params = dist.barrier
    pieces = [torch.ones(2), torch.zeros(2), torch.zeros(2)]
    state = _state(1, *pieces)
    if rank == 1:
        # one piece more than this rank holds: its write fails before the collective
        state = AdapterState(
            state.shard_specs * 2, state.weights * 2, state.exp_avgs * 2, state.exp_avg_sqs * 2, step=1
        )
    error = None
    try:
        members.import_adapter_state(0, state)
    except Exception as exc:
        error = exc
    # what load_adapter_dir does after import_adapter_state
    with pytest.raises(RuntimeError, match="rank 1: ValueError"):
        raise_if_any_rank_failed(error)
    dist.destroy_process_group()


def test_a_rank_whose_import_fails_fails_every_rank_before_the_parameter_gather():
    process = multiprocessing.get_context("spawn").Process(target=run_multiprocess, args=(_one_rank_cannot_import,))
    process.start()
    process.join(timeout=60)
    hung = process.is_alive()
    process.kill()
    assert not hung and process.exitcode == 0


def _one_rank_cannot_list_its_ownership(rank, world_size, port):
    init_gloo(rank, world_size, port=port)
    init_gloo_group()
    master = torch.zeros(2)
    members = _members(master.to(torch.bfloat16), master)

    def gathered_specs(slot, holds):
        # the real specs gather every rank's pieces over Gloo
        dist.all_gather_object([None] * world_size, sorted(holds), group=dist.group.WORLD)
        return []

    members._slots.local_shard_specs = gathered_specs
    if rank == 1:
        members._slots.slot_parameters = lambda slot: []
    error = None
    try:
        members.get_local_shard_specs(0)
    except Exception as exc:
        error = exc
    # what load_adapter_dir does after get_local_shard_specs
    with pytest.raises(RuntimeError, match="rank 1: KeyError"):
        raise_if_any_rank_failed(error)
    dist.destroy_process_group()


def test_a_rank_that_cannot_list_its_ownership_fails_every_rank_before_the_spec_gather():
    process = multiprocessing.get_context("spawn").Process(
        target=run_multiprocess, args=(_one_rank_cannot_list_its_ownership,)
    )
    process.start()
    process.join(timeout=60)
    hung = process.is_alive()
    process.kill()
    assert not hung and process.exitcode == 0
