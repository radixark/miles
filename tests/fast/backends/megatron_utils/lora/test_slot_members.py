from types import SimpleNamespace

import pytest
import torch

from miles.backends.megatron_utils.lora.checkpoint import LayerWiseSlotMembers
from miles.backends.training_utils.checkpoint.adapter.adapter_dir import HeadRows
from miles.backends.training_utils.checkpoint.adapter.adapter_state import AdapterState, LocalShardSpec


def _members(param: torch.Tensor, master: torch.Tensor) -> LayerWiseSlotMembers:
    slots = SimpleNamespace(
        slot_parameters=lambda slot: [param],
        local_shard_specs=lambda slot, holds: [SimpleNamespace(local_shape=param.shape)],
        local_pieces=lambda slot, tensors, holds: list(tensors),
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


def test_the_members_declare_their_slots_head_rows():
    head_rows = HeadRows("lm_head.lora_B", max_source_rows=8)
    members = LayerWiseSlotMembers(SimpleNamespace(head_rows=head_rows), {})
    assert members.head_rows is head_rows
