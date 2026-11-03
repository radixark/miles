from types import SimpleNamespace

import pytest
import torch

from miles.backends.megatron_utils.lora.checkpoint import LayerWiseSlotMembers


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
