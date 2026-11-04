import pytest
import torch
import torch.distributed as dist
from tests.fast.dist_utils import init_gloo, run_multiprocess

from miles.backends.training_utils.checkpoint.adapter.adapter_dir import HeadRows
from miles.backends.training_utils.checkpoint.adapter.probed_slots import HfPiece, ProbedAdapterSlots, RowMapping
from miles.utils.distributed_utils import init_gloo_group
from miles.utils.lora.hf_lora_targets import get_hf_lora_targets

TARGETS = get_hf_lora_targets({"model_type": "qwen3", "num_hidden_layers": 1, "vocab_size": 8})


class _Slots(ProbedAdapterSlots):
    """A backend that only probes and names its parameters, leaving the rest to the base."""

    def __init__(self, params, row_mapping):
        self._params = params
        super().__init__(row_mapping, TARGETS, iterator=None)

    def slot_parameters(self, slot):
        return self._params


def _base_sets_up(rank, world_size, port):
    init_gloo(rank, world_size, port=port)
    init_gloo_group()
    head = torch.zeros(6, 2)
    mapping = RowMapping([HfPiece("lm_head.lora_B", (6, 2), (0, 0), torch.arange(12).view(6, 2))], (), ("lm_head",))

    slots = _Slots([head], mapping)

    assert slots.head_rows == HeadRows("lm_head.lora_B", max_source_rows=8)
    assert [(s.key, s.global_shape) for s in slots.get_local_shard_specs(0)] == [("lm_head.lora_B", (6, 2))]
    dist.destroy_process_group()


def test_the_base_sets_up_specs_and_head_rows_from_the_probe():
    run_multiprocess(_base_sets_up, world_size=1)


def test_a_backend_without_slot_parameters_cannot_be_built():
    class _NoParams(ProbedAdapterSlots):
        pass

    mapping = RowMapping([], (), ())
    with pytest.raises(TypeError, match="abstract method 'slot_parameters'|abstract methods? .*slot_parameters"):
        _NoParams(mapping, TARGETS, iterator=None)
