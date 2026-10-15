from functools import partial
from types import SimpleNamespace

import pytest
import torch

from miles.ray.rollout.train_data_conversion import split_train_data_by_dp_scheduled_raw
from miles.tinker.runtime import MilesBackend


class _Store:
    def __init__(self):
        self.values = {}
        self.next_ref = 0

    def put(self, *, value, value_spec):
        ref = self.next_ref
        self.next_ref += 1
        self.values[ref] = value
        return ref

    def remove(self, ref):
        del self.values[ref]


@pytest.mark.parametrize("count", [1, 5])
@pytest.mark.parametrize("cp_size", [1, 2])
@pytest.mark.parametrize("balance_data", [False, True])
async def test_scheduled_batches_preserve_targets_slots_and_output_order(monkeypatch, count, cp_size, balance_data):
    args = SimpleNamespace(
        use_dynamic_batch_size=True,
        max_tokens_per_gpu=100 // cp_size,
        balance_data=balance_data,
        balance_by_flops=False,
        global_batch_size=1,
        enable_sample_ownership_checker=False,
        multi_lora_n_adapters=3,
    )
    schedule = partial(
        split_train_data_by_dp_scheduled_raw,
        args,
        train_parallel_config=dict(dp_size=2, cp_size=cp_size, vpp_size=1, microbatch_group_size_per_vp_stage=1),
    )
    lengths = [60, 60, 40, 20, 20][:count]
    slots = [2, 0, 1, 0, 2][:count]
    datums = [
        (slot, dict(tokens=[i] * length, target_tokens=[i + 1] * (length - 1), target_len=length - 1))
        for i, (length, slot) in enumerate(zip(lengths, slots, strict=True))
    ]
    store = _Store()
    monkeypatch.setattr("miles.tinker.runtime.object_store.get_instance", lambda: store)

    class _Trainer:
        async def forward_backward(self, *, batch_id, data_ref):
            shards = [store.values[ref] for ref in data_ref]
            assert len(shards) == 2
            assert shards[0]["num_microbatches"] == shards[1]["num_microbatches"]
            assert sum(len(shard["tokens"]) for shard in shards) == max(count, 2)
            outputs = []
            for shard in shards:
                assert sorted(i for mb in shard["micro_batch_indices"] for i in mb) == list(
                    range(len(shard["tokens"]))
                )
                for mb in shard["micro_batch_indices"]:
                    assert sum(len(shard["tokens"][i]) for i in mb) <= 100
                    mb_slots = [shard["adapter_slots"][i] for i in mb]
                    assert mb_slots == sorted(mb_slots)
                per_datum = []
                for local, index in enumerate(shard["sample_indices"]):
                    slot, datum = datums[min(index, count - 1)]
                    assert shard["tokens"][local] == datum["tokens"]
                    assert shard["target_tokens"][local] == datum["target_tokens"]
                    assert shard["adapter_slots"][local] == slot
                    assert shard["loss_masks"][local] == [int(index < count)] * datum["target_len"]
                    per_datum.append(dict(sample_index=index, loss=index, logprobs=torch.tensor([float(index)])))
                outputs.append({"per_datum": per_datum})
            return outputs[::-1]

    backend = MilesBackend(_Trainer(), "http://router", dp_size=2, schedule_train_data=schedule)
    outputs = await backend.forward_backward(0, datums, "cross_entropy", {})
    assert outputs == [dict(loss=float(i), logprobs=[float(i)]) for i in range(count)]
    assert not store.values


async def test_scheduled_shards_are_released_on_trainer_failure(monkeypatch):
    store = _Store()
    monkeypatch.setattr("miles.tinker.runtime.object_store.get_instance", lambda: store)

    class _Trainer:
        async def forward_only(self, **kwargs):
            raise RuntimeError("trainer failed")

    backend = MilesBackend(_Trainer(), "http://router", schedule_train_data=lambda data: [data, data])
    with pytest.raises(RuntimeError, match="trainer failed"):
        await backend._call_trainer("forward_only", 0, {"sample_indices": [0, 1]})
    assert not store.values
