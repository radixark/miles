from types import SimpleNamespace

import miles.rollout.multi_lora.data_source as data_source_module
from miles.rollout.data_source import RolloutDataSource
from miles.rollout.multi_lora.data_source import MultiLoRAAsyncDataSource


def _adapter(slot: int):
    return SimpleNamespace(
        slot=slot,
        config=SimpleNamespace(rm_type=None, custom_rm_path=None, metadata={}),
    )


def test_parent_assigns_canonical_identity_across_child_sources(monkeypatch):
    args = SimpleNamespace(rollout_global_dataset=False, n_samples_per_prompt=2)
    data_source = MultiLoRAAsyncDataSource(args)
    data_source.sources = {
        "A": RolloutDataSource(args),
        "B": RolloutDataSource(args),
    }
    data_source.source_queue.extend(["A", "B"])
    snapshot = {
        "active": {"A": _adapter(slot=0), "B": _adapter(slot=1)},
        "retiring": {},
    }
    monkeypatch.setattr(data_source_module, "fetch_snapshot", lambda: snapshot)

    group_a = data_source.get_samples()[0]
    group_b = data_source.get_samples()[0]

    assert [sample.index for sample in group_a] == [0, 1]
    assert [sample.index for sample in group_b] == [0, 1]
    assert {sample.group_index for sample in group_a} == {0}
    assert {sample.group_index for sample in group_b} == {1}
    assert [sample.rollout_id for sample in group_a] == [0, 1]
    assert [sample.rollout_id for sample in group_b] == [2, 3]
    assert {sample.adapter.name for sample in group_a} == {"A"}
    assert {sample.adapter.name for sample in group_b} == {"B"}
