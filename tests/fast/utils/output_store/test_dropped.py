import numpy as np
from tests.fast.utils.output_store.test_plan import ref

from miles.utils.output_store import dropped
from miles.utils.output_store.dropped import (
    record_dropped_samples,
    regenerated_on_restore,
    store_refs_of,
    take_dropped_refs,
)
from miles.utils.output_store.plan import StorePlan
from miles.utils.types import Sample


def sample_owning(*keys: str) -> Sample:
    return Sample(rollout_store_plan=StorePlan(fields={}, owned_refs=tuple(ref(key) for key in keys)))


def test_store_refs_of_walks_nested_groups_and_dedups_by_key():
    shared = sample_owning("a", "b")
    groups = [[shared, sample_owning("b", "c")], [[Sample()], sample_owning("d")]]

    assert sorted(store_refs_of(groups)) == ["a", "b", "c", "d"]
    assert store_refs_of([Sample()]) == {}


def test_take_hands_over_what_was_recorded_once(monkeypatch):
    monkeypatch.setattr(dropped, "_dropped", {})
    record_dropped_samples([sample_owning("a")])
    record_dropped_samples([[sample_owning("a", "b")]])

    assert sorted(take_dropped_refs()) == ["a", "b"]
    assert take_dropped_refs() == {}


def test_restore_regenerates_refs_always_and_inline_replay_only_under_the_switch():
    inline = Sample(rollout_routed_experts=np.zeros((1, 1, 1), dtype=np.int32))

    assert regenerated_on_restore([[sample_owning("a")]], use_output_store=False)
    assert not regenerated_on_restore([[inline], [Sample()]], use_output_store=False)
    assert regenerated_on_restore([[inline], [Sample()]], use_output_store=True)
    assert not regenerated_on_restore([[Sample()]], use_output_store=True)
