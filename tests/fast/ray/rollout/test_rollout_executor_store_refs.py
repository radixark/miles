"""The rollout executor deletes a rollout's output store objects once that rollout has trained."""

import pytest
from tests.fast.ray.rollout.conftest import make_args
from tests.fast.utils.output_store.test_dropped import sample_owning

from miles.ray.rollout import rollout_executor as rollout_executor_module
from miles.ray.rollout.output_snapshotter import _RolloutExecutorOutputSnapshotter
from miles.ray.rollout.rollout_executor import RolloutExecutor
from miles.utils.object_store import _MooncakeStoreObjectRef
from miles.utils.output_store import dropped
from miles.utils.output_store.dropped import record_dropped_samples
from miles.utils.output_store.plan import StorePlanError
from miles.utils.types import Sample


class _Store:
    def put(self, *, value, value_spec):
        return _MooncakeStoreObjectRef(payload=value)


@pytest.fixture(autouse=True)
def _fakes(monkeypatch):
    monkeypatch.setattr(dropped, "_dropped", {})
    monkeypatch.setattr(rollout_executor_module, "convert_samples_to_train_data", lambda *args, **kwargs: {})
    monkeypatch.setattr(rollout_executor_module.object_store, "get_instance", _Store)


def make_executor(batches: dict[int, list[Sample]], *, during_get=None) -> tuple[RolloutExecutor, list[list[str]]]:
    """An executor whose rollout ``k`` returns ``batches[k]``, after running ``during_get(k)``."""
    deleted: list[list[str]] = []
    executor = RolloutExecutor.__new__(RolloutExecutor)
    executor.args = make_args(delay_split_train_data_by_dp=True)
    executor.last_get_rollout_id_of_model_id = {}
    executor.custom_convert_samples_to_train_data_func = None
    executor.custom_reward_post_process_func = None
    executor._output_snapshotter = _RolloutExecutorOutputSnapshotter(args=executor.args)
    executor._unreleased_store_refs = {}
    executor._delete_store_refs = lambda refs: deleted.append(sorted(ref.key for ref in refs))

    async def generate(*, rollout_id, trainer_model_id):
        if during_get is not None:
            during_get(rollout_id)
        return batches[rollout_id], {}

    executor._generate_rollout_data = generate
    return executor, deleted


async def test_release_deletes_the_trained_and_dropped_refs_of_its_rollout_once():
    executor, deleted = make_executor(
        {5: [sample_owning("trained", "shared"), Sample()]},
        during_get=lambda rollout_id: record_dropped_samples([sample_owning("dropped", "shared")]),
    )

    await executor.get(rollout_id=5)
    assert deleted == []
    await executor.release(5)
    await executor.release(5)

    assert deleted == [["dropped", "shared", "trained"]]


async def test_drops_of_the_next_rollout_wait_for_it_to_train():
    # train_async: get(6) runs while rollout 5 trains, so release(5) comes after get(6)'s drops.
    executor, deleted = make_executor(
        {5: [sample_owning("a")], 6: [sample_owning("b")]},
        during_get=lambda rollout_id: record_dropped_samples([sample_owning(f"dropped-{rollout_id}")]),
    )
    await executor.get(rollout_id=5)
    record_dropped_samples([sample_owning("dropped-between")])
    await executor.get(rollout_id=6)

    await executor.release(5)
    assert deleted == [["a", "dropped-5"]]
    await executor.release(6)
    assert deleted[-1] == ["b", "dropped-6", "dropped-between"]


async def test_a_rollout_without_refs_deletes_nothing_and_needs_no_client():
    executor, deleted = make_executor({5: [Sample()]})
    executor._delete_store_refs = None

    await executor.get(rollout_id=5)
    await executor.release(5)

    assert executor._unreleased_store_refs == {}


async def test_refs_without_a_store_client_fail_loudly():
    executor, _ = make_executor({5: [sample_owning("a")]})
    executor._delete_store_refs = None
    await executor.get(rollout_id=5)

    with pytest.raises(StorePlanError, match="no store client"):
        await executor.release(5)
