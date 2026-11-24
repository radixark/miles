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


async def test_a_failed_delete_is_logged_and_the_run_goes_on(caplog):
    executor, _ = make_executor({5: [sample_owning("a")]})

    def delete(refs):
        raise RuntimeError("Mooncake remove failed")

    executor._delete_store_refs = delete
    await executor.get(rollout_id=5)
    await executor.release(5)

    assert "Failed to delete 1 output store refs" in caplog.text
    assert executor._unreleased_store_refs == {}


async def test_refs_without_a_store_client_fail_loudly():
    executor, _ = make_executor({5: [sample_owning("a")]})
    executor._delete_store_refs = None
    await executor.get(rollout_id=5)

    with pytest.raises(StorePlanError, match="no store client"):
        await executor.release(5)


def run_end_args(*, ci_test=True, fully_async=False, partial_rollout=False):
    return make_args(ci_test=ci_test, fully_async=fully_async, partial_rollout=partial_rollout)


@pytest.mark.parametrize(
    ("args", "fails"),
    [
        (run_end_args(), True),
        (run_end_args(fully_async=True), False),
        (run_end_args(partial_rollout=True), False),
        (run_end_args(ci_test=False), False),
    ],
)
def test_keys_left_at_the_end_fail_only_a_synchronous_ci_run(monkeypatch, caplog, args, fails):
    """A fully-async run ends with groups buffered or in flight, a partial rollout with aborted groups in its buffer."""
    executor, _ = make_executor({})
    executor.args = args
    monkeypatch.setattr(rollout_executor_module, "sweep_output_store", lambda args: 4)

    if fails:
        with pytest.raises(AssertionError, match="Swept 4 output store keys"):
            executor._sweep_output_store()
    else:
        executor._sweep_output_store()
        assert "Swept 4 output store keys" in caplog.text


def test_a_run_that_left_nothing_logs_nothing(monkeypatch, caplog):
    executor, _ = make_executor({})
    executor.args = run_end_args()
    monkeypatch.setattr(rollout_executor_module, "sweep_output_store", lambda args: 0)

    executor._sweep_output_store()

    assert "Swept" not in caplog.text


def _fail_the_generation(rollout_id):
    raise RuntimeError("engine died")


@pytest.mark.parametrize("stopped_in", ["train", "generation"])
async def test_a_run_that_stopped_before_training_a_rollout_only_logs_what_it_left(monkeypatch, caplog, stopped_in):
    """Its error is the crash itself; the untrained rollout's objects are expected, not a leak."""
    if stopped_in == "train":
        executor, _ = make_executor({5: [sample_owning("a")]})
        await executor.get(rollout_id=5)
    else:
        executor, _ = make_executor({5: [sample_owning("a")]}, during_get=_fail_the_generation)
        with pytest.raises(RuntimeError, match="engine died"):
            await executor.get(rollout_id=5)
    executor.args = run_end_args()
    monkeypatch.setattr(rollout_executor_module, "sweep_output_store", lambda args: 4)

    executor._sweep_output_store()

    assert "Swept 4 output store keys" in caplog.text


def _fail_the_sweep(args):
    raise RuntimeError("Mooncake store setup failed: -1")


@pytest.mark.parametrize(
    ("args", "stopped_early", "fails"),
    [(run_end_args(), False, True), (run_end_args(ci_test=False), False, False), (run_end_args(), True, False)],
)
async def test_a_failed_sweep_fails_only_a_ci_run_that_trained_every_rollout(
    monkeypatch, caplog, args, stopped_early, fails
):
    """Like a failed delete, a failed sweep only leaks objects nothing reads; CI still wants to see it."""
    executor, _ = make_executor({5: [sample_owning("a")]})
    if stopped_early:
        await executor.get(rollout_id=5)
    executor.args = args
    monkeypatch.setattr(rollout_executor_module, "sweep_output_store", _fail_the_sweep)

    if fails:
        with pytest.raises(RuntimeError, match="setup failed"):
            executor._sweep_output_store()
    else:
        executor._sweep_output_store()
        assert "Failed to sweep the output store" in caplog.text
