import asyncio
from argparse import Namespace
from collections import defaultdict
from types import SimpleNamespace
from typing import Any

import pytest

from miles.ray.rollout import rollout_executor as rollout_executor_module
from miles.ray.rollout.rollout_executor import RolloutExecutor
from miles.rollout.base_types import RolloutFnTrainInput
from miles.utils.timer import Timer
from miles.utils.weight_version import (
    MAX_ROLLOUTS_WITHOUT_PUBLISHED_WEIGHT_VERSION,
    assert_weight_version_is_published,
)


@pytest.fixture(autouse=True)
def _quiet_rollout_pipeline(monkeypatch):
    Timer().timers.clear()
    Timer().start_time.clear()
    monkeypatch.setattr(rollout_executor_module, "save_debug_rollout_data", lambda *a, **kw: None)
    monkeypatch.setattr(rollout_executor_module, "convert_samples_to_train_data", lambda *a, **kw: {})
    monkeypatch.setattr(rollout_executor_module, "split_train_data_by_dp", lambda *a, **kw: None)
    monkeypatch.setattr(rollout_executor_module, "assert_weight_version_is_published", lambda *a, **kw: None)
    yield
    Timer().timers.clear()
    Timer().start_time.clear()


def _make_executor() -> RolloutExecutor:
    executor = RolloutExecutor.__new__(RolloutExecutor)
    executor.args = Namespace(
        delay_split_train_data_by_dp=False,
        indep_dp=False,
        load_debug_rollout_data=None,
        ci_inject_rollout_data_path=None,
        debug_rollout_only=False,
        debug_train_only=False,
        debug_skip_weight_update=False,
        lora_rank=0,
    )
    executor.data_source = Namespace()
    executor.custom_convert_samples_to_train_data_func = None
    executor.custom_reward_post_process_func = None
    executor._weight_versions_of_model_id = {}
    executor._train_parallel_configs_of_model_id = {}
    executor._rollouts_since_publish_of_model_id = defaultdict(int)
    executor.rollout_id = -1
    return executor


def _record_generate_inputs(executor: RolloutExecutor, monkeypatch) -> list[RolloutFnTrainInput]:
    received: list[RolloutFnTrainInput] = []
    executor.use_experimental_refactor = True
    executor.generate_rollout = object()

    def _call_rollout_function(rollout_function, rollout_input: RolloutFnTrainInput):
        received.append(rollout_input)
        return SimpleNamespace(samples=[], metrics=None)

    monkeypatch.setattr(rollout_executor_module, "call_rollout_function", _call_rollout_function)
    monkeypatch.setattr(rollout_executor_module, "assert_samples_weight_version_sane", lambda *a, **kw: None)
    return received


def _record_postprocess_configs(monkeypatch) -> list[dict[str, Any]]:
    seen: list[dict[str, Any]] = []

    def _postprocess_rollout_data(args, data, *, train_parallel_config: dict[str, Any]):
        seen.append(train_parallel_config)
        return data, None

    monkeypatch.setattr(rollout_executor_module, "postprocess_rollout_data", _postprocess_rollout_data)
    return seen


def _record_logged_model_ids(monkeypatch) -> list[str | None]:
    logged: list[str | None] = []
    monkeypatch.setattr(
        rollout_executor_module,
        "log_rollout_data",
        lambda *a, trainer_model_id=None, **kw: logged.append(trainer_model_id),
    )
    return logged


class TestPerPolicyKeying:
    def test_every_policy_keeps_its_own_weight_version(self):
        """One shared version would let a policy's samples be judged against another policy's weights."""
        executor = _make_executor()

        executor.set_weight_version(3, trainer_model_id="a")
        executor.set_weight_version(7, trainer_model_id="b")

        assert executor._weight_versions_of_model_id == {"a": 3, "b": 7}

    def test_a_single_policy_run_keys_everything_under_none(self):
        """None is the single-policy key, so the base path must not grow a name of its own."""
        executor = _make_executor()

        executor.set_weight_version(3)
        executor.set_train_parallel_config({"dp_size": 2})

        assert executor._weight_versions_of_model_id == {None: 3}
        assert executor._train_parallel_configs_of_model_id == {None: {"dp_size": 2}}

    def test_a_version_going_backwards_for_one_policy_is_still_refused(self):
        """The regression check must compare a policy against itself, not against whoever published last."""
        executor = _make_executor()
        executor.set_weight_version(7, trainer_model_id="a")

        executor.set_weight_version(9, trainer_model_id="b")

        with pytest.raises(AssertionError, match="went backwards"):
            executor.set_weight_version(5, trainer_model_id="a")

    async def test_each_policy_is_sharded_by_its_own_parallel_config(self, monkeypatch):
        """Splitting a policy's batch by another policy's dp size hands its ranks the wrong shards."""
        executor = _make_executor()
        executor.set_train_parallel_config({"dp_size": 8}, trainer_model_id="a")
        executor.set_train_parallel_config({"dp_size": 4}, trainer_model_id="b")
        seen: list[dict] = []
        monkeypatch.setattr(
            rollout_executor_module, "split_train_data_by_dp", lambda args, data, config: seen.append(config)
        )
        _record_logged_model_ids(monkeypatch)

        async def _get_rollout_data(rollout_id, trainer_model_id=None):
            return [], None, None

        executor._get_rollout_data = _get_rollout_data

        await executor.get(0, trainer_model_id="b")

        assert seen == [{"dp_size": 4}]

    async def test_a_policy_asks_for_data_against_its_own_weight_version(self, monkeypatch):
        """The rollout function stamps its samples with the version it is told, so the wrong one mislabels a batch."""
        executor = _make_executor()
        executor.set_weight_version(3, trainer_model_id="a")
        executor.set_weight_version(7, trainer_model_id="b")
        executor.set_train_parallel_config({"dp_size": 4}, trainer_model_id="b")
        received = _record_generate_inputs(executor, monkeypatch)
        _record_postprocess_configs(monkeypatch)
        _record_logged_model_ids(monkeypatch)

        await executor.get(0, trainer_model_id="b")

        [rollout_input] = received
        assert (rollout_input.weight_version, rollout_input.trainer_model_id) == (7, "b")

    async def test_the_postprocess_step_uses_the_same_parallel_config_as_the_split(self, monkeypatch):
        """Postprocessing and splitting disagreeing on dp size would group samples one way and shard them another."""
        executor = _make_executor()
        executor.set_train_parallel_config({"dp_size": 8}, trainer_model_id="a")
        executor.set_train_parallel_config({"dp_size": 4}, trainer_model_id="b")
        _record_generate_inputs(executor, monkeypatch)
        postprocessed = _record_postprocess_configs(monkeypatch)
        split: list[dict[str, Any]] = []
        monkeypatch.setattr(
            rollout_executor_module, "split_train_data_by_dp", lambda args, data, config: split.append(config)
        )
        _record_logged_model_ids(monkeypatch)

        await executor.get(0, trainer_model_id="b")

        assert postprocessed == split == [{"dp_size": 4}]

    async def test_a_policy_without_a_parallel_config_fails_loudly(self, monkeypatch):
        """A policy whose trainer never published its layout must name itself instead of sharding by another's."""
        executor = _make_executor()
        executor.set_train_parallel_config({"dp_size": 4}, trainer_model_id="b")
        _record_generate_inputs(executor, monkeypatch)
        _record_postprocess_configs(monkeypatch)
        _record_logged_model_ids(monkeypatch)

        with pytest.raises(KeyError, match="'a'"):
            await executor.get(0, trainer_model_id="a")


class TestRolloutTimerNaming:
    async def test_two_policies_may_be_generating_at_the_same_time(self, monkeypatch):
        """The rollout timer is a process singleton that refuses a second start under the same name."""
        logged = _record_logged_model_ids(monkeypatch)
        executor = _make_executor()
        executor.set_train_parallel_config({"dp_size": 4}, trainer_model_id="a")
        executor.set_train_parallel_config({"dp_size": 4}, trainer_model_id="b")
        both_arrived = asyncio.Event()
        arrivals = 0

        async def _get_rollout_data(rollout_id, trainer_model_id=None):
            nonlocal arrivals
            arrivals += 1
            if arrivals == 2:
                both_arrived.set()
            await asyncio.wait_for(both_arrived.wait(), timeout=5)
            return [], None, None

        executor._get_rollout_data = _get_rollout_data
        for trainer_model_id in ("a", "b"):
            executor.set_train_parallel_config({"dp_size": 1}, trainer_model_id=trainer_model_id)

        await asyncio.wait_for(
            asyncio.gather(executor.get(0, trainer_model_id="a"), executor.get(0, trainer_model_id="b")), timeout=5
        )

        assert sorted(Timer().log_dict()) == ["a/rollout", "b/rollout"]
        assert sorted(logged) == ["a", "b"]

    async def test_a_single_policy_run_keeps_the_timer_and_metric_names_it_had(self, monkeypatch):
        """Every existing dashboard query is written against the unprefixed names."""
        logged = _record_logged_model_ids(monkeypatch)
        executor = _make_executor()
        executor.set_train_parallel_config({"dp_size": 4})

        async def _get_rollout_data(rollout_id, trainer_model_id=None):
            return [], None, None

        executor._get_rollout_data = _get_rollout_data
        executor.set_train_parallel_config({"dp_size": 1})

        await executor.get(0)

        assert list(Timer().log_dict()) == ["rollout"]
        assert logged == [None]


class TestWeightVersionWatchdog:
    @pytest.fixture(autouse=True)
    def _real_watchdog(self, monkeypatch):
        monkeypatch.setattr(
            rollout_executor_module, "assert_weight_version_is_published", assert_weight_version_is_published
        )

    @staticmethod
    def _make_ready_executor(*model_ids: str | None) -> RolloutExecutor:
        executor = _make_executor()
        for model_id in model_ids:
            executor.set_weight_version(1, trainer_model_id=model_id)
            executor.set_train_parallel_config({"dp_size": 4}, trainer_model_id=model_id)

        async def _get_rollout_data(rollout_id, trainer_model_id=None):
            return [], None, None

        executor._get_rollout_data = _get_rollout_data
        return executor

    async def test_one_round_of_more_policies_than_the_threshold_is_not_a_stall(self, monkeypatch):
        """A shared counter turns the fourth policy's first rollout into a false 'nobody published' failure."""
        _record_logged_model_ids(monkeypatch)
        model_ids = ["a", "b", "c", "d"]
        executor = self._make_ready_executor(*model_ids)

        for model_id in model_ids:
            await executor.get(0, trainer_model_id=model_id)

    async def test_a_publishing_policy_does_not_clear_a_frozen_one(self, monkeypatch):
        """Otherwise a policy whose weights never move again is covered by its neighbour's updates."""
        _record_logged_model_ids(monkeypatch)
        executor = self._make_ready_executor("a", "b")

        for rollout_id in range(MAX_ROLLOUTS_WITHOUT_PUBLISHED_WEIGHT_VERSION):
            await executor.get(rollout_id, trainer_model_id="b")
            await executor.get(rollout_id, trainer_model_id="a")
            executor.set_weight_version(rollout_id + 2, trainer_model_id="a")

        with pytest.raises(AssertionError, match="without anyone calling"):
            await executor.get(MAX_ROLLOUTS_WITHOUT_PUBLISHED_WEIGHT_VERSION, trainer_model_id="b")
