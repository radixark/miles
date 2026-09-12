from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="stage-a-cpu", labels=[])

import asyncio
from argparse import Namespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
import train_multi_policy as multi_policy_driver
from tests.fast.fixtures.args_fixtures import parser_defaults
from tests.fast.fixtures.megatron_config_fixtures import encode_megatron_config
from train_multi_policy import train_multi_policy

from miles.utils.multi_policy.checkpoint_state import MultiPolicyCheckpointState
from miles.utils.multi_policy.utils import TrainerInfo


def _make_args(**overrides: Any) -> Namespace:
    defaults = dict(
        megatron_config=encode_megatron_config("a", "b"),
        num_rollout=2,
        update_weights_interval=1,
        save=None,
        save_interval=None,
        save_trigger_sentinel=None,
        debug_exit_after_rollout=None,
        check_weight_update_equal=False,
        check_weight_update_allow_quant_error=False,
        check_weight_update_selector="all",
        check_weight_update_skip_list=None,
    )
    defaults.update(overrides)
    return Namespace(**{**parser_defaults(), **defaults})


def _make_trainers(model_ids, handles=None, start_rollout_ids=None) -> dict[str, TrainerInfo]:
    handles = {model_id: AsyncMock() for model_id in model_ids} if handles is None else handles
    start_rollout_ids = start_rollout_ids or {}
    return {
        model_id: TrainerInfo(model_id=model_id, start_rollout_id=start_rollout_ids.get(model_id, 0), handle=handle)
        for model_id, handle in handles.items()
    }


async def _run(
    args,
    *,
    model_ids: tuple[str, ...] = ("a", "b"),
    trainers: dict[str, AsyncMock] | None = None,
    start_rollout_ids: dict[str, int] | None = None,
    rollout_executor: AsyncMock | None = None,
) -> dict:
    """Drive the whole driver with every out-of-process dependency stubbed out."""
    infos = _make_trainers(model_ids, handles=trainers, start_rollout_ids=start_rollout_ids)
    context = dict(
        trainers={model_id: info.handle for model_id, info in infos.items()},
        inference_controller=AsyncMock(),
        rollout_executor=AsyncMock() if rollout_executor is None else rollout_executor,
    )
    multi_policy_driver.create_trainers.return_value = infos
    multi_policy_driver.create_rollout_components.return_value = (
        context["inference_controller"],
        context["rollout_executor"],
        None,
    )
    await asyncio.wait_for(train_multi_policy(args), timeout=30)
    return context


@pytest.fixture(autouse=True)
def _stub_driver_environment(monkeypatch):
    """Everything the driver reaches outside its own loop: cluster, tracking and logging."""
    for name in (
        "init_orchestration_script",
        "define_policy_metric_groups",
        "maybe_start_api_server",
        "maybe_start_mini_ft_controller",
        "validate_multi_policy_args",
        "assert_consistent_restore",
    ):
        monkeypatch.setattr(multi_policy_driver, name, lambda *a, **kw: None)
    monkeypatch.setattr(multi_policy_driver, "create_trainers", AsyncMock(return_value={}))
    monkeypatch.setattr(multi_policy_driver, "create_rollout_components", AsyncMock())


@pytest.fixture(autouse=True)
def _no_object_store(monkeypatch):
    monkeypatch.setattr(multi_policy_driver, "remove_rollout_data_refs", lambda args, ref: None)


@pytest.fixture(autouse=True)
def _stub_update_weights(monkeypatch):
    monkeypatch.setattr(multi_policy_driver, "update_weights", AsyncMock())


class TestInitialWeightPublication:
    async def test_every_policy_compares_its_engines_against_its_own_trainer(self):
        """--ci-test asks for this comparison, and running it for one policy would leave the others unchecked."""
        context = await _run(_make_args(num_rollout=0, check_weight_update_equal=True))

        compared = [call.kwargs["model_id"] for call in context["inference_controller"].check_weights.await_args_list]
        assert sorted(compared) == ["a", "b"]

    async def test_a_run_that_does_not_ask_for_the_comparison_does_not_pay_for_it(self):
        """The comparison walks every parameter, so it stays off unless the run turns it on."""
        context = await _run(_make_args(num_rollout=0))

        context["inference_controller"].check_weights.assert_not_awaited()


class TestRunPolicies:
    async def test_every_policy_drains_and_updates_only_its_own_model(self, monkeypatch):
        """Two policies sharing one executor must never train on, or publish into, each other's model."""
        updated: list[tuple[str, int]] = []
        monkeypatch.setattr(
            multi_policy_driver,
            "update_weights",
            AsyncMock(
                side_effect=lambda *a, rollout_id=None, trainer_model_id=None, **kw: updated.append(
                    (trainer_model_id, rollout_id)
                )
            ),
        )

        context = await _run(_make_args())

        drained = [call.kwargs["trainer_model_id"] for call in context["rollout_executor"].get.await_args_list]
        assert drained.count("a") == 2
        assert set(drained) == {"a", "b"}
        assert [rollout_id for model_id, rollout_id in updated if model_id == "a"] == [None, 0, 1]
        assert {model_id for model_id, _ in updated} == {"a", "b"}

    async def test_a_policy_only_resumes_the_health_probing_of_its_own_engines(self):
        """Resuming the whole fleet here un-pauses probing of a policy that is mid weight broadcast."""
        context = await _run(_make_args(num_rollout=1))

        prepared = context["inference_controller"].prepare_rollout.await_args_list
        assert sorted((call.args[0], call.kwargs["model_id"]) for call in prepared) == [(0, "a"), (0, "b")]

    async def test_two_policies_are_inside_the_executor_at_the_same_time(self):
        """The whole point of one loop per policy is that they overlap; the executor must tolerate it."""
        arrivals = 0
        both_arrived = asyncio.Event()

        async def _get(rollout_id: int, trainer_model_id: str | None = None):
            nonlocal arrivals
            arrivals += 1
            if arrivals == 2:
                both_arrived.set()
            await asyncio.wait_for(both_arrived.wait(), timeout=5)
            return dict(data_ref=None)

        rollout_executor = AsyncMock()
        rollout_executor.get = _get

        await _run(_make_args(num_rollout=1), rollout_executor=rollout_executor)

        assert both_arrived.is_set()

    async def test_a_failing_policy_stops_the_others_instead_of_orphaning_them(self):
        """A surviving loop keeps training and writing checkpoints while the run is already dead."""
        rounds_of_b = 0

        async def _train(rollout_id: int, rollout_data_ref, **kwargs) -> None:
            nonlocal rounds_of_b
            rounds_of_b += 1
            await asyncio.sleep(0.05)

        trainers = {"a": AsyncMock(), "b": AsyncMock()}
        trainers["a"].train = AsyncMock(side_effect=RuntimeError("trainer a died"))
        trainers["b"].train = _train

        with pytest.raises(RuntimeError, match="trainer a died"):
            await _run(_make_args(num_rollout=100), trainers=trainers)

        assert rounds_of_b <= 2

    async def test_a_policy_resumes_from_its_own_position(self):
        """Each trainer restores its own checkpoint, so the policies need not stand at the same rollout."""
        trainers = {"a": AsyncMock(), "b": AsyncMock()}

        await _run(_make_args(num_rollout=3), trainers=trainers, start_rollout_ids=dict(a=0, b=2))

        assert [call.args[0] for call in trainers["a"].train.await_args_list] == [0, 1, 2]
        rounds_of_b = [call.args[0] for call in trainers["b"].train.await_args_list]
        assert rounds_of_b == list(range(2, 2 + len(rounds_of_b)))

    async def test_a_policy_updates_its_weights_on_its_own_interval(self, monkeypatch):
        """The rhythm is counted on the absolute rollout id, so publishing every round would be wrong."""
        updated: list[tuple[str, int]] = []
        monkeypatch.setattr(
            multi_policy_driver,
            "update_weights",
            AsyncMock(
                side_effect=lambda *a, rollout_id=None, trainer_model_id=None, **kw: updated.append(
                    (trainer_model_id, rollout_id)
                )
            ),
        )

        await _run(_make_args(num_rollout=4, update_weights_interval=2))

        assert [rollout_id for model_id, rollout_id in updated if model_id == "a"] == [None, 1, 3]
        rounds_of_b = [rollout_id for model_id, rollout_id in updated if model_id == "b" and rollout_id is not None]
        assert all(rollout_id % 2 == 1 for rollout_id in rounds_of_b)

    async def test_a_debug_run_stops_the_leader_after_its_own_rounds(self):
        """--debug-exit-after-rollout counts from where the policy resumed, not from rollout zero."""
        trainers = {"a": AsyncMock(), "b": AsyncMock()}

        await _run(
            _make_args(num_rollout=10, debug_exit_after_rollout=1), trainers=trainers, start_rollout_ids=dict(a=0, b=5)
        )

        assert [call.args[0] for call in trainers["a"].train.await_args_list] == [0]

    async def test_a_debug_run_leaves_the_followers_running(self):
        """A follower's rounds are the leader's to end, so honouring the flag there would retire it
        from the checkpoint every remaining round waits at."""
        trainers = {"a": AsyncMock(), "b": AsyncMock()}
        trainers["a"].train = _slow_train

        await _run(_make_args(num_rollout=10, debug_exit_after_rollout=1), trainers=trainers)

        assert len(trainers["b"].train.await_args_list) >= 2

    async def test_the_run_ends_when_the_leader_runs_out_of_rounds(self):
        """The leader owns --num-rollout; a follower resuming further back must not extend the run."""
        trainers = {"a": AsyncMock(), "b": AsyncMock()}

        await _run(_make_args(num_rollout=2), trainers=trainers, start_rollout_ids=dict(a=0, b=0))

        assert [call.args[0] for call in trainers["a"].train.await_args_list] == [0, 1]

    async def test_a_follower_is_never_the_one_that_ends_the_run(self):
        """Followers train unbounded rounds, so the run must not stop because one of them reached num_rollout."""
        trainers = {"a": AsyncMock(), "b": AsyncMock()}
        trainers["a"].train = _slow_train

        await _run(_make_args(num_rollout=2), trainers=trainers)

        assert len(trainers["b"].train.await_args_list) >= 2


class TestSaving:
    async def test_the_leader_parks_everybody_and_records_where_they_stood(self, tmp_path):
        """DataSource and RolloutExecutor are global, so their snapshot needs one agreed moment."""
        trainers = {"a": AsyncMock(), "b": AsyncMock()}
        trainers["b"].train = _slow_train

        await _run(_make_args(num_rollout=2, save=str(tmp_path), save_interval=1), trainers=trainers)

        state = MultiPolicyCheckpointState.load(tmp_path, leader_rollout_id=1)
        assert state.leader_model_id == "a"
        assert state.rollout_ids["a"] == 1
        assert state.rollout_ids["b"] == [call.args[0] for call in trainers["b"].save_model.await_args_list][-1]

    async def test_a_parked_follower_is_saved_at_the_round_it_reached(self):
        """A record naming a policy at a rollout it never checkpointed cannot be resumed."""
        trainers = {"a": AsyncMock(), "b": AsyncMock()}

        await _run(_make_args(num_rollout=1, save=None, save_interval=1), trainers=trainers)

        [saved_at] = [call.args[0] for call in trainers["b"].save_model.await_args_list]
        assert saved_at == trainers["b"].train.await_args_list[-1].args[0]

    async def test_every_policy_is_on_disk_before_the_record_claims_the_checkpoint_exists(self):
        """An asynchronous follower save still running would leave the record pointing at files nobody wrote."""
        trainers = {"a": AsyncMock(), "b": AsyncMock()}

        await _run(_make_args(num_rollout=1, save=None, save_interval=1), trainers=trainers)

        for trainer in trainers.values():
            assert [call.kwargs["force_sync"] for call in trainer.save_model.await_args_list] == [True]

    async def test_only_the_leader_snapshots_the_rollout_executor(self):
        """One buffer serves every policy, so a second policy saving it would snapshot it twice per round."""
        kwargs = await _run(_make_args(num_rollout=2, save=None, save_interval=1))

        assert [call.args[0] for call in kwargs["rollout_executor"].save.await_args_list] == [0, 1]

    async def test_every_checkpoint_of_the_leader_is_saved_synchronously(self):
        """An asynchronous save would let the run publish a record of files that are still being written."""
        trainers = {"a": AsyncMock(), "b": AsyncMock()}

        await _run(_make_args(num_rollout=2, save=None, save_interval=1), trainers=trainers)

        assert [call.kwargs["force_sync"] for call in trainers["a"].save_model.await_args_list] == [True, True]

    async def test_two_checkpoints_in_a_row_both_reach_every_policy(self):
        """The second round must wait for the first one to disperse instead of reading a stale arrival."""
        trainers = {"a": AsyncMock(), "b": AsyncMock()}

        await _run(_make_args(num_rollout=2, save=None, save_interval=1), trainers=trainers)

        assert [call.args[0] for call in trainers["a"].save_model.await_args_list] == [0, 1]
        rounds_of_b = [call.args[0] for call in trainers["b"].save_model.await_args_list]
        assert len(rounds_of_b) == 2 and rounds_of_b[0] <= rounds_of_b[1]

    async def test_a_run_without_a_save_directory_records_nothing(self, tmp_path):
        """--save is what asks for checkpoints on disk; the record must not invent a directory."""
        await _run(_make_args(num_rollout=1, save=None, save_interval=1))

        assert list(tmp_path.iterdir()) == []

    async def test_the_external_save_sentinel_is_read_and_removed_by_the_leader_alone(self, tmp_path):
        """Every policy racing to remove one sentinel would leave the losers deleting a missing file."""
        sentinel = tmp_path / "save-now"
        sentinel.write_text("")
        trainers = {"a": AsyncMock(), "b": AsyncMock()}

        await _run(
            _make_args(num_rollout=1, save=None, save_trigger_sentinel=str(sentinel)),
            trainers=trainers,
        )

        assert not sentinel.exists()
        assert [call.args[0] for call in trainers["a"].save_model.await_args_list] == [0]

    async def test_a_sentinel_triggered_save_is_synchronous_for_the_leader(self, tmp_path):
        """Whoever wrote the sentinel is waiting for the files, so the leader may not defer its own write."""
        sentinel = tmp_path / "save-now"
        sentinel.write_text("")
        trainers = {"a": AsyncMock(), "b": AsyncMock()}

        await _run(
            _make_args(num_rollout=2, save=None, save_trigger_sentinel=str(sentinel)),
            trainers=trainers,
        )

        assert [call.kwargs["force_sync"] for call in trainers["a"].save_model.await_args_list] == [True]
        assert [call.kwargs["force_sync"] for call in trainers["b"].save_model.await_args_list] == [True]

    async def test_a_run_without_a_save_rhythm_snapshots_nothing(self):
        """Parking every policy on a round that saves nothing would cost the run its whole overlap."""
        trainers = {"a": AsyncMock(), "b": AsyncMock()}

        context = await _run(_make_args(num_rollout=2, save=None, save_interval=None), trainers=trainers)

        trainers["a"].save_model.assert_not_awaited()
        trainers["b"].save_model.assert_not_awaited()
        context["rollout_executor"].save.assert_not_awaited()
