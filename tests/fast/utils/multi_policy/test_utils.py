from argparse import Namespace
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from tests.fast.fixtures.args_fixtures import parser_defaults
from tests.fast.fixtures.megatron_config_fixtures import encode_megatron_config

from miles.backends.megatron_utils.megatron_config import MegatronConfig, resolve_megatron_config
from miles.utils.multi_policy import utils as multi_policy_utils
from miles.utils.multi_policy.checkpoint_state import MultiPolicyCheckpointState
from miles.utils.multi_policy.utils import TrainerInfo


def _make_config(*model_ids: str) -> MegatronConfig:
    return resolve_megatron_config(Namespace(megatron_config=encode_megatron_config(*model_ids), use_critic=False))


def _make_args(*model_ids: str, **overrides: Any) -> Namespace:
    defaults = dict(
        megatron_config=encode_megatron_config(*model_ids),
        use_critic=False,
        colocate=False,
        fully_async=True,
        eval_interval=None,
        save=None,
        save_interval=None,
        sglang_config="sglang.yaml",
        debug_rollout_only=False,
        async_unused_samples_handler="drop",
        dump_details=None,
        save_debug_rollout_data=None,
        load_debug_rollout_data=None,
        ci_inject_rollout_data_path=None,
    )
    defaults.update(overrides)
    return Namespace(**{**parser_defaults(), **defaults})


def _stub_sglang_models(monkeypatch, *names_updatable: tuple[str, bool]) -> None:
    """Answer resolve_sglang_config with just the model names and their update_weights flag."""
    models = [SimpleNamespace(name=name, update_weights=updatable) for name, updatable in names_updatable]
    monkeypatch.setattr(multi_policy_utils, "resolve_sglang_config", lambda args: SimpleNamespace(models=models))


class TestValidateMultiPolicyArgs:
    def _validate(self, args: Namespace) -> None:
        multi_policy_utils.validate_multi_policy_args(args, megatron_config=resolve_megatron_config(args))

    def test_a_run_naming_every_policy_on_both_sides_is_accepted(self, monkeypatch):
        """The happy path has to stay reachable, or every refusal below is vacuous."""
        _stub_sglang_models(monkeypatch, ("a", True), ("b", True))

        self._validate(_make_args("a", "b"))

    def test_a_policy_without_a_matching_sglang_model_is_refused(self, monkeypatch):
        """This is the only startup check that stops a weight update from finding no engine at all."""
        _stub_sglang_models(monkeypatch, ("a", True))

        with pytest.raises(AssertionError, match=r"--megatron-config models \['b'\] have no matching"):
            self._validate(_make_args("a", "b"))

    def test_an_sglang_model_that_is_frozen_does_not_count_as_a_match(self, monkeypatch):
        """A frozen engine is never handed weights, so the policy pointing at it would train into nothing."""
        _stub_sglang_models(monkeypatch, ("a", True), ("b", False))

        with pytest.raises(AssertionError, match=r"\['b'\] have no matching"):
            self._validate(_make_args("a", "b"))

    def test_a_single_policy_config_is_sent_back_to_train_async(self, monkeypatch):
        """One policy through this driver would silently skip everything train_async.py does."""
        _stub_sglang_models(monkeypatch, ("a", True))

        with pytest.raises(AssertionError, match="run train_async.py instead"):
            self._validate(_make_args("a"))

    @pytest.mark.parametrize(
        "overrides, message",
        [
            (dict(colocate=True), "does not support --colocate"),
            (dict(fully_async=False), "only supported for --fully-async"),
            (dict(use_critic=True), "does not support --use-critic"),
            (dict(debug_rollout_only=True), "does not support --debug-rollout-only"),
            (dict(async_unused_samples_handler="retry"), "retry"),
            (dict(dump_details="/tmp/dump"), "--dump-details"),
            (dict(load_debug_rollout_data="/tmp/{rollout_id}.pt"), "--load-debug-rollout-data"),
            (dict(ci_inject_rollout_data_path="/tmp/{rollout_id}.pt"), "--ci-inject-rollout-data-path"),
        ],
    )
    def test_the_unsupported_run_shapes_are_refused(self, monkeypatch, overrides, message):
        """Each of these silently breaks a different assumption the one-loop-per-policy driver makes."""
        _stub_sglang_models(monkeypatch, ("a", True), ("b", True))

        with pytest.raises(AssertionError, match=message):
            self._validate(_make_args("a", "b", **overrides))

    def test_an_evaluating_run_is_refused(self, monkeypatch):
        """There is no eval dispatcher here, so --eval-interval would be accepted and never honored."""
        _stub_sglang_models(monkeypatch, ("a", True), ("b", True))

        with pytest.raises(AssertionError, match="does not evaluate"):
            self._validate(_make_args("a", "b", eval_interval=10))

    def test_a_run_without_an_sglang_config_is_refused(self, monkeypatch):
        """Without one inference model per policy, every update would land on the same engines."""
        _stub_sglang_models(monkeypatch, ("a", True), ("b", True))

        with pytest.raises(AssertionError, match="needs --sglang-config"):
            self._validate(_make_args("a", "b", sglang_config=None))

    def test_a_checkpointing_run_is_accepted(self, monkeypatch):
        """Saving was refused until the global checkpoint existed, and nothing else must reintroduce the refusal."""
        _stub_sglang_models(monkeypatch, ("a", True), ("b", True))

        self._validate(_make_args("a", "b", save="/ckpt", save_interval=10))


def _make_trainer_args(*model_ids: str, **overrides: Any) -> Namespace:
    defaults = dict(
        megatron_config=encode_megatron_config(*model_ids),
        use_critic=False,
        rollout_global_dataset=True,
        trainer_model_id=None,
        tokenizer_model="/ckpt/hf",
        hf_checkpoint="/ckpt/hf",
        save=None,
        load=None,
        ref_load=None,
        ref_ckpt_step=None,
        megatron_to_hf_mode="core",
        no_load_optim=False,
        no_load_rng=False,
        finetune=False,
        start_rollout_id=None,
        trainer_controller_addrs=None,
    )
    defaults.update(overrides)
    return Namespace(**{**parser_defaults(), **defaults})


class TestCreatePolicyTrainers:
    @staticmethod
    def _stub_create_training_model(monkeypatch, start_rollout_ids: dict[str, int]) -> list[dict]:
        created: list[dict] = []

        async def _create(trainer_args, *, trainer_id):
            handle = AsyncMock()
            handle.get_train_parallel_config = AsyncMock(return_value=f"parallel-config-of-{trainer_id}")
            created.append(dict(trainer_id=trainer_id, args=trainer_args, handle=handle))
            return SimpleNamespace(handle=handle, start_rollout_id=start_rollout_ids[trainer_args.trainer_model_id])

        monkeypatch.setattr(multi_policy_utils, "create_training_model", _create)
        return created

    async def test_every_policy_gets_a_trainer_keyed_by_its_model_id(self, monkeypatch):
        """The driver looks a trainer up by model id on every round; a wrong key trains the wrong policy."""
        created = self._stub_create_training_model(monkeypatch, dict(a=0, b=0))

        trainers = await multi_policy_utils.create_trainers(_make_trainer_args("a", "b"), rollout_executor=AsyncMock())

        assert [entry["trainer_id"] for entry in created] == ["a-actor", "b-actor"]
        assert [entry["args"].trainer_model_id for entry in created] == ["a", "b"]
        assert list(trainers) == ["a", "b"]
        assert [trainer.model_id for trainer in trainers.values()] == ["a", "b"]
        assert [trainer.handle for trainer in trainers.values()] == [entry["handle"] for entry in created]

    async def test_every_trainer_carries_the_rollout_its_own_checkpoint_restored(self, monkeypatch):
        """Each policy resumes from its own position, so the position travels beside its handle."""
        self._stub_create_training_model(monkeypatch, dict(a=4, b=2))

        trainers = await multi_policy_utils.create_trainers(_make_trainer_args("a", "b"), rollout_executor=AsyncMock())

        assert {model_id: trainer.start_rollout_id for model_id, trainer in trainers.items()} == dict(a=4, b=2)

    async def test_every_policy_publishes_its_own_parallel_config(self, monkeypatch):
        """The executor splits each policy's batch by that policy's own dp size."""
        self._stub_create_training_model(monkeypatch, dict(a=0, b=0))
        rollout_executor = AsyncMock()

        await multi_policy_utils.create_trainers(_make_trainer_args("a", "b"), rollout_executor=rollout_executor)

        assert [
            (call.args[0], call.kwargs["trainer_model_id"])
            for call in rollout_executor.set_train_parallel_config.await_args_list
        ] == [("parallel-config-of-a-actor", "a"), ("parallel-config-of-b-actor", "b")]

    async def test_the_executor_is_loaded_at_the_leader_policys_position(self, monkeypatch):
        """One executor serves every policy, and the leader is the policy whose index names the run."""
        self._stub_create_training_model(monkeypatch, dict(a=4, b=2))
        rollout_executor = AsyncMock()

        await multi_policy_utils.create_trainers(_make_trainer_args("a", "b"), rollout_executor=rollout_executor)

        rollout_executor.load.assert_awaited_once_with(3)

    async def test_a_lagging_leader_still_decides_where_the_executor_loads(self, monkeypatch):
        """Picking any other trainer would replay the global rollout data from a position the leader never stood at."""
        self._stub_create_training_model(monkeypatch, dict(a=2, b=9))
        rollout_executor = AsyncMock()

        await multi_policy_utils.create_trainers(_make_trainer_args("a", "b"), rollout_executor=rollout_executor)

        rollout_executor.load.assert_awaited_once_with(1)

    async def test_a_resume_without_the_global_rollout_state_is_refused(self, monkeypatch, tmp_path):
        """The models would resume at rollout 4 while the data source silently restarts at the first prompt."""
        self._stub_create_training_model(monkeypatch, dict(a=4, b=4))

        with pytest.raises(AssertionError, match="global_dataset_state_dict_3.pt is missing"):
            await multi_policy_utils.create_trainers(
                _make_trainer_args("a", "b", load=str(tmp_path)), rollout_executor=AsyncMock()
            )

    async def test_a_resume_with_the_global_rollout_state_loads_it(self, monkeypatch, tmp_path):
        """The supported resume shape has to stay reachable, or no multi policy run could ever restart."""
        self._stub_create_training_model(monkeypatch, dict(a=4, b=4))
        state = tmp_path / "rollout" / "global_dataset_state_dict_3.pt"
        state.parent.mkdir(parents=True)
        state.write_bytes(b"")
        rollout_executor = AsyncMock()

        await multi_policy_utils.create_trainers(
            _make_trainer_args("a", "b", load=str(tmp_path)), rollout_executor=rollout_executor
        )

        rollout_executor.load.assert_awaited_once_with(3)

    async def test_a_trainer_config_without_a_policy_model_id_is_refused(self, monkeypatch):
        """Every state this driver keys is keyed by model id, so an unnamed trainer has nowhere to live."""
        self._stub_create_training_model(monkeypatch, {})

        with pytest.raises(AssertionError, match="carries no policy model id"):
            await multi_policy_utils.create_trainers(
                _make_trainer_args(megatron_config=None), rollout_executor=AsyncMock()
            )


class TestDefinePolicyMetricGroups:
    def test_a_multi_policy_run_binds_every_prefixed_curve_to_its_own_step(self, monkeypatch):
        """An undeclared prefix is plotted against wandb's internal counter instead of its own rollout step."""
        calls: list[dict] = []
        monkeypatch.setattr(multi_policy_utils, "define_step_key_metric_group", lambda **kwargs: calls.append(kwargs))

        multi_policy_utils.define_policy_metric_groups(_make_config("a", "b"))

        assert calls == [
            dict(prefix="a", step_key="a/rollout/step"),
            dict(prefix="a/train", step_key="a/train/step"),
            dict(prefix="b", step_key="b/rollout/step"),
            dict(prefix="b/train", step_key="b/train/step"),
        ]


class TestAssertConsistentRestore:
    @staticmethod
    def _trainers(**start_rollout_ids: int) -> dict[str, TrainerInfo]:
        return {
            model_id: TrainerInfo(model_id=model_id, start_rollout_id=value, handle=AsyncMock())
            for model_id, value in start_rollout_ids.items()
        }

    def _assert(self, args, **start_rollout_ids: int) -> None:
        multi_policy_utils.assert_consistent_restore(
            args, trainers=self._trainers(**start_rollout_ids), leader_model_id="a"
        )

    def test_a_fresh_run_needs_no_record(self, tmp_path):
        """Nothing was ever saved, so there is nothing to be consistent with."""
        self._assert(Namespace(save=str(tmp_path), load=None), a=0, b=0)

    def test_the_policies_resume_at_the_positions_the_record_names(self, tmp_path):
        """Fully async policies run at their own pace, so a lagging policy is a legal resume."""
        MultiPolicyCheckpointState(leader_model_id="a", rollout_ids={"a": 4, "b": 2}).save(tmp_path)

        self._assert(Namespace(save=str(tmp_path), load=None), a=5, b=3)

    def test_the_record_is_read_from_the_load_directory(self, tmp_path):
        """Resuming with --load elsewhere and a fresh --save is the common shape and used to skip the check."""
        load_dir = tmp_path / "old"
        MultiPolicyCheckpointState(leader_model_id="a", rollout_ids={"a": 4, "b": 2}).save(load_dir)
        args = Namespace(save=str(tmp_path / "new"), load=str(load_dir))

        with pytest.raises(AssertionError, match="must resume exactly where"):
            self._assert(args, a=5, b=5)

    def test_a_policy_restored_past_its_recorded_position_is_refused(self, tmp_path):
        """The global rollout data was snapshotted at the recorded moment, not at this one."""
        MultiPolicyCheckpointState(leader_model_id="a", rollout_ids={"a": 4, "b": 2}).save(tmp_path)

        with pytest.raises(AssertionError, match="must resume exactly where"):
            self._assert(Namespace(save=str(tmp_path), load=None), a=5, b=4)

    def test_a_resume_without_a_record_fails_loudly(self, tmp_path):
        """Silently skipping the check is exactly the mixture of checkpoints the record exists to refuse."""
        with pytest.raises(AssertionError, match="no record of"):
            self._assert(Namespace(save=str(tmp_path), load=None), a=5, b=3)

    def test_a_record_written_under_another_leader_is_refused(self, tmp_path):
        """The global rollout index means whatever the leader's index meant when it was written."""
        MultiPolicyCheckpointState(leader_model_id="b", rollout_ids={"a": 4, "b": 4}).save(tmp_path)

        with pytest.raises(AssertionError, match="as the leader policy"):
            self._assert(Namespace(save=str(tmp_path), load=None), a=5, b=5)

    def test_a_policy_restored_while_the_leader_starts_from_scratch_is_refused(self, tmp_path):
        """The data source would be replayed from zero into a policy that already trained on it."""
        with pytest.raises(AssertionError, match="starts from scratch"):
            self._assert(Namespace(save=str(tmp_path), load=None), a=0, b=3)

    def test_a_resume_without_any_state_directory_is_refused(self):
        """A run resuming with neither --load nor --save has nowhere to read the other policies from."""
        with pytest.raises(AssertionError, match="without --load or --save"):
            self._assert(Namespace(save=None, load=None), a=5, b=3)

    def test_a_leader_that_is_not_one_of_the_policies_is_reported(self, tmp_path):
        """Every position this check compares hangs off the leader, so a leader nobody trains is unanswerable."""
        with pytest.raises(KeyError, match="'c'"):
            multi_policy_utils.assert_consistent_restore(
                Namespace(save=str(tmp_path), load=None), trainers=self._trainers(a=5, b=3), leader_model_id="c"
            )
