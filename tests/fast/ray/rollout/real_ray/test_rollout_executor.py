from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import ray
from tests.fast.ray.rollout.conftest import make_args, make_samples_grouped

from miles.ray.rollout.debug_data import save_debug_rollout_data
from miles.ray.rollout.rollout_executor import RolloutExecutor
from miles.rollout.base_types import (
    BaseRolloutFn,
    RolloutFnEvalInput,
    RolloutFnEvalOutput,
    RolloutFnTrainInput,
    RolloutFnTrainOutput,
)
from miles.rollout.checkpoint_eval import CheckpointEvalFn
from miles.utils.types import WeightVersionSpan, WeightVersionsPerCall
from miles.utils.weight_version import max_rollouts_without_published_weight_version


@pytest.fixture
def http_client_calls(monkeypatch) -> list[str]:
    import miles.ray.rollout.rollout_executor as rexec

    recorded: list[str] = []
    monkeypatch.setattr(rexec, "init_http_client", lambda args: recorded.append("init_http_client"))
    return recorded


@pytest.fixture
def patch_low_level(monkeypatch, http_client_calls):
    import miles.ray.rollout.rollout_executor as rexec

    monkeypatch.setattr(rexec, "configure_logger", lambda *a, **kw: None)
    monkeypatch.setattr(rexec, "init_tracking", lambda *a, **kw: None)
    monkeypatch.setattr(rexec, "load_function", lambda path: lambda *a, **kw: None)
    monkeypatch.setattr(rexec, "load_rollout_function", lambda input, path: lambda *a, **kw: None)
    monkeypatch.setattr(rexec, "log_rollout_data", lambda *a, **kw: None)
    monkeypatch.setattr(rexec, "log_eval_rollout_data", lambda *a, **kw: None)
    monkeypatch.setattr(rexec, "save_debug_rollout_data", lambda *a, **kw: None)


def _make_executor(args):
    return RolloutExecutor.__ray_actor_class__(args=args)


def _make_test_args(**overrides):
    return make_args(
        sglang_router_ip="127.0.0.1",
        sglang_router_port=30000,
        use_wandb=False,
        use_tensorboard=False,
        use_mlflow=False,
        use_distributed_post=False,
        **overrides,
    )


@pytest.mark.asyncio
class TestProcessSetup:
    async def test_initializes_the_http_client(self, ray_local_mode, patch_low_level, http_client_calls):
        """The rollout functions issue their HTTP from this actor, so the client is created here."""
        _make_executor(_make_test_args())

        assert http_client_calls == ["init_http_client"]

    async def test_skips_the_http_client_in_debug_train_only(self, ray_local_mode, patch_low_level, http_client_calls):
        """No engines exist in this mode, so there is nothing to talk to."""
        args = _make_test_args()
        args.debug_train_only = True

        _make_executor(args)

        assert http_client_calls == []


@pytest.mark.asyncio
class TestRolloutFunctionConstruction:
    async def test_debug_rollout_replay_skips_class_based_rollout_construction(
        self,
        ray_local_mode,
        tmp_path,
        patch_low_level,
        monkeypatch,
    ):
        """Replaying dumped rollout data must not build the rollout functions."""
        import miles.ray.rollout.rollout_executor as rexec

        args = _make_test_args()
        args.debug_train_only = True
        args.load_debug_rollout_data = str(tmp_path / "rollout-{rollout_id}.pt")
        args.rollout_num_gpus = None
        monkeypatch.delenv("MILES_USE_LEGACY_ROLLOUT_V1", raising=False)

        def fail_if_loaded(*args, **kwargs):
            pytest.fail("debug rollout replay must not construct rollout functions")

        monkeypatch.setattr(rexec, "load_rollout_function", fail_if_loaded)

        executor = _make_executor(args)

        assert executor.generate_rollout is None
        assert executor.eval_generate_rollout is None

    async def test_debug_train_only_without_replay_constructs_rollout_function(
        self,
        ray_local_mode,
        tmp_path,
        patch_low_level,
        monkeypatch,
    ):
        """Without replay data, debug_train_only still builds both rollout functions."""
        import miles.ray.rollout.rollout_executor as rexec

        args = _make_test_args()
        args.debug_train_only = True
        monkeypatch.delenv("MILES_USE_LEGACY_ROLLOUT_V1", raising=False)
        loaded_paths: list[str] = []

        def record_load(input, path):
            loaded_paths.append(path)
            return lambda *args, **kwargs: None

        monkeypatch.setattr(rexec, "load_rollout_function", record_load)

        executor = _make_executor(args)

        assert loaded_paths == [args.rollout_function_path, args.eval_function_path]
        assert executor.generate_rollout is not None
        assert executor.eval_generate_rollout is not None


@pytest.mark.asyncio
class TestGenerate:
    async def test_invokes_rollout_fn_with_correct_input_and_returns_dp_split(self, ray_local_mode, patch_low_level):
        """generate passes a train input and returns the samples split per dp rank."""
        args = _make_test_args()
        args.global_batch_size = 8

        executor = _make_executor(args)
        executor.set_train_parallel_config({"dp_size": 2})

        captured: list = []

        def fake_rollout_fn(input):
            captured.append(input)
            return RolloutFnTrainOutput(
                samples=[make_samples_grouped(n_groups=2, group_size=4)],
                metrics={"my_metric": 1.23},
            )

        executor.generate_rollout = fake_rollout_fn

        result = await executor.get(rollout_id=42)

        assert len(captured) == 1
        assert isinstance(captured[0], RolloutFnTrainInput)
        assert captured[0].rollout_id == 42
        assert set(result) == {"sample_indices", "data_ref"}
        data_refs = result["data_ref"]
        assert len(data_refs) == 2
        partitions = ray.get([box.inner for box in data_refs])
        for partition in partitions:
            assert "tokens" in partition
            assert "rewards" in partition
            assert "loss_masks" in partition
            assert len(partition["tokens"]) == 4

    async def test_rejects_samples_generated_under_the_default_weight_version(self, ray_local_mode, patch_low_level):
        """A batch carrying the sglang never-updated version must fail get(), not reach training."""
        args = _make_test_args()
        args.global_batch_size = 8

        executor = _make_executor(args)
        executor.set_train_parallel_config({"dp_size": 2})

        samples = make_samples_grouped(n_groups=2, group_size=4)
        samples[0].weight_versions = [
            WeightVersionsPerCall(spans=[WeightVersionSpan(version="default", abs_start=0, abs_end=1)])
        ]

        executor.generate_rollout = lambda input: RolloutFnTrainOutput(samples=[samples], metrics=None)

        with pytest.raises(AssertionError, match="never updated"):
            await executor.get(rollout_id=42)

    async def test_a_frozen_weight_version_eventually_fails_the_run(self, ray_local_mode, patch_low_level):
        """A driver that never forwards the version must fail loudly, not silently disable staleness filtering."""
        args = _make_test_args()
        args.global_batch_size = 8

        executor = _make_executor(args)
        executor.set_train_parallel_config({"dp_size": 2})
        executor.generate_rollout = lambda input: RolloutFnTrainOutput(
            samples=[make_samples_grouped(n_groups=2, group_size=4)], metrics=None
        )

        with pytest.raises(AssertionError, match="set_weight_version"):
            for rollout_id in range(max_rollouts_without_published_weight_version(args) + 1):
                await executor.get(rollout_id=rollout_id)

    async def test_publishing_a_version_every_step_keeps_the_run_alive(self, ray_local_mode, patch_low_level):
        """The supported driver publishes after each update, which must never trip the staleness assert."""
        args = _make_test_args()
        args.global_batch_size = 8

        executor = _make_executor(args)
        executor.set_train_parallel_config({"dp_size": 2})
        executor.generate_rollout = lambda input: RolloutFnTrainOutput(
            samples=[make_samples_grouped(n_groups=2, group_size=4)], metrics=None
        )

        for rollout_id in range(3 * max_rollouts_without_published_weight_version(args)):
            executor.set_weight_version(rollout_id)
            await executor.get(rollout_id=rollout_id)

    async def test_does_not_touch_the_inference_side(self, ray_local_mode, patch_low_level):
        """The controller is a driver-side object the executor cannot reach, so generate must not need it."""
        args = _make_test_args()
        args.global_batch_size = 4

        executor = _make_executor(args)
        executor.set_train_parallel_config({"dp_size": 1})
        executor.generate_rollout = lambda input: RolloutFnTrainOutput(
            samples=[make_samples_grouped(n_groups=1, group_size=4)], metrics={}
        )

        await executor.get(rollout_id=7)

        assert not hasattr(executor, "servers")
        assert not hasattr(executor, "_health_monitors")


class _RecordingRolloutFn(BaseRolloutFn):
    def __init__(self, name: str, log: list[tuple[str, str, object]]) -> None:
        self._name = name
        self._log = log

    def __call__(self, input):
        raise AssertionError("not exercised by the checkpointing tests")

    def save(self, rollout_id: int) -> None:
        self._log.append((self._name, "save", rollout_id))

    def load(self, rollout_id: int | None) -> None:
        self._log.append((self._name, "load", rollout_id))


class _RecordingEventLoggerCheckpoint:
    def __init__(self) -> None:
        self.restored: list = []
        self.snapshots: list[tuple] = []

    def restore(self, args) -> None:
        self.restored.append(args)

    def snapshot(self, args, rollout_id) -> None:
        self.snapshots.append((args, rollout_id))


@pytest.mark.asyncio
class TestCheckpointing:
    async def test_save_and_load_reach_every_distinct_rollout_function(
        self,
        ray_local_mode,
        patch_low_level,
        monkeypatch,
    ):
        """Both the train and the eval instance carry state, so each is written and read back in turn."""
        import miles.ray.rollout.rollout_executor as rexec

        monkeypatch.setattr(rexec, "event_logger_checkpoint", MagicMock())
        args = _make_test_args(rollout_global_dataset=False)

        executor = _make_executor(args)
        executor.use_legacy_rollout_v1 = False
        calls: list[tuple[str, str, object]] = []
        executor.generate_rollout = _RecordingRolloutFn("train", calls)
        executor.eval_generate_rollout = _RecordingRolloutFn("eval", calls)
        executor.data_source = MagicMock()

        executor.save(rollout_id=7)
        executor.load(rollout_id=7)

        assert calls == [
            ("train", "save", 7),
            ("eval", "save", 7),
            ("train", "load", 7),
            ("eval", "load", 7),
        ]

    async def test_save_forwards_to_the_data_source_for_a_global_dataset(
        self,
        ray_local_mode,
        patch_low_level,
        monkeypatch,
    ):
        """With a global dataset both the data source and the rollout functions are checkpointed."""
        import miles.ray.rollout.rollout_executor as rexec

        monkeypatch.setattr(rexec, "event_logger_checkpoint", MagicMock())
        args = _make_test_args(rollout_global_dataset=True)

        executor = _make_executor(args)
        executor.use_legacy_rollout_v1 = False
        calls: list[tuple[str, str, object]] = []
        executor.generate_rollout = _RecordingRolloutFn("train", calls)
        executor.eval_generate_rollout = _RecordingRolloutFn("eval", calls)
        executor.data_source = MagicMock()

        executor.save(rollout_id=5)

        executor.data_source.save.assert_called_once_with(5)
        assert ("train", "save", 5) in calls

    async def test_save_forwards_to_the_data_source_without_a_global_dataset(
        self,
        ray_local_mode,
        patch_low_level,
        monkeypatch,
    ):
        """A custom data source is saved as unconditionally as it is loaded, so its state can be restored."""
        import miles.ray.rollout.rollout_executor as rexec

        monkeypatch.setattr(rexec, "event_logger_checkpoint", MagicMock())
        args = _make_test_args(rollout_global_dataset=False)

        executor = _make_executor(args)
        executor.use_legacy_rollout_v1 = False
        calls: list[tuple[str, str, object]] = []
        executor.generate_rollout = _RecordingRolloutFn("train", calls)
        executor.eval_generate_rollout = _RecordingRolloutFn("eval", calls)
        executor.data_source = MagicMock()

        executor.save(rollout_id=3)
        executor.load(rollout_id=3)

        executor.data_source.save.assert_called_once_with(3)
        executor.data_source.load.assert_called_once_with(3)
        assert ("train", "save", 3) in calls

    async def test_legacy_function_path_does_not_get_save_load(
        self,
        ray_local_mode,
        patch_low_level,
        monkeypatch,
    ):
        """Without the experimental flag the rollout functions are bare callables, so they are not checkpointed."""
        import miles.ray.rollout.rollout_executor as rexec

        monkeypatch.setattr(rexec, "event_logger_checkpoint", MagicMock())
        args = _make_test_args(rollout_global_dataset=False)

        executor = _make_executor(args)
        executor.use_legacy_rollout_v1 = True
        executor.generate_rollout = lambda *a, **kw: None
        executor.eval_generate_rollout = lambda *a, **kw: None
        executor.data_source = MagicMock()

        executor.save(rollout_id=1)
        executor.load(rollout_id=1)

        executor.data_source.load.assert_called_once_with(1)

    async def test_save_snapshots_the_event_log_for_the_rollout_id(
        self,
        ray_local_mode,
        patch_low_level,
        monkeypatch,
    ):
        """The audit event log is checkpointed alongside the rollout state it explains."""
        import miles.ray.rollout.rollout_executor as rexec

        recorder = _RecordingEventLoggerCheckpoint()
        monkeypatch.setattr(rexec, "event_logger_checkpoint", recorder)
        args = _make_test_args(rollout_global_dataset=False)

        executor = _make_executor(args)
        executor.use_legacy_rollout_v1 = False
        executor.generate_rollout = _RecordingRolloutFn("train", [])
        executor.eval_generate_rollout = None
        executor.data_source = MagicMock()

        executor.save(rollout_id=9)

        assert recorder.snapshots == [(args, 9)]


@pytest.mark.asyncio
class TestEval:
    async def test_invokes_eval_fn_with_eval_input(self, ray_local_mode, patch_low_level):
        """eval passes an eval input carrying the rollout id."""
        executor = _make_executor(_make_test_args())

        captured: list = []

        def fake_eval_fn(input):
            captured.append(input)
            return RolloutFnEvalOutput(data={"my_dataset": {"rewards": [0.5, 1.0]}}, metrics={})

        executor.eval_generate_rollout = fake_eval_fn

        await executor.eval(rollout_id=10)

        assert len(captured) == 1
        assert isinstance(captured[0], RolloutFnEvalInput)
        assert captured[0].rollout_id == 10

    async def test_skipped_in_debug_train_only_mode(self, ray_local_mode, patch_low_level):
        """debug_train_only short-circuits eval before the rollout function runs."""
        args = _make_test_args()
        args.debug_train_only = True

        executor = _make_executor(args)

        called: list = []
        executor.eval_generate_rollout = lambda inp: called.append(inp)

        await executor.eval(rollout_id=10)

        assert called == []


class _NamedRolloutFn(BaseRolloutFn):
    def __init__(self, path: str) -> None:
        self.path = path

    def __call__(self, input):
        raise AssertionError("not exercised by the loading tests")


@pytest.mark.asyncio
class TestRolloutFunctionLoading:
    async def test_matching_train_and_eval_paths_share_one_rollout_instance(
        self, ray_local_mode, patch_low_level, monkeypatch
    ):
        """One configured path means one stateful object, so train and eval share its state."""
        import miles.ray.rollout.rollout_executor as rexec

        monkeypatch.setattr(rexec, "load_rollout_function", lambda input, path: _NamedRolloutFn(path))
        args = _make_test_args(rollout_function_path="pkg.same_fn", eval_function_path="pkg.same_fn")

        executor = _make_executor(args)

        assert executor.eval_generate_rollout is executor.generate_rollout

    async def test_distinct_train_and_eval_paths_get_their_own_instance(
        self, ray_local_mode, patch_low_level, monkeypatch
    ):
        """Two configured paths must each be loaded, so eval never silently runs the train function."""
        import miles.ray.rollout.rollout_executor as rexec

        monkeypatch.setattr(rexec, "load_rollout_function", lambda input, path: _NamedRolloutFn(path))
        args = _make_test_args(rollout_function_path="pkg.train_fn", eval_function_path="pkg.eval_fn")

        executor = _make_executor(args)

        assert executor.generate_rollout.path == "pkg.train_fn"
        assert executor.eval_generate_rollout.path == "pkg.eval_fn"


@pytest.mark.asyncio
class TestCustomHooks:
    async def test_get_converts_the_batch_with_the_configured_conversion_hook(
        self, ray_local_mode, patch_low_level, monkeypatch
    ):
        """The configured conversion hook receives the generated samples and its output is what ships."""
        import miles.ray.rollout.rollout_executor as rexec

        seen_samples: list = []

        def conversion_hook(args, samples):
            seen_samples.append(samples)
            return dict(tokens=[[1, 2], [3, 4]], sample_indices=[70, 71])

        monkeypatch.setattr(
            rexec,
            "load_function",
            lambda path: conversion_hook if path == "pkg.convert" else (lambda *a, **kw: None),
        )
        args = _make_test_args(global_batch_size=4, custom_convert_samples_to_train_data_path="pkg.convert")

        executor = _make_executor(args)
        executor.set_train_parallel_config({"dp_size": 2})
        executor.generate_rollout = lambda input: RolloutFnTrainOutput(
            samples=[make_samples_grouped(n_groups=1, group_size=4)], metrics={}
        )

        result = await executor.get(rollout_id=1)

        assert [sample.index for sample in seen_samples[0]] == [0, 1, 2, 3]
        assert result["sample_indices"] == [70, 71]
        partitions = ray.get([box.inner for box in result["data_ref"]])
        assert [partition["tokens"] for partition in partitions] == [[[1, 2]], [[3, 4]]]

    async def test_get_scores_the_batch_with_the_configured_reward_hook(
        self, ray_local_mode, patch_low_level, monkeypatch
    ):
        """The configured reward post-process hook replaces the built-in reward math."""
        import miles.ray.rollout.rollout_executor as rexec

        seen_samples: list = []

        def reward_hook(args, samples):
            seen_samples.append(samples)
            return [1.0] * len(samples), [9.0] * len(samples)

        monkeypatch.setattr(
            rexec,
            "load_function",
            lambda path: reward_hook if path == "pkg.reward" else (lambda *a, **kw: None),
        )
        args = _make_test_args(global_batch_size=4, custom_reward_post_process_path="pkg.reward")

        executor = _make_executor(args)
        executor.set_train_parallel_config({"dp_size": 2})
        executor.generate_rollout = lambda input: RolloutFnTrainOutput(
            samples=[make_samples_grouped(n_groups=1, group_size=4)], metrics={}
        )

        result = await executor.get(rollout_id=1)

        assert [sample.index for sample in seen_samples[0]] == [0, 1, 2, 3]
        partitions = ray.get([box.inner for box in result["data_ref"]])
        assert [partition["rewards"] for partition in partitions] == [[9.0, 9.0], [9.0, 9.0]]


@pytest.mark.asyncio
class TestWeightVersion:
    async def test_get_threads_the_latest_weight_version_into_the_train_input(self, ray_local_mode, patch_low_level):
        """The rollout function is told which engine weight version this batch is generated under."""
        args = _make_test_args(global_batch_size=4, indep_dp=False)

        executor = _make_executor(args)
        executor.set_train_parallel_config({"dp_size": 1})
        captured: list = []

        def fake_rollout_fn(input):
            captured.append(input)
            return RolloutFnTrainOutput(samples=[make_samples_grouped(n_groups=1, group_size=4)], metrics={})

        executor.generate_rollout = fake_rollout_fn
        executor.set_weight_version(3)
        executor.set_weight_version(7)

        await executor.get(rollout_id=1)

        assert captured[0].weight_version == 7

    async def test_a_decreasing_weight_version_is_rejected(self, ray_local_mode, patch_low_level):
        """Weight versions only move forward, so a lower one signals a broken update path."""
        executor = _make_executor(_make_test_args(indep_dp=False))
        executor.set_weight_version(7)

        with pytest.raises(AssertionError, match="went backwards"):
            executor.set_weight_version(3)

        assert executor.weight_version == 7

    async def test_independent_dp_accepts_a_decreasing_weight_version_with_a_warning(
        self, ray_local_mode, patch_low_level, caplog
    ):
        """Independent-DP fault tolerance may rewind a replica, so the rewind warns instead of failing."""
        executor = _make_executor(_make_test_args(indep_dp=True))
        executor.set_weight_version(7)

        with caplog.at_level(logging.WARNING):
            executor.set_weight_version(3)

        assert executor.weight_version == 3
        assert any("went backwards" in record.getMessage() for record in caplog.records)


@pytest.mark.asyncio
class TestDelayedDpSplit:
    async def test_get_stores_one_unsplit_batch(self, ray_local_mode, patch_low_level):
        """With the split delayed to the training side, one whole-batch reference is published."""
        args = _make_test_args(global_batch_size=4, delay_split_train_data_by_dp=True)

        executor = _make_executor(args)
        executor.set_train_parallel_config({"dp_size": 2})
        executor.generate_rollout = lambda input: RolloutFnTrainOutput(
            samples=[make_samples_grouped(n_groups=1, group_size=4)], metrics={}
        )

        result = await executor.get(rollout_id=1)

        assert not isinstance(result["data_ref"], list)
        stored = ray.get(result["data_ref"].inner)
        assert len(stored["tokens"]) == 4
        assert result["sample_indices"] == stored["sample_indices"] == [0, 1, 2, 3]


@pytest.mark.asyncio
class TestDebugRolloutData:
    async def test_recorded_data_is_replayed_instead_of_generated(self, ray_local_mode, patch_low_level, tmp_path):
        """Replaying a recording must not call the rollout function, and its metadata comes back verbatim."""
        template = str(tmp_path / "rollout_{rollout_id}.pt")
        save_debug_rollout_data(
            make_args(save_debug_rollout_data=template),
            make_samples_grouped(n_groups=1, group_size=4),
            rollout_id=3,
            evaluation=False,
            metadata={"dynamic_global_batch_size": 4},
        )
        args = _make_test_args(load_debug_rollout_data=template)

        executor = _make_executor(args)

        def unreachable(input):
            raise AssertionError("the rollout function must not run when a recording is replayed")

        executor.generate_rollout = unreachable

        data, metadata, metrics = await executor._get_rollout_data(rollout_id=3)

        assert [sample.index for sample in data] == [0, 1, 2, 3]
        assert metadata == {"dynamic_global_batch_size": 4}
        assert metrics is None

    async def test_injected_recording_replaces_the_verified_generated_batch(
        self, ray_local_mode, patch_low_level, tmp_path
    ):
        """CI injection swaps in the recording it verified against, and drops the generated metrics with it."""
        template = str(tmp_path / "inject_{rollout_id}.pt")
        injected = make_samples_grouped(n_groups=1, group_size=4)
        for sample in injected:
            sample.reward = 5.0
        save_debug_rollout_data(
            make_args(save_debug_rollout_data=template),
            injected,
            rollout_id=2,
            evaluation=False,
            metadata={"source": "recording"},
        )
        args = _make_test_args(
            global_batch_size=4,
            ci_inject_rollout_data_path=template,
            ci_inject_rollout_data_start_rollout_id=2,
        )

        executor = _make_executor(args)
        executor.set_train_parallel_config({"dp_size": 1})

        mismatched = make_samples_grouped(n_groups=1, group_size=4)
        for sample in mismatched:
            sample.tokens[-1] = 99
        executor.generate_rollout = lambda input: RolloutFnTrainOutput(
            samples=[mismatched], metrics={"generated": 1.0}
        )

        with pytest.raises(AssertionError, match="generated responses match"):
            await executor._get_rollout_data(rollout_id=2)

        executor.generate_rollout = lambda input: RolloutFnTrainOutput(
            samples=[make_samples_grouped(n_groups=1, group_size=4)], metrics={"generated": 1.0}
        )
        data, metadata, metrics = await executor._get_rollout_data(rollout_id=2)

        assert [sample.reward for sample in data] == [5.0, 5.0, 5.0, 5.0]
        assert metadata == {"source": "recording"}
        assert metrics is None


@pytest.mark.asyncio
class TestLegacyRolloutProtocol:
    async def test_train_generation_keeps_the_legacy_call_signature(self, ray_local_mode, patch_low_level):
        """Without the experimental flag the train fn is still called as (args, rollout_id, data_source)."""
        args = _make_test_args(global_batch_size=4)

        executor = _make_executor(args)
        executor.use_legacy_rollout_v1 = True
        executor.data_source = SimpleNamespace()
        executor.set_train_parallel_config({"dp_size": 1})
        calls: list[tuple] = []

        def legacy_rollout_fn(passed_args, rollout_id, data_source, evaluation):
            calls.append((passed_args, rollout_id, data_source, evaluation))
            return make_samples_grouped(n_groups=1, group_size=4)

        executor.generate_rollout = legacy_rollout_fn

        await executor.get(rollout_id=11)

        assert calls == [(args, 11, executor.data_source, False)]

    async def test_eval_keeps_the_legacy_call_signature_with_the_evaluation_flag(
        self, ray_local_mode, patch_low_level
    ):
        """Without the experimental flag eval uses the same legacy protocol, flagged as evaluation."""
        args = _make_test_args()

        executor = _make_executor(args)
        executor.use_legacy_rollout_v1 = True
        executor.data_source = SimpleNamespace()
        calls: list[tuple] = []

        def legacy_eval_fn(passed_args, rollout_id, data_source, evaluation):
            calls.append((passed_args, rollout_id, data_source, evaluation))
            return {"my_dataset": {"rewards": [1.0]}}

        executor.eval_generate_rollout = legacy_eval_fn

        await executor.eval(rollout_id=12)

        assert calls == [(args, 12, executor.data_source, True)]


class _RecordingMetricChecker:
    def __init__(self) -> None:
        self.evaluated: list[dict] = []
        self.disposed = False

    def on_eval(self, metrics: dict) -> None:
        self.evaluated.append(metrics)

    def dispose(self) -> None:
        self.disposed = True


class _RecordingCheckpointEvalFn(CheckpointEvalFn):
    def __init__(self) -> None:
        self.disposed = False

    async def evaluate_checkpoint(self, checkpoint_dir, input):
        raise AssertionError("not exercised by the lifecycle tests")

    def dispose(self) -> None:
        self.disposed = True


@pytest.mark.asyncio
class TestLifecycle:
    async def test_eval_reports_the_logged_metrics_to_the_metric_checker(
        self, ray_local_mode, patch_low_level, monkeypatch
    ):
        """The CI accuracy checker is fed exactly the eval metrics that were logged."""
        import miles.ray.rollout.rollout_executor as rexec

        monkeypatch.setattr(
            rexec, "log_eval_rollout_data", lambda rollout_id, args, data, metrics: {"eval/accuracy": 0.75}
        )

        executor = _make_executor(_make_test_args())
        checker = _RecordingMetricChecker()
        executor._metric_checker = checker
        executor.eval_generate_rollout = lambda input: RolloutFnEvalOutput(
            data={"my_dataset": {"rewards": [1.0]}}, metrics={}
        )

        await executor.eval(rollout_id=4)

        assert checker.evaluated == [{"eval/accuracy": 0.75}]

    async def test_dispose_releases_every_executor_owned_resource(self, ray_local_mode, patch_low_level, monkeypatch):
        """Teardown closes the data source, runs event analysis and disposes the checker and the eval fn."""
        import miles.ray.rollout.rollout_executor as rexec

        analyzed: list = []
        monkeypatch.setattr(rexec, "event_analyzer", SimpleNamespace(run_analysis_from_args=analyzed.append))
        args = _make_test_args()

        executor = _make_executor(args)
        closed: list = []
        executor.data_source = SimpleNamespace(close=lambda: closed.append("closed"))
        checker = _RecordingMetricChecker()
        executor._metric_checker = checker
        eval_fn = _RecordingCheckpointEvalFn()
        executor.eval_generate_rollout = eval_fn

        executor.dispose()

        assert closed == ["closed"]
        assert analyzed == [args]
        assert checker.disposed
        assert eval_fn.disposed


@pytest.mark.asyncio
class TestNumRolloutPerEpoch:
    async def test_counts_only_complete_global_batches(self, ray_local_mode, patch_low_level):
        """A trailing partial batch is not a rollout, so the epoch length floors the division."""
        executor = _make_executor(_make_test_args(rollout_global_dataset=True, rollout_batch_size=8))
        executor.data_source = SimpleNamespace(dataset=list(range(20)))

        assert executor.get_num_rollout_per_epoch() == 2

    async def test_rejects_a_non_global_data_source(self, ray_local_mode, patch_low_level):
        """Without a global dataset there is no epoch length to report."""
        executor = _make_executor(_make_test_args(rollout_global_dataset=False))
        executor.data_source = SimpleNamespace(dataset=list(range(20)))

        with pytest.raises(AssertionError):
            executor.get_num_rollout_per_epoch()


@pytest.mark.asyncio
class TestCheckpointWithoutARolloutFunction:
    async def test_checkpointing_a_replay_run_touches_only_the_data_source(self, ray_local_mode, patch_low_level):
        """--load-debug-rollout-data replays recorded samples and builds no rollout function, so there is no rollout state to write or read back."""
        executor = _make_executor(_make_test_args(load_debug_rollout_data="/nonexistent/rollout_{rollout_id}.pt"))
        executor.data_source = MagicMock()

        executor.save(3)
        executor.load(3)

        assert executor.generate_rollout is None
        executor.data_source.save.assert_called_once_with(3)
        executor.data_source.load.assert_called_once_with(3)

    async def test_a_replay_run_restores_at_startup_without_a_rollout_id(
        self,
        ray_local_mode,
        patch_low_level,
        monkeypatch,
    ):
        """Startup restores the executor before the first step, so the default rollout id must reach the data source."""
        monkeypatch.delenv("MILES_USE_LEGACY_ROLLOUT_V1", raising=False)
        executor = _make_executor(_make_test_args(load_debug_rollout_data="/nonexistent/rollout_{rollout_id}.pt"))
        executor.data_source = MagicMock()

        executor.load()

        executor.data_source.load.assert_called_once_with(None)

    async def test_saving_a_replay_run_still_snapshots_the_event_log(
        self,
        ray_local_mode,
        patch_low_level,
        monkeypatch,
    ):
        """Skipping the absent rollout function must not skip the rest of save, which still snapshots the audit log."""
        import miles.ray.rollout.rollout_executor as rexec

        monkeypatch.delenv("MILES_USE_LEGACY_ROLLOUT_V1", raising=False)
        recorder = _RecordingEventLoggerCheckpoint()
        monkeypatch.setattr(rexec, "event_logger_checkpoint", recorder)
        args = _make_test_args(load_debug_rollout_data="/nonexistent/rollout_{rollout_id}.pt")

        executor = _make_executor(args)
        executor.data_source = MagicMock()

        executor.save(6)

        executor.data_source.save.assert_called_once_with(6)
        assert recorder.snapshots == [(args, 6)]

    async def test_an_absent_train_rollout_function_is_not_replaced_by_the_eval_one(
        self,
        ray_local_mode,
        patch_low_level,
        monkeypatch,
    ):
        """With no train rollout function, checkpointing skips it rather than falling back to the eval instance."""
        import miles.ray.rollout.rollout_executor as rexec

        monkeypatch.setattr(rexec, "event_logger_checkpoint", MagicMock())

        executor = _make_executor(_make_test_args())
        executor.use_legacy_rollout_v1 = False
        executor.generate_rollout = None
        calls: list[tuple[str, str, object]] = []
        executor.eval_generate_rollout = _RecordingRolloutFn("eval", calls)
        executor.data_source = MagicMock()

        executor.save(rollout_id=2)
        executor.load(rollout_id=2)

        assert calls == [("eval", "save", 2), ("eval", "load", 2)]
        executor.data_source.save.assert_called_once_with(2)
        executor.data_source.load.assert_called_once_with(2)


class _AlwaysEqualRolloutFn(_RecordingRolloutFn):
    def __eq__(self, other: object) -> bool:
        return True

    def __hash__(self) -> int:
        return 0


@pytest.mark.asyncio
class TestCheckpointOfADistinctEvalRolloutFunction:
    async def test_a_separate_eval_instance_is_checkpointed_on_its_own(self, ray_local_mode, patch_low_level):
        """A distinct --eval-function-path instance owns its own state, which restore would otherwise silently drop."""
        executor = _make_executor(_make_test_args())
        executor.data_source = MagicMock()
        executor.generate_rollout = MagicMock()
        executor.eval_generate_rollout = MagicMock()

        executor.save(7)
        executor.load(7)

        executor.generate_rollout.save.assert_called_once_with(7)
        executor.generate_rollout.load.assert_called_once_with(7)
        executor.eval_generate_rollout.save.assert_called_once_with(7)
        executor.eval_generate_rollout.load.assert_called_once_with(7)

    async def test_a_shared_eval_instance_is_checkpointed_once(self, ray_local_mode, patch_low_level):
        """Train and eval reuse one instance when the paths match, so a second hook call would checkpoint it twice."""
        executor = _make_executor(_make_test_args())
        executor.data_source = MagicMock()
        shared = MagicMock()
        executor.generate_rollout = shared
        executor.eval_generate_rollout = shared

        executor.save(7)
        executor.load(7)

        shared.save.assert_called_once_with(7)
        shared.load.assert_called_once_with(7)

    async def test_an_eval_instance_that_compares_equal_is_still_checkpointed_separately(
        self, ray_local_mode, patch_low_level
    ):
        """Two objects that compare equal still hold two independent states, so equality must not stand in for identity."""
        executor = _make_executor(_make_test_args())
        executor.data_source = MagicMock()
        calls: list[tuple[str, str, object]] = []
        executor.generate_rollout = _AlwaysEqualRolloutFn("train", calls)
        executor.eval_generate_rollout = _AlwaysEqualRolloutFn("eval", calls)

        executor.save(4)
        executor.load(4)

        assert calls == [
            ("train", "save", 4),
            ("eval", "save", 4),
            ("train", "load", 4),
            ("eval", "load", 4),
        ]

    async def test_loading_without_a_rollout_id_forwards_none_to_the_eval_instance(
        self, ray_local_mode, patch_low_level
    ):
        """A restore of the latest checkpoint must reach the eval instance with the same unset rollout id."""
        executor = _make_executor(_make_test_args())
        executor.data_source = MagicMock()
        calls: list[tuple[str, str, object]] = []
        executor.generate_rollout = _RecordingRolloutFn("train", calls)
        executor.eval_generate_rollout = _RecordingRolloutFn("eval", calls)

        executor.load()

        assert calls == [
            ("train", "load", None),
            ("eval", "load", None),
        ]

    async def test_the_legacy_protocol_leaves_a_distinct_eval_function_untouched(
        self, ray_local_mode, patch_low_level
    ):
        """Legacy rollout functions are bare callables with no state, so neither instance is checkpointed."""
        executor = _make_executor(_make_test_args())
        executor.use_legacy_rollout_v1 = True
        executor.data_source = MagicMock()
        calls: list[tuple[str, str, object]] = []
        executor.generate_rollout = _RecordingRolloutFn("train", calls)
        executor.eval_generate_rollout = _RecordingRolloutFn("eval", calls)

        executor.save(2)
        executor.load(2)

        assert calls == []
        executor.data_source.save.assert_called_once_with(2)
        executor.data_source.load.assert_called_once_with(2)
