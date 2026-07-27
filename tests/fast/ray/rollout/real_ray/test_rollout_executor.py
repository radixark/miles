from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import ray
from tests.fast.ray.rollout.conftest import make_args, make_samples_grouped

from miles.ray.rollout.rollout_executor import RolloutExecutor
from miles.rollout.base_types import (
    BaseRolloutFn,
    RolloutFnEvalInput,
    RolloutFnEvalOutput,
    RolloutFnTrainInput,
    RolloutFnTrainOutput,
)


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
    return RolloutExecutor.__ray_actor_class__(args)


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


@pytest.mark.asyncio
class TestCheckpointing:
    async def test_save_and_load_reach_only_the_train_rollout_function(
        self,
        ray_local_mode,
        patch_low_level,
        monkeypatch,
    ):
        """One checkpoint is enough: the train and eval instances are becoming a single object."""
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
            ("train", "load", 7),
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

    async def test_save_forwards_to_the_data_source_only_for_a_global_dataset(
        self,
        ray_local_mode,
        patch_low_level,
        monkeypatch,
    ):
        """The data source is checkpointed only when it owns a global dataset; the rollout functions always are."""
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

        executor.data_source.save.assert_not_called()
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
