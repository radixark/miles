import asyncio
from argparse import Namespace
from collections import defaultdict
from pathlib import Path

import pytest
import torch
from tests.fast.ray.rollout.conftest import make_args, make_sample

from miles.ray.rollout import rollout_executor as rollout_executor_module
from miles.ray.rollout.eval_fleet import EvalFleetInfo, EvalFleetPin
from miles.ray.rollout.rollout_data_conversion import postprocess_rollout_data
from miles.ray.rollout.rollout_executor import (
    LastBatch,
    RolloutExecutor,
    compute_checkpoint_complete_marker_path,
    compute_executor_state_path,
)
from miles.rollout.base_types import (
    BaseRolloutFn,
    RolloutFnConstructorInput,
    RolloutFnEvalInput,
    RolloutFnEvalOutput,
    RolloutFnTrainInput,
    RolloutFnTrainOutput,
)
from miles.rollout.data_source import RolloutDataSource, compute_global_dataset_state_path
from miles.rollout.inference_rollout import inference_rollout_common
from miles.rollout.inference_rollout.inference_rollout_common import GenerateState
from miles.utils.audit_utils.event_logger import checkpoint as event_logger_checkpoint
from miles.utils.audit_utils.event_logger.logger import EventLogger, set_event_logger
from miles.utils.audit_utils.event_logger.models import ExplicitlyDroppedSamplesEvent
from miles.utils.audit_utils.process_identity import SimpleProcessIdentity
from miles.utils.audit_utils.sample_ownership.checker import SampleOwnershipChecker
from miles.utils.audit_utils.sample_ownership.flow import insert_data_source_issue_recorder
from miles.utils.types import Sample
from miles.utils.workers.worker_spec import HostAndPort

_REAL_EVENT_SNAPSHOT = event_logger_checkpoint.snapshot


class FakeInferenceController:
    def __init__(self) -> None:
        self.pins: list[tuple[str, str]] = []

    async def pin_eval_fleet(self, *, checkpoint_dir: str, weight_version: str) -> EvalFleetPin:
        self.pins.append((checkpoint_dir, weight_version))
        return EvalFleetPin(skip_reason=None)


class FakeInferenceControllerProvider:
    def __init__(self, controller: FakeInferenceController) -> None:
        self.controller = controller

    def get_handle(self, worker_name: str) -> FakeInferenceController:
        return self.controller


class FakeEvalFunction:
    def __init__(self) -> None:
        self.inputs: list[RolloutFnEvalInput] = []

    def __call__(self, input: RolloutFnEvalInput) -> RolloutFnEvalOutput:
        self.inputs.append(input)
        return RolloutFnEvalOutput(data={})


class _SynchronousDisposable:
    def __init__(self, disposed: list[str], name: str) -> None:
        self._disposed = disposed
        self._name = name

    def dispose(self) -> None:
        self._disposed.append(self._name)


class TestDispose:
    async def test_synchronous_train_and_eval_rollout_disposers_are_accepted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Out-of-tree rollout hooks may tear down synchronously without returning an awaitable."""
        disposed: list[str] = []
        executor = RolloutExecutor.__new__(RolloutExecutor)
        executor.use_legacy_rollout_v1 = False
        executor.generate_rollout = _SynchronousDisposable(disposed, "train")
        executor.eval_generate_rollout = _SynchronousDisposable(disposed, "eval")
        executor.data_source = object()
        executor.args = Namespace()
        executor._metric_checker = None
        monkeypatch.setattr(rollout_executor_module, "CheckpointEvalFn", _SynchronousDisposable)
        monkeypatch.setattr(rollout_executor_module.event_analyzer, "run_analysis_from_args", lambda _args: None)

        await executor.dispose()

        assert disposed == ["train", "eval"]


class TestSetEvalFleetInfo:
    async def test_setting_and_clearing_eval_fleet_info_changes_checkpoint_evaluation_routing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Installing a fleet pins checkpoint evals, while clearing it restores unpinned routing."""
        monkeypatch.setattr(inference_rollout_common, "load_tokenizer", lambda *args, **kwargs: object())
        monkeypatch.setattr(inference_rollout_common, "load_processor", lambda *args, **kwargs: object())
        controller = FakeInferenceController()
        provider = FakeInferenceControllerProvider(controller)
        eval_function = FakeEvalFunction()
        executor = RolloutExecutor.__new__(RolloutExecutor)
        executor.args = Namespace(
            chat_template_path=None,
            custom_eval_rollout_log_function_path=None,
            custom_generate_function_path=None,
            global_batch_size=1,
            hf_checkpoint="unused",
            log_passrate=False,
            n_samples_per_prompt=1,
            rollout_batch_size=1,
            rollout_max_response_len=16,
            rollout_num_gpus=1,
            rollout_num_gpus_per_engine=1,
            rollout_skip_special_tokens=True,
            rollout_stop=None,
            rollout_stop_token_ids=None,
            rollout_temperature=1.0,
            rollout_top_k=-1,
            rollout_top_p=1.0,
            save_debug_rollout_data=None,
            sglang_server_concurrency=2,
            wandb_always_use_train_step=False,
        )
        executor._inference_controller_provider = provider
        executor._eval_fleet = None
        executor._eval_lock = asyncio.Lock()
        executor.eval_generate_rollout = eval_function
        executor.rollout_id = 9
        executor._metric_checker = None
        info = EvalFleetInfo(
            router=HostAndPort(host="10.0.0.2", port=31000),
            num_gpus=2,
            num_gpus_per_engine=1,
        )

        await executor.set_eval_fleet_info(info)
        await executor._eval_checkpoint(
            rollout_id=5,
            hf_dir="/snap/step_5",
            export_time_seconds=None,
            require_marker=False,
        )
        await executor.set_eval_fleet_info(None)
        await executor._eval_checkpoint(
            rollout_id=6,
            hf_dir="/snap/step_6",
            export_time_seconds=None,
            require_marker=False,
        )

        assert controller.pins == [("/snap/step_5", "5")]
        first, second = eval_function.inputs
        assert isinstance(first.generate_state, GenerateState)
        assert first.generate_state.args.sglang_router_ip == info.router.host
        assert first.generate_state.args.sglang_router_port == info.router.port
        assert first.generate_state.args.rollout_num_gpus == info.num_gpus
        assert first.generate_state.args.rollout_num_gpus_per_engine == info.num_gpus_per_engine
        assert second.generate_state is None


class _FakeDataSource(RolloutDataSource):
    def __init__(self, path: Path) -> None:
        self._path = path
        self.loaded: list[int | None] = []

    def save(self, rollout_id: int) -> None:
        path = compute_global_dataset_state_path(self._path, rollout_id=rollout_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"sample_group_index": 1, "sample_index": 1}, path)

    def load(self, rollout_id: int | None) -> None:
        self.loaded.append(rollout_id)


class _CustomDataSource:
    def save(self, rollout_id: int) -> None:
        pass

    def load(self, rollout_id: int | None) -> None:
        pass


class _CountingRolloutFn:
    def __init__(self, start_index: int = 0) -> None:
        self.next_index = start_index
        self.num_calls = 0

    def __call__(self, args, rollout_id, data_source, evaluation) -> RolloutFnTrainOutput:
        self.num_calls += 1
        self.next_index += 1
        sample = Sample(
            index=self.next_index,
            group_index=self.next_index,
            prompt="p",
            status=Sample.Status.COMPLETED,
        )
        return RolloutFnTrainOutput(samples=[[sample]])


def _make_executor(tmp_path: Path, rollout_fn: _CountingRolloutFn) -> RolloutExecutor:
    executor = RolloutExecutor.__new__(RolloutExecutor)
    executor.args = make_args(load=str(tmp_path), save=str(tmp_path))
    executor._sample_ownership_checker = SampleOwnershipChecker(args=executor.args)
    executor.use_legacy_rollout_v1 = True
    executor.generate_rollout = rollout_fn
    executor.eval_generate_rollout = rollout_fn
    executor.data_source = _FakeDataSource(tmp_path)
    executor._train_parallel_configs_of_model_id = {None: {}}
    executor._weight_versions_of_model_id = {}
    executor._last_batch = None
    executor._replay = None
    executor._get_save_lock = asyncio.Lock()
    return executor


@pytest.fixture(autouse=True)
def _stub_rollout_postprocessing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rollout_executor_module, "postprocess_rollout_data", lambda args, data, **kwargs: (data, {}))
    monkeypatch.setattr(rollout_executor_module, "assert_samples_weight_version_sane", lambda args, samples: None)
    monkeypatch.setattr(rollout_executor_module.event_logger_checkpoint, "snapshot", lambda args, rollout_id: None)


class TestLastBatchReplay:
    async def test_checkpoint_stage_round_trips_real_event_snapshot_and_replay_pipeline(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """A checkpoint restores issued history without repeating terminal drops from any saved stage."""
        event_dir = tmp_path / "active-events"
        args = make_args(
            load=str(tmp_path),
            requested_load=str(tmp_path),
            save=str(tmp_path),
            save_debug_event_data=str(event_dir),
            global_batch_size=2,
            rewards_normalization=False,
        )
        raw = [[make_sample(group_index=group, index=group * 10)] for group in range(1, 4)]

        class AsyncRolloutFn(BaseRolloutFn):
            async def __call__(self, input: RolloutFnTrainInput) -> RolloutFnTrainOutput:
                return RolloutFnTrainOutput(samples=raw)

        class FailingRolloutFn(BaseRolloutFn):
            async def __call__(self, input: RolloutFnTrainInput) -> RolloutFnTrainOutput:
                raise AssertionError("restored raw handoff must replace generation")

        class Store:
            def put(self, *, value, value_spec):
                return value

        monkeypatch.setattr(rollout_executor_module.event_logger_checkpoint, "snapshot", _REAL_EVENT_SNAPSHOT)
        monkeypatch.setattr(rollout_executor_module, "postprocess_rollout_data", postprocess_rollout_data)
        monkeypatch.setattr(rollout_executor_module, "log_rollout_data", lambda *args, **kwargs: None)
        monkeypatch.setattr(rollout_executor_module.object_store, "get_instance", Store)
        monkeypatch.setattr(
            "miles.ray.rollout.train_data_conversion.can_schedule_on_rollout_side",
            lambda *args, **kwargs: True,
        )
        monkeypatch.setattr(
            "miles.ray.rollout.train_data_conversion.build_dp_schedule",
            lambda *args, **kwargs: ([[0]], [[[0]]], 1, 1),
        )
        event_logger = EventLogger(log_dir=event_dir, source=SimpleProcessIdentity(component="rollout_executor"))
        set_event_logger(event_logger)
        try:
            data_source = _FakeDataSource(tmp_path)
            data_source.get_samples = lambda _num_samples: raw
            insert_data_source_issue_recorder(data_source)
            data_source.get_samples(1)
            rollout_fn = AsyncRolloutFn(RolloutFnConstructorInput(args=args, data_source=data_source))
            executor = _make_executor(tmp_path, _CountingRolloutFn())
            executor.data_source = data_source
            self._configure_async_executor(executor, args=args, rollout_fn=rollout_fn)
            await executor.get(rollout_id=1)
            await executor.save(0)
            (tmp_path / "latest_checkpointed_iteration.txt").write_text("0")

            resumed_fn = FailingRolloutFn(RolloutFnConstructorInput(args=args, data_source=_FakeDataSource(tmp_path)))
            resumed = _make_executor(tmp_path, _CountingRolloutFn())
            self._configure_async_executor(resumed, args=args, rollout_fn=resumed_fn)
            await resumed.load(0)
            await resumed.get(rollout_id=1)
        finally:
            set_event_logger(None)

        events = event_logger.read_events_strict()
        drops = [event for event in events if isinstance(event, ExplicitlyDroppedSamplesEvent)]
        assert [(event.sample_indices, event.reason) for event in drops] == [
            ([30], "trim"),
            ([20], "dp_schedule_trim"),
        ]
        assert compute_checkpoint_complete_marker_path(tmp_path, rollout_id=0).is_file()

    @staticmethod
    def _configure_async_executor(executor: RolloutExecutor, *, args: Namespace, rollout_fn: BaseRolloutFn) -> None:
        executor.args = args
        executor.use_legacy_rollout_v1 = False
        executor.generate_rollout = rollout_fn
        executor.eval_generate_rollout = rollout_fn
        executor._rollouts_since_publish_of_model_id = defaultdict(int)
        executor._sample_ownership_checker = SampleOwnershipChecker(args=args)
        executor._metric_checker = None
        executor.custom_convert_samples_to_train_data_func = None
        executor.custom_reward_post_process_func = None
        executor._train_parallel_configs_of_model_id = {None: {"dp_size": 1}}

    @pytest.mark.parametrize("trained", [False, True])
    async def test_only_untrained_batches_are_replayed_once(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, trained: bool
    ) -> None:
        """Checkpoint replay preserves a pending batch and excludes already trained data."""
        executor = _make_executor(tmp_path, _CountingRolloutFn())
        executor._last_batch = LastBatch(rollout_id=3, train_data={"sample_indices": [7]})
        await executor.save(3 if trained else 2)
        restored = _make_executor(tmp_path, _CountingRolloutFn())
        await restored.load(3 if trained else 2)
        if trained:
            assert restored._replay is None
            return
        monkeypatch.setattr(restored, "_publish_train_data", lambda **kwargs: kwargs["data"])
        assert await restored.get(rollout_id=3) == {"sample_indices": [7]}
        assert restored._replay is None

    async def test_checkpoint_waits_for_active_get(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A save during generation includes the completed prefetched batch."""
        executor = _make_executor(tmp_path, _CountingRolloutFn())
        entered, release = asyncio.Event(), asyncio.Event()

        async def get(rollout_id: int, trainer_model_id: str | None) -> None:
            entered.set()
            await release.wait()
            executor._last_batch = LastBatch(rollout_id=rollout_id, train_data={"sample_indices": [7]})

        monkeypatch.setattr(executor, "_get", get)
        fetching = asyncio.create_task(executor.get(rollout_id=3))
        await entered.wait()
        saving = asyncio.create_task(executor.save(2))
        await asyncio.sleep(0)
        assert not saving.done()
        release.set()
        await fetching
        await saving
        state = torch.load(compute_executor_state_path(tmp_path, rollout_id=2), weights_only=False)
        assert state["last_batch"].train_data == {"sample_indices": [7]}


class TestCheckpointCompleteMarker:
    async def test_save_publishes_the_complete_marker_after_all_state(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The marker exists only after data source, rollout, executor, and event state finish saving."""
        executor = _make_executor(tmp_path, _CountingRolloutFn())
        marker = compute_checkpoint_complete_marker_path(tmp_path, rollout_id=2)

        def snapshot(args: Namespace, rollout_id: int) -> None:
            assert not marker.exists()

        monkeypatch.setattr(rollout_executor_module.event_logger_checkpoint, "snapshot", snapshot)

        await executor.save(2)

        assert compute_executor_state_path(tmp_path, rollout_id=2).is_file()
        assert marker.is_file()

    async def test_a_failed_overwrite_removes_the_previous_complete_marker(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An interrupted overwrite cannot leave an old marker claiming the new state is complete."""
        executor = _make_executor(tmp_path, _CountingRolloutFn())
        await executor.save(2)
        marker = compute_checkpoint_complete_marker_path(tmp_path, rollout_id=2)

        async def fail_save(rollout_id: int) -> None:
            assert not marker.exists()
            raise RuntimeError("save interrupted")

        monkeypatch.setattr(executor, "_save_state", fail_save)
        with pytest.raises(RuntimeError, match="save interrupted"):
            await executor.save(2)
        assert not marker.exists()

    async def test_partial_rollout_state_without_a_marker_is_refused(self, tmp_path: Path) -> None:
        """Files from a mid-save crash cannot be mistaken for a restorable checkpoint."""
        state_dir = tmp_path / "rollout"
        state_dir.mkdir()
        (state_dir / "executor_state_5.pt").write_text("partial")
        executor = _make_executor(tmp_path, _CountingRolloutFn())

        with pytest.raises(AssertionError, match="no complete_5 marker"):
            await executor.load(5)

    async def test_a_restored_trainer_requires_complete_rollout_state(self, tmp_path: Path) -> None:
        """A numbered trainer checkpoint cannot resume with absent rollout-side state."""
        executor = _make_executor(tmp_path, _CountingRolloutFn())

        with pytest.raises(AssertionError, match="no complete_5 marker"):
            await executor.load(5, require_complete=True)

    async def test_a_marker_without_executor_state_is_refused(self, tmp_path: Path) -> None:
        """A corrupt checkpoint cannot use its marker to hide a missing mandatory executor file."""
        executor = _make_executor(tmp_path, _CountingRolloutFn())
        await executor.save(5)
        compute_executor_state_path(tmp_path, rollout_id=5).unlink()

        with pytest.raises(AssertionError, match="executor_state_5.pt"):
            await executor.load(5)

    async def test_a_marker_without_data_source_state_is_refused(self, tmp_path: Path) -> None:
        """Sample identity cursors are mandatory even when the global dataset is disabled."""
        executor = _make_executor(tmp_path, _CountingRolloutFn())
        await executor.save(5)
        compute_global_dataset_state_path(tmp_path, rollout_id=5).unlink()

        with pytest.raises(AssertionError, match="global_dataset_state_dict_5.pt"):
            await executor.load(5)

    async def test_custom_data_source_does_not_imply_the_builtin_state_file(self, tmp_path: Path) -> None:
        """A custom source keeps its own checkpoint contract instead of writing the built-in cursor file."""
        executor = _make_executor(tmp_path, _CountingRolloutFn())
        executor.data_source = _CustomDataSource()

        await executor.save(5)
        await executor.load(5)

        assert compute_checkpoint_complete_marker_path(tmp_path, rollout_id=5).is_file()

    async def test_a_marker_without_fully_async_state_is_refused(self, tmp_path: Path) -> None:
        """A fully async checkpoint cannot discard its queued and in-flight work."""
        executor = _make_executor(tmp_path, _CountingRolloutFn())
        await executor.save(5)
        executor.args.fully_async = True

        with pytest.raises(AssertionError, match="fully_async_state_5.pt"):
            await executor.load(5)

    async def test_a_marker_without_event_snapshot_is_refused(self, tmp_path: Path) -> None:
        """An accounting-enabled checkpoint cannot forget the issued and terminal events."""
        executor = _make_executor(tmp_path, _CountingRolloutFn())
        await executor.save(5)
        executor.args.save_debug_event_data = str(tmp_path / "events")

        with pytest.raises(AssertionError, match="missing required state"):
            await executor.load(5)

    async def test_an_empty_optional_load_warns_and_continues(self, tmp_path: Path) -> None:
        """A fresh run with no trainer checkpoint may start without rollout state."""
        executor = _make_executor(tmp_path, _CountingRolloutFn())

        await executor.load(5)

        assert executor.data_source.loaded == [5]
