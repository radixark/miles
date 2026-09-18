import asyncio
from argparse import Namespace
from collections import defaultdict
from pathlib import Path

import pytest
import torch
from tests.fast.ray.rollout.conftest import make_args, make_sample

from miles.backends.megatron_utils.ft.types import TrainStepOutcome
from miles.ray.rollout import rollout_executor as rollout_executor_module
from miles.ray.rollout.eval_fleet import EvalFleetInfo, EvalFleetPin
from miles.ray.rollout.output_snapshotter import _RolloutExecutorOutputSnapshotter
from miles.ray.rollout.rollout_data_conversion import postprocess_rollout_data
from miles.ray.rollout.rollout_executor import RolloutExecutor, compute_rollout_checkpoint_dir
from miles.rollout.base_types import (
    BaseRolloutFn,
    RolloutFnConstructorInput,
    RolloutFnEvalInput,
    RolloutFnEvalOutput,
    RolloutFnTrainInput,
    RolloutFnTrainOutput,
)
from miles.rollout.data_source import RolloutDataSource
from miles.rollout.inference_rollout import inference_rollout_common
from miles.rollout.inference_rollout.inference_rollout_common import GenerateState
from miles.utils.audit_utils.event_logger.logger import EventLogger, read_events, set_event_logger
from miles.utils.audit_utils.event_logger.models import (
    DataSourceIssuedSamplesEvent,
    ExplicitlyDroppedSamplesEvent,
    TrainGroupStepEndEvent,
)
from miles.utils.audit_utils.process_identity import SimpleProcessIdentity
from miles.utils.audit_utils.sample_ownership.recorder import SampleOwnershipRecorder
from miles.utils.object_store import _MooncakeStoreObjectRef
from miles.utils.types import Sample, SampleLineage
from miles.utils.workers.worker_spec import HostAndPort


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
        executor.args = Namespace(enable_sample_ownership_checker=False)
        executor._metric_checker = None
        monkeypatch.setattr(rollout_executor_module, "CheckpointEvalFn", _SynchronousDisposable)
        monkeypatch.setattr(rollout_executor_module.event_analyzer, "run_analysis_from_args", lambda _args: None)
        monkeypatch.setattr(
            rollout_executor_module.event_analyzer, "run_sample_ownership_analysis", lambda *, args: None
        )

        await executor.dispose()

        assert disposed == ["train", "eval"]

    async def test_the_final_ownership_check_runs_before_the_run_ends(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Violations that only mature after the last get would otherwise never be checked."""
        checked: list[Namespace] = []
        executor = RolloutExecutor.__new__(RolloutExecutor)
        executor.use_legacy_rollout_v1 = False
        executor.generate_rollout = None
        executor.eval_generate_rollout = None
        executor.data_source = object()
        executor.args = Namespace(enable_sample_ownership_checker=False)
        executor._metric_checker = None
        monkeypatch.setattr(rollout_executor_module, "CheckpointEvalFn", _SynchronousDisposable)
        monkeypatch.setattr(rollout_executor_module.event_analyzer, "run_analysis_from_args", lambda _args: None)
        monkeypatch.setattr(
            rollout_executor_module.event_analyzer,
            "run_sample_ownership_analysis",
            lambda *, args: checked.append(args),
        )

        await executor.dispose()

        assert checked == [executor.args]


class TestShutdownAccounting:
    @pytest.mark.parametrize("with_lineage", [False, True])
    async def test_shutdown_does_not_drop_a_trimmed_source_twice(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, with_lineage: bool
    ) -> None:
        """A raw prefetched snapshot retains sources already dropped during DP scheduling."""
        executor, event_dir = _make_shutdown_executor(tmp_path, monkeypatch)
        samples = [Sample(index=index) for index in (22, 23)]
        if with_lineage:
            for sample in samples:
                sample.lineage = SampleLineage(source_sample_index=sample.index, output_index=0, output_count=1)
                sample.index += 100
        executor._output_snapshotter.capture(trainer_model_id=None, rollout_id=2, data=samples, metadata={})
        event_logger = EventLogger(log_dir=event_dir, source=SimpleProcessIdentity(component="rollout_executor"))
        event_logger.log(
            ExplicitlyDroppedSamplesEvent,
            dict(source_sample_indices=[23], reason="dp_schedule_trim"),
            print_log=False,
        )

        await _dispose_with_one_trained_step(executor, event_dir)

        drops = [event for event in read_events(event_dir) if isinstance(event, ExplicitlyDroppedSamplesEvent)]
        assert [(event.source_sample_indices, event.reason) for event in drops] == [
            ([23], "dp_schedule_trim"),
            ([22], "shutdown_prefetched"),
        ]

    async def test_a_prefetched_batch_no_trainer_consumed_is_recorded_as_dropped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Only the snapshotter holds the batch generated ahead of a shutdown, so nothing else records its loss."""
        executor, event_dir = _make_shutdown_executor(tmp_path, monkeypatch)
        executor._output_snapshotter.capture(trainer_model_id=None, rollout_id=1, data=[Sample(index=11)], metadata={})
        executor._output_snapshotter.capture(trainer_model_id=None, rollout_id=2, data=[Sample(index=22)], metadata={})

        await _dispose_with_one_trained_step(executor, event_dir)

        drops = [event for event in read_events(event_dir) if isinstance(event, ExplicitlyDroppedSamplesEvent)]
        assert [(event.source_sample_indices, event.reason) for event in drops] == [([22], "shutdown_prefetched")]

    async def test_a_dispose_without_a_pending_batch_records_nothing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A clean shutdown owes the ownership log no drops at all."""
        executor, event_dir = _make_shutdown_executor(tmp_path, monkeypatch)

        await _dispose_with_one_trained_step(executor, event_dir)

        assert [event for event in read_events(event_dir) if isinstance(event, ExplicitlyDroppedSamplesEvent)] == []

    async def test_disposing_twice_records_the_abandoned_batch_once(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A second drop of one source would itself be an ownership violation."""
        executor, event_dir = _make_shutdown_executor(tmp_path, monkeypatch)
        executor._output_snapshotter.capture(trainer_model_id=None, rollout_id=2, data=[Sample(index=22)], metadata={})

        await _dispose_with_one_trained_step(executor, event_dir, times=2)

        drops = [event for event in read_events(event_dir) if isinstance(event, ExplicitlyDroppedSamplesEvent)]
        assert [(event.source_sample_indices, event.reason) for event in drops] == [([22], "shutdown_prefetched")]


def _make_shutdown_executor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[RolloutExecutor, Path]:
    event_dir = tmp_path / "events"
    executor = _make_executor(tmp_path, _CountingRolloutFn())
    executor.args = make_args(
        load=str(tmp_path),
        save=str(tmp_path),
        save_debug_event_data=str(event_dir),
        enable_sample_ownership_checker=True,
    )
    executor._metric_checker = None
    executor._output_snapshotter = _RolloutExecutorOutputSnapshotter(args=executor.args)
    monkeypatch.setattr(
        rollout_executor_module.event_analyzer, "run_sample_ownership_analysis", lambda **_kwargs: None
    )
    monkeypatch.setattr(rollout_executor_module.event_analyzer, "run_analysis_from_args", lambda _args: None)
    return executor, event_dir


async def _dispose_with_one_trained_step(executor: RolloutExecutor, event_dir: Path, *, times: int = 1) -> None:
    event_logger = EventLogger(log_dir=event_dir, source=SimpleProcessIdentity(component="rollout_executor"))
    set_event_logger(event_logger)
    try:
        event_logger.log(
            TrainGroupStepEndEvent,
            dict(rollout_id=1, attempt=0, role="actor", cell_outcomes={0: [TrainStepOutcome.NORMAL]}),
            print_log=False,
        )
        for _ in range(times):
            await executor.dispose()
    finally:
        set_event_logger(None)


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
        executor.last_get_rollout_id_of_model_id = {None: 9}
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
        self.args = Namespace(load=path, rollout_global_dataset=False)
        self.loaded: list[Path] = []

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        torch.save({"sample_group_index": 1, "sample_index": 1}, directory / "state.pt")

    def load(self, directory: Path) -> None:
        super().load(directory)
        self.loaded.append(directory)


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


def _load_executor_state(directory: Path, *, rollout_id: int) -> dict:
    path = compute_rollout_checkpoint_dir(directory, rollout_id=rollout_id) / "executor" / "state.pt"
    return torch.load(path, weights_only=False)


def _make_executor(tmp_path: Path, rollout_fn: _CountingRolloutFn) -> RolloutExecutor:
    executor = RolloutExecutor.__new__(RolloutExecutor)
    executor.args = make_args(load=str(tmp_path), save=str(tmp_path))
    executor.use_legacy_rollout_v1 = True
    executor.generate_rollout = rollout_fn
    executor.eval_generate_rollout = rollout_fn
    executor.data_source = _FakeDataSource(tmp_path)
    executor._train_parallel_configs_of_model_id = {None: {}}
    executor._weight_versions_of_model_id = {}
    executor.last_get_rollout_id_of_model_id = {}
    executor._rollout_id_being_served = None
    executor.custom_convert_samples_to_train_data_func = None
    executor.custom_reward_post_process_func = None
    executor._output_snapshotter = _RolloutExecutorOutputSnapshotter(args=executor.args)
    return executor


@pytest.fixture(autouse=True)
def _stub_rollout_postprocessing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rollout_executor_module, "postprocess_rollout_data", lambda args, data, **kwargs: (data, {}))
    monkeypatch.setattr(rollout_executor_module, "assert_samples_weight_version_sane", lambda args, samples: None)


class TestOutputSnapshotReplay:
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
            enable_sample_ownership_checker=True,
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
                return _MooncakeStoreObjectRef(payload=value)

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
            SampleOwnershipRecorder.install(args=args, data_source=data_source, current_rollout_id=lambda: 0)
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

        events = read_events(event_logger.log_dir)
        drops = [event for event in events if isinstance(event, ExplicitlyDroppedSamplesEvent)]
        assert [(event.source_sample_indices, event.reason) for event in drops] == [
            ([30], "trim"),
            ([20], "dp_schedule_trim"),
        ]

    @staticmethod
    def _configure_async_executor(executor: RolloutExecutor, *, args: Namespace, rollout_fn: BaseRolloutFn) -> None:
        executor.args = args
        executor.use_legacy_rollout_v1 = False
        executor.generate_rollout = rollout_fn
        executor.eval_generate_rollout = rollout_fn
        executor._rollouts_since_publish_of_model_id = defaultdict(int)
        executor._metric_checker = None
        executor.custom_convert_samples_to_train_data_func = None
        executor.custom_reward_post_process_func = None
        executor._train_parallel_configs_of_model_id = {None: {"dp_size": 1}}

    async def test_a_pending_sample_snapshot_is_replayed_after_a_checkpoint(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Checkpoint replay hands a batch the trainer never received to the resumed run."""
        executor = _make_executor(tmp_path, _CountingRolloutFn())
        executor._output_snapshotter.capture(trainer_model_id=None, rollout_id=3, data=[Sample(index=7)], metadata={})
        await executor.save(2)
        restored = _make_executor(tmp_path, _CountingRolloutFn())
        await restored.load(2)

        caller_loop = asyncio.get_running_loop()

        def convert(_args, data, **kwargs):
            assert asyncio.get_running_loop() is caller_loop
            return {"sample_indices": [sample.index for sample in data]}

        monkeypatch.setattr(rollout_executor_module, "convert_samples_to_train_data", convert)
        monkeypatch.setattr(rollout_executor_module, "split_train_data_by_dp", lambda *_args: None)
        assert (await restored.get(rollout_id=3)).sample_indices == [7]
        await restored.save(2)
        resumed_again = _make_executor(tmp_path, _CountingRolloutFn())
        await resumed_again.load(2)
        assert (await resumed_again.get(rollout_id=3)).sample_indices == [7]

    async def test_a_generated_batch_is_captured_before_a_concurrent_save_runs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An await between generation and capture would let a save in that window miss the new batch."""
        executor = _make_executor(tmp_path, _CountingRolloutFn())
        generated = asyncio.Event()

        async def generate_rollout_data(*, rollout_id: int, trainer_model_id: str | None):
            generated.set()
            return [Sample(index=7)], {}

        async def save_once_generated() -> None:
            await generated.wait()
            await executor.save(2)

        monkeypatch.setattr(executor, "_generate_rollout_data", generate_rollout_data)
        monkeypatch.setattr(rollout_executor_module, "convert_samples_to_train_data", lambda *_args, **_kw: {})
        monkeypatch.setattr(rollout_executor_module, "split_train_data_by_dp", lambda *_args: None)
        saving = asyncio.create_task(save_once_generated())
        await asyncio.sleep(0)

        await executor.get(rollout_id=3)
        await saving

        data, _metadata = _load_executor_state(tmp_path, rollout_id=2)[None, 3]
        assert [sample.index for sample in data] == [7]

    async def test_checkpoint_captures_state_without_waiting_for_suspended_get(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Checkpointing snapshots current state atomically while generation is suspended."""
        executor = _make_executor(tmp_path, _CountingRolloutFn())
        entered, release = asyncio.Event(), asyncio.Event()

        async def generate_rollout_data(
            *, rollout_id: int, trainer_model_id: str | None
        ) -> tuple[list[Sample], dict[str, object]]:
            entered.set()
            await release.wait()
            return [Sample(index=7)], {}

        monkeypatch.setattr(executor, "_generate_rollout_data", generate_rollout_data)
        monkeypatch.setattr(rollout_executor_module, "convert_samples_to_train_data", lambda *_args, **_kw: {})
        monkeypatch.setattr(rollout_executor_module, "split_train_data_by_dp", lambda *_args: None)
        fetching = asyncio.create_task(executor.get(rollout_id=3))
        await entered.wait()
        await executor.save(2)
        assert _load_executor_state(tmp_path, rollout_id=2) == {}
        release.set()
        await fetching
        await executor.save(2)
        data, _metadata = _load_executor_state(tmp_path, rollout_id=2)[None, 3]
        assert [sample.index for sample in data] == [7]


class TestSampleOwnershipRolloutId:
    async def test_samples_issued_during_a_get_are_recorded_against_that_rollout_id(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The data source issues samples mid-get, so the recorder must stamp the rollout being served."""
        event_dir = tmp_path / "events"
        executor = _make_executor(tmp_path, _CountingRolloutFn())
        executor.args = make_args(
            load=str(tmp_path),
            save=str(tmp_path),
            save_debug_event_data=str(event_dir),
            enable_sample_ownership_checker=True,
        )
        executor._rollout_id_being_served = None
        executor.last_get_rollout_id_of_model_id = {}
        executor.data_source.get_samples = lambda _num_samples: [[make_sample(group_index=3, index=10)]]
        SampleOwnershipRecorder.install(
            args=executor.args, data_source=executor.data_source, current_rollout_id=executor._current_rollout_id
        )

        async def generate_rollout_data(*, rollout_id: int, trainer_model_id: str | None):
            executor.data_source.get_samples(1)
            return [Sample(index=10)], {}

        monkeypatch.setattr(executor, "_generate_rollout_data", generate_rollout_data)
        monkeypatch.setattr(rollout_executor_module, "convert_samples_to_train_data", lambda *_args, **_kw: {})
        monkeypatch.setattr(rollout_executor_module, "split_train_data_by_dp", lambda *_args: None)
        monkeypatch.setattr(
            rollout_executor_module.event_analyzer, "run_sample_ownership_analysis", lambda *, args: None
        )
        set_event_logger(EventLogger(log_dir=event_dir, source=SimpleProcessIdentity(component="rollout_executor")))
        try:
            await executor.get(rollout_id=7)
        finally:
            set_event_logger(None)

        [event] = [x for x in read_events(event_dir) if isinstance(x, DataSourceIssuedSamplesEvent)]
        assert event.rollout_id == 7


class _CustomDataSource:
    def save(self, directory: Path) -> None:
        pass

    def load(self, directory: Path) -> None:
        pass


class TestOneDirectoryPerRolloutCheckpoint:
    async def test_the_directory_appears_only_after_every_component_saved(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A half-written checkpoint under the published name would be restored as if it were whole."""
        executor = _make_executor(tmp_path, _CountingRolloutFn())
        published = compute_rollout_checkpoint_dir(tmp_path, rollout_id=2)
        original = executor._output_snapshotter.save

        def save(directory: Path) -> None:
            assert not published.exists()
            original(directory)

        monkeypatch.setattr(executor._output_snapshotter, "save", save)

        await executor.save(2)

        assert sorted(one.name for one in published.iterdir()) == ["data_source", "executor"]

    async def test_an_interrupted_save_publishes_nothing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A crash mid-save must leave no directory a resume would trust, and no rubbish behind either."""
        executor = _make_executor(tmp_path, _CountingRolloutFn())

        def fail(directory: Path) -> None:
            raise RuntimeError("save interrupted")

        monkeypatch.setattr(executor._output_snapshotter, "save", fail)
        with pytest.raises(RuntimeError, match="save interrupted"):
            await executor.save(2)

        assert not compute_rollout_checkpoint_dir(tmp_path, rollout_id=2).exists()
        assert list((tmp_path / "rollout").glob(".tmp-*")) == []

    async def test_saving_the_same_rollout_again_replaces_the_published_directory(self, tmp_path: Path) -> None:
        """A re-save of one rollout id has to land whole, so the old directory is swapped out, not written into."""
        executor = _make_executor(tmp_path, _CountingRolloutFn())
        await executor.save(2)
        executor._output_snapshotter.capture(trainer_model_id=None, rollout_id=3, data=[Sample(index=7)], metadata={})

        await executor.save(2)

        data, _metadata = _load_executor_state(tmp_path, rollout_id=2)[None, 3]
        assert [sample.index for sample in data] == [7]

    async def test_a_restored_trainer_requires_the_rollout_directory(self, tmp_path: Path) -> None:
        """A numbered trainer checkpoint cannot resume with absent rollout-side state."""
        executor = _make_executor(tmp_path, _CountingRolloutFn())

        with pytest.raises(AssertionError, match="cannot resume that state"):
            await executor.load(5)

    async def test_a_step_that_was_never_trained_is_refused(self, tmp_path: Path) -> None:
        """A run whose trainer starts from scratch has no rollout state, and must not be asked for any."""
        executor = _make_executor(tmp_path, _CountingRolloutFn())

        with pytest.raises(AssertionError, match="is not a trained step"):
            await executor.load(-1)

        assert executor.data_source.loaded == []

    async def test_a_file_missing_from_the_directory_is_refused(self, tmp_path: Path) -> None:
        """The directory is published whole, so a file missing inside it is corruption, not a fresh start."""
        executor = _make_executor(tmp_path, _CountingRolloutFn())
        await executor.save(5)
        (compute_rollout_checkpoint_dir(tmp_path, rollout_id=5) / "executor" / "state.pt").unlink()

        with pytest.raises(AssertionError, match="executor/state.pt"):
            await executor.load(5)

    async def test_a_custom_data_source_does_not_imply_the_builtin_state_file(self, tmp_path: Path) -> None:
        """A custom source keeps its own checkpoint contract instead of writing the built-in cursor file."""
        executor = _make_executor(tmp_path, _CountingRolloutFn())
        executor.data_source = _CustomDataSource()

        await executor.save(5)
        await executor.load(5)

        assert not (compute_rollout_checkpoint_dir(tmp_path, rollout_id=5) / "data_source").exists()

    async def test_a_run_that_saves_nowhere_refuses_to_save(self, tmp_path: Path) -> None:
        """Without --save the orchestration never asks for a checkpoint, so a save is a bug."""
        executor = _make_executor(tmp_path, _CountingRolloutFn())
        executor.args.save = None

        with pytest.raises(AssertionError, match="only saves when --save"):
            await executor.save(2)

        assert not (tmp_path / "rollout").exists()

    async def test_a_configured_event_log_must_reach_the_checkpoint(self, tmp_path: Path) -> None:
        """An accounting-enabled run cannot publish a checkpoint that forgot the issued and terminal events."""
        executor = _make_executor(tmp_path, _CountingRolloutFn())
        executor.args.save_debug_event_data = str(tmp_path / "absent-events")

        with pytest.raises(AssertionError, match="absent-events"):
            await executor.save(5)

        assert not compute_rollout_checkpoint_dir(tmp_path, rollout_id=5).exists()


class TestRolloutCheckpointDir:
    def test_every_rollout_gets_its_own_directory(self, tmp_path: Path) -> None:
        """The rollout id names the directory, so two checkpoints never share one."""
        assert compute_rollout_checkpoint_dir(tmp_path, rollout_id=3) == tmp_path / "rollout" / "3"
