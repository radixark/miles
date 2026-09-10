import asyncio
import json
from argparse import Namespace
from pathlib import Path

import pytest
import torch
from tests.fast.ray.rollout.conftest import make_args

from miles.ray.rollout import rollout_executor as rollout_executor_module
from miles.ray.rollout.eval_fleet import EvalFleetInfo, EvalFleetPin
from miles.ray.rollout.rollout_executor import (
    RolloutExecutor,
    compute_checkpoint_complete_marker_path,
    compute_executor_state_path,
)
from miles.rollout.base_types import RolloutFnEvalInput, RolloutFnEvalOutput, RolloutFnTrainOutput
from miles.rollout.data_source import RolloutDataSource, compute_global_dataset_state_path
from miles.rollout.inference_rollout import inference_rollout_common
from miles.rollout.inference_rollout.inference_rollout_common import GenerateState
from miles.utils.types import Sample
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
    executor.use_legacy_rollout_v1 = True
    executor.generate_rollout = rollout_fn
    executor.eval_generate_rollout = rollout_fn
    executor.data_source = _FakeDataSource(tmp_path)
    executor._train_parallel_configs_of_model_id = {None: {}}
    executor._weight_versions_of_model_id = {}
    executor._last_batch = None
    executor._replay = None
    executor._replay_stage = None
    executor._replay_train_data = None
    return executor


@pytest.fixture(autouse=True)
def _stub_rollout_postprocessing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rollout_executor_module, "postprocess_rollout_data", lambda args, data, **kwargs: (data, {}))
    monkeypatch.setattr(rollout_executor_module, "assert_samples_weight_version_sane", lambda args, samples: None)
    monkeypatch.setattr(rollout_executor_module.event_logger_checkpoint, "snapshot", lambda args, rollout_id: None)


class TestLastBatchReplay:
    async def test_a_prefetched_batch_after_the_checkpoint_is_replayed_after_restore(self, tmp_path: Path) -> None:
        """A crash between get(r+1) and train(r+1) cannot silently lose the prefetched batch."""
        rollout_fn = _CountingRolloutFn()
        executor = _make_executor(tmp_path, rollout_fn)
        before, _, _ = await executor._get_rollout_data(rollout_id=1)
        executor.save(0)

        resumed_fn = _CountingRolloutFn(start_index=rollout_fn.next_index)
        resumed = _make_executor(tmp_path, resumed_fn)
        resumed.load(0)
        after, _, _ = await resumed._get_rollout_data(rollout_id=1)

        assert [[sample.index for sample in group] for group in after] == [
            [sample.index for sample in group] for group in before
        ]
        assert resumed_fn.num_calls == 0

    async def test_a_replay_is_consumed_once(self, tmp_path: Path) -> None:
        """A restored batch stands in for one get and the following rollout generates normally."""
        executor = _make_executor(tmp_path, _CountingRolloutFn())
        await executor._get_rollout_data(rollout_id=1)
        executor.save(0)
        resumed = _make_executor(tmp_path, _CountingRolloutFn())
        resumed.load(0)

        await resumed._get_rollout_data(rollout_id=1)
        await resumed._get_rollout_data(rollout_id=2)

        assert resumed.generate_rollout.num_calls == 1

    async def test_a_delivered_batch_resumes_after_postprocessing(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A replay saved after delivery does not repeat trim processing or its terminal events."""
        sample = Sample(index=1, group_index=1, status=Sample.Status.COMPLETED)
        executor = _make_executor(tmp_path, _CountingRolloutFn())
        executor._record_processed_batch(
            rollout_id=1,
            samples=[sample],
            metadata={"stage": "final"},
            stage="delivered",
        )
        executor.save(0)
        resumed = _make_executor(tmp_path, _CountingRolloutFn())
        resumed.load(0)
        monkeypatch.setattr(
            rollout_executor_module,
            "postprocess_rollout_data",
            lambda *args, **kwargs: pytest.fail("delivered replay must not rerun postprocessing"),
        )

        samples, metadata, _ = await resumed._get_rollout_data(rollout_id=1)

        assert samples[0].index == 1
        assert metadata == {"stage": "final"}
        assert resumed._replay_stage == "delivered"

    async def test_a_different_rollout_id_does_not_consume_the_pending_replay(self, tmp_path: Path) -> None:
        """An unrelated get leaves the exact recorded rollout available for its own id."""
        executor = _make_executor(tmp_path, _CountingRolloutFn())
        await executor._get_rollout_data(rollout_id=1)
        executor.save(0)
        resumed = _make_executor(tmp_path, _CountingRolloutFn())
        resumed.load(0)

        await resumed._get_rollout_data(rollout_id=2)
        await resumed._get_rollout_data(rollout_id=1)

        assert resumed.generate_rollout.num_calls == 1

    async def test_a_batch_covered_by_the_checkpoint_is_not_saved_for_replay(self, tmp_path: Path) -> None:
        """A batch at or before the saved weight step is already represented by the checkpoint."""
        executor = _make_executor(tmp_path, _CountingRolloutFn())
        await executor._get_rollout_data(rollout_id=3)

        executor.save(3)

        state = torch.load(compute_executor_state_path(tmp_path, rollout_id=3), weights_only=False)
        assert state["last_batch"] is None

    def test_downstream_mutation_does_not_change_the_recorded_raw_batch(self, tmp_path: Path) -> None:
        """Conversion and metadata mutation cannot corrupt the raw batch kept for checkpoint replay."""
        executor = _make_executor(tmp_path, _CountingRolloutFn())
        batch = [
            [
                Sample(
                    index=1,
                    group_index=1,
                    prompt="p",
                    metadata={"slots": [3]},
                    status=Sample.Status.COMPLETED,
                )
            ]
        ]
        executor._record_last_batch(rollout_id=1, samples=batch)
        batch[0][0].metadata.pop("slots")

        executor.save(0)

        state = torch.load(compute_executor_state_path(tmp_path, rollout_id=0), weights_only=False)
        assert state["last_batch"].samples[0][0].metadata == {"slots": [3]}


class TestCheckpointCompleteMarker:
    def test_save_publishes_the_complete_marker_after_all_state(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The marker exists only after data source, rollout, executor, and event state finish saving."""
        executor = _make_executor(tmp_path, _CountingRolloutFn())
        marker = compute_checkpoint_complete_marker_path(tmp_path, rollout_id=2)

        def snapshot(args: Namespace, rollout_id: int) -> None:
            assert not marker.exists()

        monkeypatch.setattr(rollout_executor_module.event_logger_checkpoint, "snapshot", snapshot)

        executor.save(2)

        assert compute_executor_state_path(tmp_path, rollout_id=2).is_file()
        assert marker.is_file()

    def test_a_failed_overwrite_removes_the_previous_complete_marker(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An interrupted overwrite cannot leave an old marker claiming the new state is complete."""
        executor = _make_executor(tmp_path, _CountingRolloutFn())
        executor.save(2)
        marker = compute_checkpoint_complete_marker_path(tmp_path, rollout_id=2)

        async def fail_save(rollout_id: int) -> None:
            assert not marker.exists()
            raise RuntimeError("save interrupted")

        monkeypatch.setattr(executor, "_save_state", fail_save)
        with pytest.raises(RuntimeError, match="save interrupted"):
            executor.save(2)
        assert not marker.exists()

    def test_partial_rollout_state_without_a_marker_is_refused(self, tmp_path: Path) -> None:
        """Files from a mid-save crash cannot be mistaken for a restorable checkpoint."""
        state_dir = tmp_path / "rollout"
        state_dir.mkdir()
        (state_dir / "executor_state_5.pt").write_text("partial")
        executor = _make_executor(tmp_path, _CountingRolloutFn())

        with pytest.raises(AssertionError, match="no complete_5 marker"):
            executor.load(5)

    def test_a_restored_trainer_requires_complete_rollout_state(self, tmp_path: Path) -> None:
        """A numbered trainer checkpoint cannot resume with absent rollout-side state."""
        executor = _make_executor(tmp_path, _CountingRolloutFn())

        with pytest.raises(AssertionError, match="no complete_5 marker"):
            executor.load(5, require_complete=True)

    def test_a_marker_without_executor_state_is_refused(self, tmp_path: Path) -> None:
        """A corrupt checkpoint cannot use its marker to hide a missing mandatory executor file."""
        executor = _make_executor(tmp_path, _CountingRolloutFn())
        executor.save(5)
        compute_executor_state_path(tmp_path, rollout_id=5).unlink()

        with pytest.raises(AssertionError, match="no executor_state_5.pt"):
            executor.load(5)

    def test_a_marker_without_data_source_state_is_refused(self, tmp_path: Path) -> None:
        """Sample identity cursors are mandatory even when the global dataset is disabled."""
        executor = _make_executor(tmp_path, _CountingRolloutFn())
        executor.save(5)
        compute_global_dataset_state_path(tmp_path, rollout_id=5).unlink()

        with pytest.raises(AssertionError, match="global_dataset_state_dict_5.pt"):
            executor.load(5)

    def test_custom_data_source_does_not_imply_the_builtin_state_file(self, tmp_path: Path) -> None:
        """A custom source keeps its own checkpoint contract instead of writing the built-in cursor file."""
        executor = _make_executor(tmp_path, _CountingRolloutFn())
        executor.data_source = _CustomDataSource()

        executor.save(5)
        executor.load(5)

        assert compute_checkpoint_complete_marker_path(tmp_path, rollout_id=5).is_file()

    def test_a_marker_rejects_invalid_data_source_identity(self, tmp_path: Path) -> None:
        """A complete checkpoint cannot resume with missing sample identity cursors."""
        executor = _make_executor(tmp_path, _CountingRolloutFn())
        executor.save(5)
        torch.save({}, compute_global_dataset_state_path(tmp_path, rollout_id=5))

        with pytest.raises(AssertionError, match="invalid sample_group_index"):
            executor.load(5)

    def test_a_marker_without_fully_async_state_is_refused(self, tmp_path: Path) -> None:
        """A fully async checkpoint cannot discard its queued and in-flight work."""
        executor = _make_executor(tmp_path, _CountingRolloutFn())
        executor.save(5)
        executor.args.fully_async = True
        marker = compute_checkpoint_complete_marker_path(tmp_path, rollout_id=5)
        marker.write_text(json.dumps(executor._checkpoint_manifest(5, directory=tmp_path), sort_keys=True))

        with pytest.raises(AssertionError, match="fully_async_state_5.pt"):
            executor.load(5)

    def test_a_marker_without_event_snapshot_is_refused(self, tmp_path: Path) -> None:
        """An accounting-enabled checkpoint cannot forget the issued and terminal events."""
        executor = _make_executor(tmp_path, _CountingRolloutFn())
        executor.save(5)
        executor.args.save_debug_event_data = str(tmp_path / "events")
        marker = compute_checkpoint_complete_marker_path(tmp_path, rollout_id=5)
        marker.write_text(json.dumps(executor._checkpoint_manifest(5, directory=tmp_path), sort_keys=True))

        with pytest.raises(AssertionError, match="no mandatory state"):
            executor.load(5)

    def test_a_marker_manifest_must_match_the_restored_mode(self, tmp_path: Path) -> None:
        """The marker records which component set made the checkpoint complete."""
        executor = _make_executor(tmp_path, _CountingRolloutFn())
        executor.save(5)
        marker = compute_checkpoint_complete_marker_path(tmp_path, rollout_id=5)
        manifest = json.loads(marker.read_text())
        manifest["files"] = []
        marker.write_text(json.dumps(manifest))

        with pytest.raises(AssertionError, match="describes"):
            executor.load(5)

    def test_an_empty_optional_load_warns_and_continues(self, tmp_path: Path) -> None:
        """A fresh run with no trainer checkpoint may start without rollout state."""
        executor = _make_executor(tmp_path, _CountingRolloutFn())

        executor.load(5)

        assert executor.data_source.loaded == [5]
