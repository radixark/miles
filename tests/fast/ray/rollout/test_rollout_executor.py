import asyncio
from argparse import Namespace
from pathlib import Path

import pytest
import torch
from tests.fast.ray.rollout.conftest import UnevenLegacyRolloutFn, make_args

from miles.backends.megatron_utils.cpu_witness import record_optimizer_step
from miles.ray.rollout import rollout_executor as rollout_executor_module
from miles.ray.rollout.eval_fleet import EvalFleetInfo, EvalFleetPin
from miles.ray.rollout.rollout_data_conversion import postprocess_rollout_data
from miles.ray.rollout.rollout_executor import (
    RolloutExecutor,
    compute_checkpoint_complete_marker_path,
    compute_executor_state_path,
)
from miles.rollout.base_types import RolloutFnEvalInput, RolloutFnEvalOutput, RolloutFnTrainOutput
from miles.rollout.inference_rollout import inference_rollout_common
from miles.rollout.inference_rollout.inference_rollout_common import GenerateState
from miles.utils.audit_utils import sample_ownership
from miles.utils.audit_utils.event_analyzer.rules.sample_ownership import check
from miles.utils.audit_utils.event_logger.logger import read_events
from miles.utils.audit_utils.event_logger.models import (
    RolloutStateRestoreEvent,
    SampleOwner,
    SampleOwnerTransitionEvent,
)
from miles.utils.audit_utils.witness.cpu import CpuWitness
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
        executor._train_parallel_configs_of_model_id = {}
        executor._last_batches = {}
        executor.rollout_id = -1
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


class FakeDataSource:
    def __init__(self) -> None:
        self.saved: list[int] = []
        self.loaded: list[int | None] = []

    def save(self, rollout_id: int) -> None:
        self.saved.append(rollout_id)

    def load(self, rollout_id: int | None = None) -> None:
        self.loaded.append(rollout_id)


class CountingLegacyRolloutFn:
    def __init__(self, start_index: int = 0) -> None:
        self.next_index = start_index
        self.num_calls = 0

    def __call__(self, args, rollout_id, data_source, evaluation):
        self.num_calls += 1
        self.next_index += 1
        sample = Sample(index=self.next_index, group_index=self.next_index, prompt="p", status=Sample.Status.COMPLETED)
        return RolloutFnTrainOutput(samples=[[sample]])


class TestTrimmedSampleOwnership:
    @pytest.mark.parametrize("dynamic", [False, True])
    async def test_trimmed_tail_is_dropped_on_generation_and_checkpoint_replay(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ownership_event_dir: Path, dynamic: bool
    ) -> None:
        """Fixed and dynamic batch trimming must account for the same raw tail after resume."""
        monkeypatch.setattr(rollout_executor_module, "postprocess_rollout_data", postprocess_rollout_data)
        rollout_fn = UnevenLegacyRolloutFn()
        executor = make_executor(tmp_path, rollout_fn=rollout_fn)
        executor.args = make_args(
            save=str(tmp_path), load=str(tmp_path), global_batch_size=2, use_dynamic_global_batch_size=dynamic
        )
        executor._train_parallel_configs_of_model_id[None] = {"dp_size": 2}

        data, metadata, _ = await executor._get_rollout_data(rollout_id=1)
        assert [sample.index for sample in data] == [0, 1, 2, 3]
        if dynamic:
            assert metadata["dynamic_global_batch_size"] == 4
        assert prompt_indices(executor._last_batches[None].samples) == [0, 1, 2, 3, 4]
        executor.save(0)

        resumed = make_executor(tmp_path, rollout_fn=rollout_fn)
        resumed.args = executor.args
        resumed._train_parallel_configs_of_model_id = executor._train_parallel_configs_of_model_id
        resumed.load(0)
        replayed, _, _ = await resumed._get_rollout_data(rollout_id=1)

        assert [sample.index for sample in replayed] == [0, 1, 2, 3]
        assert rollout_fn.num_calls == 1
        assert prompt_indices(resumed._last_batches[None].samples) == [0, 1, 2, 3, 4]
        model = torch.nn.Module()
        model.add_module("cpu_witness", CpuWitness(pipeline_rank=0, chunk_index=0, replica_id=(0,)))
        record_optimizer_step(
            args=Namespace(trainer_model_id=None),
            model=[model],
            rows=[[index, index, -1] for index in range(4)],
            rollout_data={"ownership_lineage_id": resumed._lineage_id},
            rollout_id=1,
            step_id=0,
            attempt=0,
        )
        sample_ownership.log_holdings_snapshot(
            rollout_id=1, trainer_model_id=None, holdings={}, replays_samples=False, reason="final"
        )
        events = read_events(ownership_event_dir)
        drops = [event for event in events if isinstance(event, SampleOwnerTransitionEvent) and event.reason == "trim"]
        assert len(drops) == 2
        assert all(event.sample_indices == [4] for event in drops)
        assert all(
            event.from_owner == SampleOwner.HANDED_TO_TRAINER and event.to_owner == SampleOwner.DROPPED
            for event in drops
        )
        assert all(event.rollout_id == 1 and event.trainer_model_id is None for event in drops)
        assert check(events) == []
        assert check([*events, drops[-1]]) == []


class TestDeliveryOwnership:
    async def test_generated_delivery_and_trim_have_distinct_ownership_events(
        self, tmp_path: Path, ownership_event_dir: Path
    ) -> None:
        """Delivered samples and postprocessing drops remain individually attributable."""
        executor = make_executor(tmp_path, rollout_fn=CountingLegacyRolloutFn())
        data, _, _ = await executor._get_rollout_data(0)
        samples = [sample for group in data for sample in group]
        executor._log_trimmed_samples(data=samples[:-1], rollout_id=0, trainer_model_id=None)

        transitions = [
            event for event in read_events(ownership_event_dir) if isinstance(event, SampleOwnerTransitionEvent)
        ]

        assert transitions[0].to_owner is SampleOwner.HANDED_TO_TRAINER
        assert transitions[0].sample_indices == [sample.index for sample in samples]
        assert transitions[-1].to_owner is SampleOwner.DROPPED
        assert transitions[-1].sample_indices == [samples[-1].index]
        assert transitions[-1].reason == "trim"


def make_executor(tmp_path, rollout_fn, *, data_source=None) -> RolloutExecutor:
    executor = RolloutExecutor.__new__(RolloutExecutor)
    executor.args = Namespace(
        save=str(tmp_path),
        load=str(tmp_path),
        load_debug_rollout_data=None,
        ci_inject_rollout_data_path=None,
        save_debug_event_data=None,
        train_backend="megatron",
    )
    executor.use_legacy_rollout_v1 = True
    executor.generate_rollout = rollout_fn
    executor.eval_generate_rollout = rollout_fn
    executor.data_source = data_source if data_source is not None else FakeDataSource()
    executor._train_parallel_configs_of_model_id = {None: {}, "solver": {}, "verifier": {}}
    executor._weight_versions_of_model_id = {}
    executor._last_batches = {}
    executor._replay = {}
    executor._lineage_id = "initial"
    executor.rollout_id = -1
    return executor


@pytest.fixture(autouse=True)
def _stub_rollout_data_postprocessing(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(rollout_executor_module, "postprocess_rollout_data", lambda args, data, **kwargs: (data, {}))
    monkeypatch.setattr(rollout_executor_module, "assert_samples_weight_version_sane", lambda args, samples: None)
    monkeypatch.setattr(
        rollout_executor_module.RolloutDataInjectionUtil, "should_inject", staticmethod(lambda args, rollout_id: False)
    )
    monkeypatch.setattr(rollout_executor_module.event_logger_checkpoint, "snapshot", lambda args, rollout_id: None)


def prompt_indices(data) -> list[int]:
    return [sample.index for group in data for sample in group]


async def _resumed_executor(tmp_path: Path) -> RolloutExecutor:
    executor = make_executor(tmp_path, rollout_fn=CountingLegacyRolloutFn())
    await executor._get_rollout_data(1)
    executor.save(0)
    resumed = make_executor(tmp_path, rollout_fn=CountingLegacyRolloutFn())
    resumed.load(0)
    return resumed


class TestLastBatchReplay:
    def test_each_actual_restore_forks_the_saved_lineage(self, tmp_path: Path, ownership_event_dir: Path) -> None:
        """Executor checkpoints persist lineage and only actual state restoration forks it."""
        executor = make_executor(tmp_path, rollout_fn=CountingLegacyRolloutFn())
        executor.save(0)
        resumed = make_executor(tmp_path, rollout_fn=CountingLegacyRolloutFn())
        resumed.load(0, rollout_ids={"solver": 0, "verifier": 2})
        resumed.save(1)
        restored = make_executor(tmp_path, rollout_fn=CountingLegacyRolloutFn())
        restored.load(1)

        first, second = [e for e in read_events(ownership_event_dir) if isinstance(e, RolloutStateRestoreEvent)]
        assert first.parent_lineage_id == executor._lineage_id
        assert first.lineage_id == resumed._lineage_id
        assert first.rollout_ids == {"solver": 0, "verifier": 2}
        assert second.parent_lineage_id == resumed._lineage_id
        assert second.lineage_id == restored._lineage_id
        assert len({executor._lineage_id, resumed._lineage_id, restored._lineage_id}) == 3

    async def test_invalid_prefetched_weights_are_rejected_again_after_restore(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A checkpoint cannot turn a rejected prefetch into an accepted replay."""

        def reject_weights(args: Namespace, *, samples: list[list[Sample]]) -> None:
            raise AssertionError("invalid sample weight version")

        monkeypatch.setattr(rollout_executor_module, "assert_samples_weight_version_sane", reject_weights)
        executor = make_executor(tmp_path, rollout_fn=CountingLegacyRolloutFn())
        with pytest.raises(AssertionError, match="invalid sample weight version"):
            await executor._get_rollout_data(1)
        executor.save(0)
        resumed = make_executor(tmp_path, rollout_fn=CountingLegacyRolloutFn())
        resumed.load(0)

        with pytest.raises(AssertionError, match="invalid sample weight version"):
            await resumed._get_rollout_data(1)

    async def test_downstream_metadata_mutation_does_not_change_the_saved_batch(self, tmp_path: Path) -> None:
        """Postprocessing a handed batch must preserve the metadata needed to replay it."""
        executor = make_executor(tmp_path, rollout_fn=CountingLegacyRolloutFn())
        batch = [[Sample(index=1, metadata={"step_slots": [3]})]]
        executor._record_last_batch(rollout_id=1, trainer_model_id=None, samples=batch)
        assert batch[0][0].metadata.pop("step_slots") == [3]

        executor.save(0)

        state = torch.load(compute_executor_state_path(tmp_path, rollout_id=0), weights_only=False)
        assert state["last_batches"][None].samples[0][0].metadata == {"step_slots": [3]}

    async def test_a_batch_taken_out_but_not_trained_comes_back_after_a_restart(self, tmp_path) -> None:
        """train_async prefetches a step ahead, so a crash between get(r+1) and train(r+1) used to lose it."""
        rollout_fn = CountingLegacyRolloutFn()
        executor = make_executor(tmp_path, rollout_fn)
        before, _, _ = await executor._get_rollout_data(1)
        executor.save(0)

        resumed = make_executor(tmp_path, CountingLegacyRolloutFn(start_index=rollout_fn.next_index))
        resumed.load(0)
        after, _, _ = await resumed._get_rollout_data(1)

        assert prompt_indices(after) == prompt_indices(before)

    async def test_a_replayed_batch_does_not_go_through_the_rollout_function_again(self, tmp_path) -> None:
        """Regenerating it would consume fresh prompts and leave the recorded ones unowned."""
        resumed = await _resumed_executor(tmp_path)
        rollout_fn = resumed.generate_rollout
        await resumed._get_rollout_data(1)

        assert rollout_fn.num_calls == 0

    async def test_a_replay_is_consumed_once(self, tmp_path) -> None:
        """It stands in for one step only; a second step must generate as usual."""
        resumed = await _resumed_executor(tmp_path)
        rollout_fn = resumed.generate_rollout
        await resumed._get_rollout_data(1)
        await resumed._get_rollout_data(2)

        assert rollout_fn.num_calls == 1

    async def test_a_different_rollout_id_preserves_the_pending_replay(self, tmp_path: Path) -> None:
        """An unrelated get generates normally without consuming the recorded batch."""
        resumed = await _resumed_executor(tmp_path)
        await resumed._get_rollout_data(2)
        assert resumed.generate_rollout.num_calls == 1

        await resumed._get_rollout_data(1)
        assert resumed.generate_rollout.num_calls == 1

    async def test_a_batch_the_checkpoint_already_covers_is_not_persisted(self, tmp_path) -> None:
        """get(k) precedes train(k) and save(r) follows train(r), so anything up to r is in the weights."""
        executor = make_executor(tmp_path, CountingLegacyRolloutFn())
        await executor._get_rollout_data(3)

        executor.save(3)

        state = torch.load(compute_executor_state_path(tmp_path, rollout_id=3), weights_only=False)
        assert state["last_batches"] == {}

    async def test_every_policy_is_measured_against_its_own_rollout_id(self, tmp_path) -> None:
        """Each policy of a multi policy run counts its own rollouts, and one shared bound would drop batches."""
        executor = make_executor(tmp_path, CountingLegacyRolloutFn())
        await executor._get_rollout_data(4, trainer_model_id="solver")
        await executor._get_rollout_data(9, trainer_model_id="verifier")

        executor.save(4, rollout_ids={"solver": 4, "verifier": 8})

        state = torch.load(compute_executor_state_path(tmp_path, rollout_id=4), weights_only=False)
        assert list(state["last_batches"]) == ["verifier"]


class TestCheckpointCompleteMarker:
    def test_overwriting_a_checkpoint_removes_the_marker_before_a_failed_save(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A failed overwrite must not leave the previous completeness marker valid."""
        executor = make_executor(tmp_path, rollout_fn=CountingLegacyRolloutFn())
        executor.save(2)
        marker = compute_checkpoint_complete_marker_path(tmp_path, rollout_id=2)
        assert marker.is_file()

        async def fail_save(rollout_id: int, *, rollout_ids: dict[str, int] | None) -> None:
            assert not marker.exists()
            raise RuntimeError("save interrupted")

        monkeypatch.setattr(executor, "_save_sample_state", fail_save)
        with pytest.raises(RuntimeError, match="save interrupted"):
            executor.save(2)

        assert not marker.exists()
        with pytest.raises(AssertionError, match="no complete_2 marker"):
            executor.load(2, require_complete=True)

    def test_a_trainer_checkpoint_requires_a_marker_even_without_rollout_files(self, tmp_path: Path) -> None:
        """A crash before the first rollout state write must not restart a restored trainer with empty state."""
        executor = make_executor(tmp_path, rollout_fn=CountingLegacyRolloutFn())

        with pytest.raises(AssertionError, match=str(tmp_path / "rollout")):
            executor.load(5, require_complete=True)

    @pytest.mark.parametrize("require_complete", [False, True])
    def test_a_rollout_state_without_the_marker_is_refused(self, tmp_path: Path, require_complete: bool) -> None:
        """A run that died mid-save leaves the trainer and the rollout side at different points."""
        directory = tmp_path / "rollout"
        directory.mkdir()
        (directory / "global_dataset_state_dict_5.pt").write_text("")

        executor = make_executor(tmp_path, CountingLegacyRolloutFn())

        with pytest.raises(AssertionError, match="no complete_5 marker"):
            executor.load(5, require_complete=require_complete)

    def test_an_empty_load_directory_only_warns(self, tmp_path) -> None:
        """A hot restart before the first checkpoint has nothing to restore and must still start."""
        executor = make_executor(tmp_path, CountingLegacyRolloutFn())

        executor.load(5)

        assert executor.data_source.loaded == [5]

    def test_save_writes_the_marker_last(self, tmp_path) -> None:
        """Its presence is what proves every other rollout state file of that rollout id is complete."""
        executor = make_executor(tmp_path, CountingLegacyRolloutFn())

        executor.save(2)

        assert compute_checkpoint_complete_marker_path(tmp_path, rollout_id=2).exists()

    @pytest.mark.parametrize("require_complete", [False, True])
    def test_a_marked_checkpoint_loads(self, tmp_path: Path, require_complete: bool) -> None:
        """The round trip a resumed run actually takes."""
        executor = make_executor(tmp_path, CountingLegacyRolloutFn())
        executor.save(2)

        resumed = make_executor(tmp_path, CountingLegacyRolloutFn())
        resumed.load(2, require_complete=require_complete)

        assert resumed.data_source.loaded == [2]


class TestOwnershipSafePoints:
    def test_save_reports_lost_samples_with_the_optional_analyzer_off(
        self, tmp_path: Path, lost_sample_event_dir: Path
    ) -> None:
        """Save reports lost samples even without enabling the full analyzer."""
        executor = _executor_with_lost_samples(tmp_path, event_dir=lost_sample_event_dir)

        with pytest.raises(ValueError, match="Event analysis found issues"):
            executor.save(0)

        assert compute_checkpoint_complete_marker_path(tmp_path, rollout_id=0).is_file()

    async def test_dispose_reports_lost_samples_with_the_optional_analyzer_off(
        self, tmp_path: Path, lost_sample_event_dir: Path
    ) -> None:
        """Disposal reports lost samples even without enabling the full analyzer."""
        executor = _executor_with_lost_samples(tmp_path, event_dir=lost_sample_event_dir)

        with pytest.raises(ValueError, match="Event analysis found issues"):
            await executor.dispose()


def _executor_with_lost_samples(tmp_path: Path, *, event_dir: Path) -> RolloutExecutor:
    executor = make_executor(tmp_path, rollout_fn=CountingLegacyRolloutFn())
    executor.args.save_debug_event_data = str(event_dir)
    executor.args.enable_event_analyzer = False
    executor._metric_checker = None
    return executor
