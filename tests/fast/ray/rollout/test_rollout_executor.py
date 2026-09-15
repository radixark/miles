import asyncio
from argparse import Namespace
from pathlib import Path

import pytest
import torch
from tests.fast.ray.rollout.conftest import make_args

from miles.ray.rollout import rollout_executor as rollout_executor_module
from miles.ray.rollout.eval_fleet import EvalFleetInfo, EvalFleetPin
from miles.ray.rollout.rollout_executor import RolloutExecutor
from miles.rollout.base_types import RolloutFnEvalInput, RolloutFnEvalOutput, RolloutFnTrainOutput
from miles.rollout.data_source import RolloutDataSource
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
        executor.args = Namespace()
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


def _make_executor(tmp_path: Path, rollout_fn: _CountingRolloutFn) -> RolloutExecutor:
    executor = RolloutExecutor.__new__(RolloutExecutor)
    executor.args = make_args(load=str(tmp_path), save=str(tmp_path))
    executor.use_legacy_rollout_v1 = True
    executor.generate_rollout = rollout_fn
    executor.eval_generate_rollout = rollout_fn
    executor.data_source = _FakeDataSource(tmp_path)
    executor._train_parallel_configs_of_model_id = {None: {}}
    executor._weight_versions_of_model_id = {}
    return executor


class TestOneDirectoryPerRolloutCheckpoint:
    def test_a_step_that_was_never_trained_is_refused(self, tmp_path: Path) -> None:
        """A run whose trainer starts from scratch has no rollout state, and must not be asked for any."""
        executor = _make_executor(tmp_path, _CountingRolloutFn())

        with pytest.raises(AssertionError, match="is not a trained step"):
            executor.load(-1)

        assert executor.data_source.loaded == []
