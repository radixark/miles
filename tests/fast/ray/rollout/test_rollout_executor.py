import asyncio
from argparse import Namespace
from pathlib import Path

import pytest

from miles.ray.rollout import rollout_executor as rollout_executor_module
from miles.ray.rollout.eval_fleet import EvalFleetInfo, EvalFleetPin
from miles.ray.rollout.rollout_executor import RolloutExecutor
from miles.rollout.base_types import RolloutFnEvalInput, RolloutFnEvalOutput, RolloutFnTrainOutput
from miles.rollout.inference_rollout import inference_rollout_common
from miles.rollout.inference_rollout.inference_rollout_common import GenerateState
from miles.utils.audit_utils.event_logger.logger import read_events
from miles.utils.audit_utils.event_logger.models import SampleOwner, SampleOwnerTransitionEvent
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
