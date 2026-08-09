"""Composed ownership path for issue #2254.

The test keeps the production seams intact while replacing only external
inference, object-store, trainer, and checkpoint side effects. Two sequential
reservations cover successful training and checkpointed replay before the same
scenario crosses shared evaluation and weight-update fencing.
"""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="stage-a-cpu", labels=[])

import asyncio
import threading
from argparse import Namespace
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

import miles.ray.rollout.rollout_manager as rollout_manager_mod
import miles.rollout.data_source as data_source_mod
import miles.rollout.fully_async_rollout as fully_async_mod
import miles.rollout.inference_rollout.fully_async as inference_fully_async_mod
from miles.ray.rollout.rollout_manager import RolloutManager
from miles.ray.train_batch_admission import (
    RayTrainerAdmissionAdapter,
    TrainerAdmissionReceipt,
    TrainerAdmissionStatus,
    TrainerCellCohort,
    TrainerCohort,
    validate_publication_data_ref,
)
from miles.ray.train_batch_coordinator import TrainBatchCoordinator
from miles.rollout.base_types import RolloutFnConstructorInput
from miles.rollout.data_source import RolloutDataSource, SourceReservationId
from miles.rollout.fully_async_rollout import FullyAsyncRolloutFn
from miles.utils.async_utils import get_async_loop
from miles.utils.ray_utils import Box
from miles.utils.types import Sample


class _FakeGenerateState:
    def __init__(self, args: Namespace) -> None:
        self.args = args
        self.sampling_params: dict[str, Any] = {}
        self.aborted = False


class _RemoteMethod:
    def __init__(self, callback) -> None:
        self._callback = callback

    def remote(self, *args, **kwargs):
        async def invoke():
            return self._callback(*args, **kwargs)

        return invoke()


class _ManagerProxy:
    def __init__(self, manager: RolloutManager) -> None:
        self.commit_trainer_admission = _RemoteMethod(manager.commit_trainer_admission)
        self.rollback_trainer_admission = _RemoteMethod(manager.rollback_trainer_admission)
        self.get_trainer_admission_status = _RemoteMethod(manager.get_trainer_admission_status)


class _RecordingTrainer:
    def __init__(self, manager: RolloutManager) -> None:
        self._manager = manager
        self.events: list[str] = []
        self.discarded: list[TrainerAdmissionReceipt] = []

    async def admit_train_batch(self, rollout_id: int, data_pack: dict[str, Any]) -> TrainerAdmissionReceipt:
        self.events.append(f"admit:{rollout_id}")
        publication = data_pack["trainer_admission"]
        validate_publication_data_ref(publication, data_pack["data_ref"])
        return TrainerAdmissionReceipt(
            publication=publication,
            role="actor",
            cohort=TrainerCohort(
                quorum_id=None,
                cells=(TrainerCellCohort(cell_index=0, ranks=(0,)),),
            ),
        )

    async def train(self, rollout_id: int, data_pack: dict[str, Any], **kwargs) -> None:
        publication = data_pack["trainer_admission"]
        status = self._manager.get_trainer_admission_status(publication)
        self.events.append(f"train:{rollout_id}:{status.value}")

    def discard_train_batch_admission(self, receipt: TrainerAdmissionReceipt) -> None:
        self.discarded.append(receipt)


def _args(tmp_path: Path) -> Namespace:
    prompt_data = tmp_path / "prompts.jsonl"
    prompt_data.write_text('{"prompt": "alpha"}\n{"prompt": "bravo"}\n{"prompt": "charlie"}\n')
    return Namespace(
        rollout_batch_size=1,
        n_samples_per_prompt=1,
        async_max_concurrent_samples=1,
        async_data_buffer_capacity_factor=1.0,
        async_unused_samples_handler="drop",
        rollout_submission_granularity="group",
        max_weight_staleness=None,
        dynamic_sampling_filter_path=None,
        rollout_sample_filter_path=None,
        custom_async_data_buffer_path=None,
        rollout_health_check_timeout=0.1,
        rollout_global_dataset=True,
        sglang_router_ip="127.0.0.1",
        sglang_router_port=30000,
        eval_num_gpus=0,
        ci_test=False,
        use_fault_tolerance=False,
        ci_inject_rollout_data_path=None,
        load_debug_rollout_data=None,
        delay_split_train_data_by_dp=True,
        use_critic=False,
        num_critic_only_steps=0,
        debug_train_only=False,
        eval_uses_snapshots=False,
        custom_convert_samples_to_train_data_path=None,
        custom_reward_post_process_path=None,
        disable_rollout_trim_samples=True,
        global_batch_size=1,
        use_dynamic_global_batch_size=False,
        advantage_estimator="grpo",
        rewards_normalization=False,
        grpo_std_normalization=False,
        reward_key=None,
        balance_data=False,
        hf_checkpoint="unused",
        chat_template_path=None,
        dump_details=None,
        prompt_data=str(prompt_data),
        rollout_max_prompt_len=None,
        input_key="prompt",
        multimodal_keys=None,
        label_key=None,
        metadata_key="metadata",
        tool_key=None,
        apply_chat_template=False,
        apply_chat_template_kwargs=None,
        rollout_seed=100,
        rollout_shuffle=False,
        save=str(tmp_path),
        load=str(tmp_path),
        save_interval=1,
        save_trigger_sentinel=None,
        buffer_filter_path=None,
    )


def _make_manager(
    args: Namespace,
    data_source: RolloutDataSource,
    lifecycle: FullyAsyncRolloutFn,
    monkeypatch: pytest.MonkeyPatch,
) -> RolloutManager:
    manager_class = cast(type[RolloutManager], object.__getattribute__(RolloutManager, "__ray_actor_class__"))
    manager = object.__new__(manager_class)
    manager.args = args
    manager.weight_version = 1
    manager.rollout_id = -1
    manager.servers = {}
    manager.data_source = data_source
    manager.train_parallel_config = {"dp_size": 1}
    manager.custom_convert_samples_to_train_data_func = None
    manager.custom_reward_post_process_func = None
    manager.use_legacy_rollout_v1 = False
    manager.generate_rollout = lifecycle
    manager.eval_generate_rollout = object()
    manager._train_rollout_lifecycle = lifecycle
    manager._rollout_lifecycles = (lifecycle,)
    manager._lifecycle_async_loop = get_async_loop()
    manager._closed_rollout_lifecycles = []
    manager._rollout_lifecycles_closing = False
    manager._dispose_lock = asyncio.Lock()
    manager._manager_resources_disposed = False
    manager._next_train_admission_hold_id = 0
    manager._train_admission_holds = {}
    manager._weight_update_fence_hold_id = None
    manager._weight_update_fence_failure = None
    manager._weight_update_fence_open = asyncio.Event()
    manager._weight_update_fence_open.set()
    manager._shared_eval_admission_open = asyncio.Event()
    manager._shared_eval_admission_open.set()
    manager._active_shared_eval_holds = set()
    manager._shared_evals_drained = asyncio.Event()
    manager._shared_evals_drained.set()
    manager._manager_incarnation = "vertical-slice-manager"
    manager._next_admission_id = 0
    manager._pending_admissions = {}
    manager._data_source_closed = True
    manager._event_analysis_completed = True
    manager._metric_checker_disposed = True
    manager._checkpoint_eval_disposed = True
    manager._stopped_health_monitors = []
    manager._health_monitors = []
    manager._metric_checker = None
    manager._active_generations = 0
    manager._generations_drained = asyncio.Event()
    manager._generations_drained.set()

    monkeypatch.setattr(manager, "_health_monitoring_resume", lambda: None)
    monkeypatch.setattr(rollout_manager_mod.dashboard_hooks, "register_engines", lambda servers: None)
    monkeypatch.setattr(rollout_manager_mod, "timer", lambda name: nullcontext())
    monkeypatch.setattr(rollout_manager_mod, "save_debug_rollout_data", lambda *args, **kwargs: None)
    monkeypatch.setattr(rollout_manager_mod, "log_rollout_data", lambda *args, **kwargs: None)
    monkeypatch.setattr(rollout_manager_mod, "log_eval_rollout_data", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        rollout_manager_mod.object_store,
        "get_instance",
        lambda: SimpleNamespace(put=lambda **kwargs: Box("vertical-slice-data"), remove=lambda ref: None),
    )
    monkeypatch.setattr(
        rollout_manager_mod.event_logger_checkpoint,
        "snapshot",
        lambda *args, **kwargs: None,
    )
    return manager


@pytest.mark.asyncio
async def test_fully_async_vertical_slice_preserves_ownership_across_train_checkpoint_and_eval(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Exercise one leased batch through every issue #2254 lifecycle seam."""
    monkeypatch.setattr(data_source_mod, "load_tokenizer", lambda *args, **kwargs: object())
    monkeypatch.setattr(data_source_mod, "load_processor", lambda *args, **kwargs: None)
    args = _args(tmp_path)
    data_source = RolloutDataSource(args)

    async def generate_group(state, samples, sampling_params, evaluation=False, sample_done_callback=None):
        for sample in samples:
            sample.response = "answer"
            sample.response_length = 1
            sample.reward = 1.0
            sample.status = Sample.Status.COMPLETED
        return samples

    monkeypatch.setattr(fully_async_mod, "GenerateState", _FakeGenerateState)
    monkeypatch.setattr(inference_fully_async_mod, "generate_and_rm_group", generate_group)
    lifecycle = FullyAsyncRolloutFn(RolloutFnConstructorInput(args=args, data_source=data_source))
    manager = _make_manager(args, data_source, lifecycle, monkeypatch)

    second_pack = None
    eval_task = None
    update_hold_id = None
    release_eval = threading.Event()
    try:
        first_pack = await manager.generate(rollout_id=7)
        first_publication = first_pack["trainer_admission"]
        assert isinstance(first_pack["data_ref"], Box)

        trainer = _RecordingTrainer(manager)
        coordinator = TrainBatchCoordinator(
            args=args,
            actor_model=trainer,
            critic_model=None,
            rollout_manager=None,
            admission_adapter=RayTrainerAdmissionAdapter(_ManagerProxy(manager)),
        )
        await coordinator.train(rollout_id=7, rollout_data_pack=first_pack)
        assert trainer.events == ["admit:7", "train:7:committed"]
        assert manager.get_trainer_admission_status(first_publication) is TrainerAdmissionStatus.COMMITTED

        second_pack = await manager.generate(rollout_id=8)
        second_publication = second_pack["trainer_admission"]
        checkpoint_path = tmp_path / "rollout" / "global_dataset_state_dict_8.pt"
        with pytest.raises(RuntimeError, match="unresolved trainer admissions"):
            await manager.save(8)

        assert not checkpoint_path.exists()
        assert manager.get_trainer_admission_status(second_publication) is TrainerAdmissionStatus.PENDING
        assert await coordinator.rollback_prefetched(second_pack)
        assert manager.get_trainer_admission_status(second_publication) is TrainerAdmissionStatus.ROLLED_BACK

        await manager.save(8)
        assert checkpoint_path.exists()

        restored = RolloutDataSource(args)
        restored.load(8)
        [replayed] = restored.reserve_samples(1)
        assert replayed.reservation_id == SourceReservationId("1")

        eval_started = threading.Event()

        def blocked_eval(rollout_fn, input):
            eval_started.set()
            assert release_eval.wait(timeout=2)
            return SimpleNamespace(data={}, metrics={})

        monkeypatch.setattr(rollout_manager_mod, "call_rollout_function", blocked_eval)
        eval_task = asyncio.create_task(manager.eval(9))
        assert await asyncio.to_thread(eval_started.wait, 2)

        update_hold_id = await manager.acquire_train_admission_hold()
        update_wait = asyncio.create_task(manager.wait_weight_update_admission(update_hold_id))
        await asyncio.sleep(0)
        assert not update_wait.done(), "weight update must wait for shared evaluation"

        release_eval.set()
        await eval_task
        await update_wait
        manager.weight_version = 2
        await manager.record_train_weight_update(update_hold_id)
        await manager.release_train_admission_hold(update_hold_id)
        update_hold_id = None
    finally:
        release_eval.set()
        if eval_task is not None and not eval_task.done():
            eval_task.cancel()
            await asyncio.gather(eval_task, return_exceptions=True)
        if update_hold_id is not None:
            try:
                await manager.release_train_admission_hold(update_hold_id)
            except BaseException:
                pass
        if second_pack is not None:
            second_publication = second_pack["trainer_admission"]
            if manager.get_trainer_admission_status(second_publication) is TrainerAdmissionStatus.PENDING:
                await coordinator.rollback_prefetched(second_pack)
        await manager.dispose()
