"""Composed ownership path for issue #2254.

The test keeps the production seams intact while replacing only GPU inference,
optimizer, logging, and checkpoint-event side effects. Two sequential
reservations cover successful remote training and checkpointed replay before
the same scenario crosses shared evaluation and weight-update fencing.
"""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="stage-a-cpu", labels=[])

import asyncio
import inspect
import threading
from argparse import Namespace
from collections.abc import Mapping
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import ray
import train_async as train_async_mod

import miles.ray.rollout.rollout_manager as rollout_manager_mod
import miles.rollout.data_source as data_source_mod
import miles.rollout.fully_async.ownership as ownership_mod
import miles.rollout.fully_async_rollout as fully_async_mod
import miles.rollout.inference_rollout.fully_async as inference_fully_async_mod
from miles.ray.rollout.rollout_manager import RolloutManager
from miles.ray.train_batch_admission import (
    RayTrainerAdmissionAdapter,
    TrainerAdmissionReceipt,
    TrainerAdmissionStatus,
    TrainerCellCohort,
    TrainerCohort,
    TrainerRankReceipt,
    validate_publication_data_ref,
)
from miles.ray.train_batch_coordinator import TrainBatchCoordinator
from miles.rollout.base_types import RolloutFnConstructorInput
from miles.rollout.data_source import RolloutDataSource, RolloutDataSourceWithBuffer, SourceReservationId
from miles.rollout.fully_async_rollout import FullyAsyncRolloutFn
from miles.utils import object_store
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
        result = self._callback(*args, **kwargs)
        if inspect.isawaitable(result):
            return result

        async def invoke():
            return result

        return invoke()


class _ManagerProxy:
    def __init__(self, manager: RolloutManager) -> None:
        self._manager = manager
        self.weight_update_wait_started = asyncio.Event()
        self.acquire_train_admission_hold = _RemoteMethod(manager.acquire_train_admission_hold)
        self.wait_weight_update_admission = _RemoteMethod(self._wait_weight_update_admission)
        self.record_train_weight_update = _RemoteMethod(manager.record_train_weight_update)
        self.release_train_admission_hold = _RemoteMethod(manager.release_train_admission_hold)
        self.commit_trainer_admission = _RemoteMethod(manager.commit_trainer_admission)
        self.rollback_trainer_admission = _RemoteMethod(manager.rollback_trainer_admission)
        self.get_trainer_admission_status = _RemoteMethod(manager.get_trainer_admission_status)

    async def _wait_weight_update_admission(self, hold_id: int | None) -> None:
        self.weight_update_wait_started.set()
        await self._manager.wait_weight_update_admission(hold_id)


@ray.remote(num_cpus=0)
class _RemoteTrainerRank:
    """Small real-Ray boundary that proves and consumes the published object ref."""

    def __init__(self) -> None:
        self._store = object_store.init_instance(Namespace(object_store_backend="ray"))
        self._events: list[tuple[str, int, int, str | None]] = []

    def _resolve_publication(self, publication, data_ref) -> int:
        validate_publication_data_ref(publication, data_ref)
        refs = data_ref if isinstance(data_ref, list) else [data_ref]
        resolved = 0
        for ref in refs:
            with self._store.get(ref) as value:
                if not isinstance(value, Mapping):
                    raise ValueError(f"Admission {publication.admission_id} resolved non-mapping data.")
            resolved += 1
        return resolved

    def admit_train_batch(self, publication, data_ref) -> TrainerRankReceipt:
        resolved_refs = self._resolve_publication(publication, data_ref)
        self._events.append(("admit", publication.admission_id, resolved_refs, None))
        return TrainerRankReceipt(publication=publication, rank=0)

    def train(self, publication, data_ref, manager_status: str) -> dict[str, Any]:
        resolved_refs = self._resolve_publication(publication, data_ref)
        self._events.append(("train", publication.admission_id, resolved_refs, manager_status))
        if manager_status != TrainerAdmissionStatus.COMMITTED.value:
            raise AssertionError(f"remote train started before commit: {manager_status!r}")
        return {"rank": 0, "resolved_refs": resolved_refs, "status": manager_status}

    def train_legacy(self, rollout_id: int, data_ref) -> dict[str, Any]:
        refs = data_ref if isinstance(data_ref, list) else [data_ref]
        resolved = 0
        for ref in refs:
            with self._store.get(ref) as value:
                if not isinstance(value, Mapping):
                    raise ValueError(f"Legacy train data for rollout {rollout_id} is not a mapping.")
            resolved += 1
        self._events.append(("train", rollout_id, resolved, "legacy"))
        return {"rank": 0, "resolved_refs": resolved, "rollout_id": rollout_id}

    def update_weights(self, rollout_id: int | None = None) -> dict[str, Any]:
        self._events.append(("update_weights", rollout_id if rollout_id is not None else -1, 0, "2"))
        return {"rollout_id": rollout_id, "weight_version": 2}

    def events(self) -> tuple[tuple[str, int, int, str | None], ...]:
        return tuple(self._events)


class _RayTrainerGroupAdapter:
    """Local trainer-group seam whose admission and training cross real Ray."""

    def __init__(self, manager: RolloutManager, rank: ray.actor.ActorHandle) -> None:
        self._manager = manager
        self._rank = rank
        self.events: list[str] = []
        self.discarded: list[TrainerAdmissionReceipt] = []

    async def admit_train_batch(self, rollout_id: int, data_pack: dict[str, Any]) -> TrainerAdmissionReceipt:
        self.events.append(f"admit:{rollout_id}")
        publication = data_pack["trainer_admission"]
        rank_receipt = await self._rank.admit_train_batch.remote(publication, data_pack["data_ref"])
        if not isinstance(rank_receipt, TrainerRankReceipt) or rank_receipt.publication != publication:
            raise AssertionError(f"unexpected remote trainer proof: {rank_receipt!r}")
        return TrainerAdmissionReceipt(
            publication=publication,
            role="actor",
            cohort=TrainerCohort(
                quorum_id=None,
                cells=(TrainerCellCohort(cell_index=0, ranks=(0,)),),
            ),
        )

    async def train(self, rollout_id: int, data_pack: dict[str, Any], **kwargs) -> None:
        if "trainer_admission" not in data_pack:
            # Legacy batches carry no manager-owned admission, so the rank reads
            # the published reference without a publication to prove against.
            self.events.append(f"train:{rollout_id}:legacy")
            result = await self._rank.train_legacy.remote(rollout_id, data_pack["data_ref"])
            assert result == {"rank": 0, "resolved_refs": 1, "rollout_id": rollout_id}
            return
        publication = data_pack["trainer_admission"]
        status = self._manager.get_trainer_admission_status(publication)
        self.events.append(f"train:{rollout_id}:{status.value}")
        result = await self._rank.train.remote(publication, data_pack["data_ref"], status.value)
        assert result == {
            "rank": 0,
            "resolved_refs": 1,
            "status": TrainerAdmissionStatus.COMMITTED.value,
        }

    def discard_train_batch_admission(self, receipt: TrainerAdmissionReceipt) -> None:
        self.discarded.append(receipt)


class _RayActorModel:
    """Trainer-model seam that publishes the remote update version to the manager."""

    def __init__(self, manager: RolloutManager, rank: ray.actor.ActorHandle) -> None:
        self._manager = manager
        self._rank = rank

    async def update_weights(self, rollout_id: int | None = None) -> None:
        result = await self._rank.update_weights.remote(rollout_id)
        assert result["rollout_id"] == rollout_id
        self._manager.weight_version = result["weight_version"]


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
        object_store_backend="ray",
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
        rollout_manager_mod.event_logger_checkpoint,
        "snapshot",
        lambda *args, **kwargs: None,
    )
    return manager


@pytest.mark.asyncio
async def test_fully_async_vertical_slice_preserves_ownership_across_train_checkpoint_and_eval(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    ray_local_mode,
) -> None:
    """Exercise one leased batch through every issue #2254 lifecycle seam."""
    _ = ray_local_mode
    monkeypatch.setattr(data_source_mod, "load_tokenizer", lambda *args, **kwargs: object())
    monkeypatch.setattr(data_source_mod, "load_processor", lambda *args, **kwargs: None)
    args = _args(tmp_path)
    object_store.init_instance(args)
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
    update_task = None
    trainer_rank = _RemoteTrainerRank.remote()
    release_eval = threading.Event()
    try:
        first_pack = await manager.generate(rollout_id=7)
        first_publication = first_pack["trainer_admission"]
        assert isinstance(first_pack["data_ref"], Box)
        assert isinstance(first_pack["data_ref"].inner, ray.ObjectRef)

        trainer = _RayTrainerGroupAdapter(manager, trainer_rank)
        manager_proxy = _ManagerProxy(manager)
        coordinator = TrainBatchCoordinator(
            args=args,
            actor_model=trainer,
            critic_model=None,
            rollout_manager=None,
            admission_adapter=RayTrainerAdmissionAdapter(manager_proxy),
        )
        await coordinator.train(rollout_id=7, rollout_data_pack=first_pack)
        assert trainer.events == ["admit:7", "train:7:committed"]
        assert await trainer_rank.events.remote() == (
            ("admit", 0, 1, None),
            ("train", 0, 1, TrainerAdmissionStatus.COMMITTED.value),
        )
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

        update_task = asyncio.create_task(
            train_async_mod._update_weights_with_admission_hold(
                manager_proxy,
                _RayActorModel(manager, trainer_rank),
                9,
            )
        )
        await asyncio.wait_for(manager_proxy.weight_update_wait_started.wait(), timeout=2)
        assert not update_task.done(), "weight update must wait for shared evaluation"
        assert [event[0] for event in await trainer_rank.events.remote()] == ["admit", "train"]

        release_eval.set()
        await eval_task
        await update_task
        assert manager.weight_version == 2
        assert (await trainer_rank.events.remote())[-1] == ("update_weights", 9, 0, "2")
    finally:
        release_eval.set()
        if eval_task is not None and not eval_task.done():
            eval_task.cancel()
            await asyncio.gather(eval_task, return_exceptions=True)
        if update_task is not None and not update_task.done():
            update_task.cancel()
            await asyncio.gather(update_task, return_exceptions=True)
        if second_pack is not None:
            second_publication = second_pack["trainer_admission"]
            if manager.get_trainer_admission_status(second_publication) is TrainerAdmissionStatus.PENDING:
                await coordinator.rollback_prefetched(second_pack)
        try:
            await manager.dispose()
        finally:
            try:
                ray.kill(trainer_rank)
            except BaseException:
                pass


@pytest.mark.asyncio
async def test_fully_async_vertical_slice_runs_the_default_buffered_data_source(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    ray_local_mode,
) -> None:
    """The ``--data-source-path`` default must still produce a train batch.

    ``RolloutDataSourceWithBuffer`` cannot hand out durable source reservations,
    so a fully-async run on the shipped default has to fall back to non-owned
    scheduling instead of dying on its first prompt group.
    """
    _ = ray_local_mode
    monkeypatch.setattr(data_source_mod, "load_tokenizer", lambda *args, **kwargs: object())
    monkeypatch.setattr(data_source_mod, "load_processor", lambda *args, **kwargs: None)
    args = _args(tmp_path)
    object_store.init_instance(args)
    data_source = RolloutDataSourceWithBuffer(args)

    async def generate_group(state, samples, sampling_params, evaluation=False, sample_done_callback=None):
        for sample in samples:
            sample.response = "answer"
            sample.response_length = 1
            sample.reward = 1.0
            sample.status = Sample.Status.COMPLETED
        return samples

    monkeypatch.setattr(fully_async_mod, "GenerateState", _FakeGenerateState)
    monkeypatch.setattr(fully_async_mod, "generate_and_rm_group", generate_group)
    monkeypatch.setattr(inference_fully_async_mod, "generate_and_rm_group", generate_group)
    lifecycle = FullyAsyncRolloutFn(RolloutFnConstructorInput(args=args, data_source=data_source))
    manager = _make_manager(args, data_source, lifecycle, monkeypatch)

    trainer_rank = _RemoteTrainerRank.remote()
    try:
        pack = await manager.generate(rollout_id=7)
        assert isinstance(pack["data_ref"], Box)
        assert isinstance(pack["data_ref"].inner, ray.ObjectRef)
        # A source without reservations cannot lease a train batch, so the
        # manager publishes no trainer admission for it.
        assert "trainer_admission" not in pack

        with object_store.get_instance().get(pack["data_ref"]) as train_data:
            assert list(train_data["response_lengths"]) == [1]
            assert list(train_data["raw_reward"]) == [1.0]

        trainer = _RayTrainerGroupAdapter(manager, trainer_rank)
        coordinator = TrainBatchCoordinator(
            args=args,
            actor_model=trainer,
            critic_model=None,
            rollout_manager=None,
            admission_adapter=RayTrainerAdmissionAdapter(_ManagerProxy(manager)),
        )
        await coordinator.train(rollout_id=7, rollout_data_pack=pack)

        assert trainer.events == ["train:7:legacy"]
        assert await trainer_rank.events.remote() == (("train", 7, 1, "legacy"),)
    finally:
        try:
            await manager.dispose()
        finally:
            try:
                ray.kill(trainer_rank)
            except BaseException:
                pass


@pytest.mark.asyncio
async def test_fully_async_vertical_slice_disposes_with_in_flight_owned_execution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    ray_local_mode,
) -> None:
    """Disposing while a group is still generating settles that group once.

    Shutdown cancels in-flight inference, but cancellation is shielded until
    the execution reaches terminal state, so the observer records the terminal
    receipt itself. Close must reuse that receipt and return both reservations
    for pristine replay instead of recording the receipt a second time.
    """
    _ = ray_local_mode
    monkeypatch.setattr(data_source_mod, "load_tokenizer", lambda *args, **kwargs: object())
    monkeypatch.setattr(data_source_mod, "load_processor", lambda *args, **kwargs: None)
    args = _args(tmp_path)
    object_store.init_instance(args)
    data_source = RolloutDataSource(args)

    generation_calls = 0
    in_flight_started = asyncio.Event()

    async def generate_group(state, samples, sampling_params, evaluation=False, sample_done_callback=None):
        nonlocal generation_calls
        generation_calls += 1
        if generation_calls > 1:
            in_flight_started.set()
            while not state.aborted:
                await asyncio.sleep(0.01)
        for sample in samples:
            sample.response = "answer"
            sample.response_length = 1
            sample.reward = 1.0
            sample.status = Sample.Status.COMPLETED
        return samples

    async def request_abort(args) -> None:
        return None

    recorded_terminals: list[tuple[SourceReservationId, int]] = []
    record_terminal = ownership_mod.ReservationOwnership.record_terminal

    def counting_record_terminal(self, receipts, *, stage_id):
        terminal_receipts = record_terminal(self, receipts, stage_id=stage_id)
        recorded_terminals.extend(
            (receipt.executor_receipt.reservation_id, receipt.executor_receipt.receipt_id)
            for receipt in terminal_receipts
        )
        return terminal_receipts

    monkeypatch.setattr(fully_async_mod, "GenerateState", _FakeGenerateState)
    monkeypatch.setattr(inference_fully_async_mod, "generate_and_rm_group", generate_group)
    monkeypatch.setattr(inference_fully_async_mod, "request_abort", request_abort)
    monkeypatch.setattr(ownership_mod.ReservationOwnership, "record_terminal", counting_record_terminal)
    lifecycle = FullyAsyncRolloutFn(RolloutFnConstructorInput(args=args, data_source=data_source))
    manager = _make_manager(args, data_source, lifecycle, monkeypatch)

    trainer_rank = _RemoteTrainerRank.remote()
    try:
        first_pack = await manager.generate(rollout_id=7)
        coordinator = TrainBatchCoordinator(
            args=args,
            actor_model=_RayTrainerGroupAdapter(manager, trainer_rank),
            critic_model=None,
            rollout_manager=None,
            admission_adapter=RayTrainerAdmissionAdapter(_ManagerProxy(manager)),
        )
        assert await coordinator.rollback_prefetched(first_pack)
        await asyncio.wait_for(in_flight_started.wait(), timeout=2)

        await manager.dispose()

        assert lifecycle._active_executions == {}
        # Two executions run: rollout 7's group, and the replay of that same group
        # after rollback, which is the one still in flight at dispose. Each records
        # its terminal receipt exactly once, so close reuses the observer's receipt
        # rather than recording a second one for the in-flight attempt.
        assert [reservation_id for reservation_id, _ in recorded_terminals] == [SourceReservationId("0")] * 2
        assert len({receipt_id for _, receipt_id in recorded_terminals}) == 2

        replayed = {reservation.reservation_id for reservation in data_source.reserve_samples(2)}
        assert replayed == {SourceReservationId("0"), SourceReservationId("1")}
    finally:
        try:
            ray.kill(trainer_rank)
        except BaseException:
            pass
