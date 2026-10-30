"""Dedicated eval pool ownership, synchronization, locking and generation admission."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from miles.backends.training_utils.weight_update import receiver_pool as pools_module
from miles.backends.training_utils.weight_update import updater as updater_module
from miles.backends.training_utils.weight_update.conn_status import ConnStatusManager
from miles.backends.training_utils.weight_update.protocols.gpu_delta.protocol import UpdateWeightFromGpuDelta
from miles.ray.rollout.eval_dispatch import EvalDispatcher
from miles.ray.rollout.inference_controller import InferenceController
from miles.ray.train_actor import WeightUpdateOutput
from miles.utils.context_lock import ContextLock


def make_updater(monkeypatch):
    updater = object.__new__(updater_module.WeightUpdater)
    updater.args = SimpleNamespace(update_weight_transfer_mode="gpu-delta")
    updater.protocol = Mock(spec=UpdateWeightFromGpuDelta, supports_receiver_pools=True)
    updater._uses_gpu_delta = True
    updater.conn_status = ConnStatusManager()
    updater.protocol._publication_version = 4
    updater._receiver_pools = updater_module.ReceiverPools()
    updater._hf_weight_iterator = SimpleNamespace()
    updater._registered_adapters = {"rollout-adapter"}
    protocols = []

    def create(_args):
        protocol = Mock(spec=UpdateWeightFromGpuDelta, supports_receiver_pools=True)
        protocol._publication_version = 0
        protocols.append(protocol)
        return protocol

    monkeypatch.setattr(pools_module, "get_weight_transfer_protocol", create)
    return updater, protocols


def test_pool_switch_rebinds_gpu_iterator_and_reuses_committed_state(monkeypatch):
    assert UpdateWeightFromGpuDelta.supports_receiver_pools is True
    updater, created = make_updater(monkeypatch)
    bind_iterator = Mock(wraps=updater._bind_protocol_iterator)
    monkeypatch.setattr(updater, "_bind_protocol_iterator", bind_iterator)
    rollout_protocol = updater.protocol
    with updater.receiver_pool("rollout"):
        pass
    bind_iterator.assert_not_called()
    with updater.receiver_pool("eval"):
        assert updater.protocol._publication_version == 0
        updater.protocol._publication_version = 1
        assert updater._hf_weight_iterator.local_consumer == updater.protocol.send_bucket
        assert updater._hf_weight_iterator.local_error_consumer == updater.protocol.record_export_error
        assert updater._hf_weight_iterator.local_plan_getter == updater.protocol.get_export_plan
        with updater.receiver_pool("eval"):
            assert updater.protocol is created[0]
        bind_iterator.assert_called_once_with()
    assert updater.protocol is rollout_protocol
    assert updater._hf_weight_iterator.local_consumer == rollout_protocol.send_bucket
    assert updater._hf_weight_iterator.local_error_consumer == rollout_protocol.record_export_error
    assert updater._hf_weight_iterator.local_plan_getter == rollout_protocol.get_export_plan
    assert bind_iterator.call_count == 2
    with updater.receiver_pool("eval"):
        assert updater.protocol is created[0]
        assert updater.protocol._publication_version == 1
    assert updater.protocol._publication_version == 4
    assert len(created) == 1


def test_failed_eval_restores_gpu_iterator_without_rolling_back_commit(monkeypatch):
    updater, _ = make_updater(monkeypatch)
    original = updater.protocol
    with pytest.raises(RuntimeError, match="post-commit"):
        with updater.receiver_pool("eval"):
            updater.protocol._publication_version = 2
            raise RuntimeError("post-commit")
    assert updater.protocol is original
    assert updater._hf_weight_iterator.local_consumer == original.send_bucket
    assert updater._hf_weight_iterator.local_error_consumer == original.record_export_error
    assert updater._hf_weight_iterator.local_plan_getter == original.get_export_plan
    with updater.receiver_pool("eval"):
        assert updater.protocol._publication_version == 2


@pytest.mark.parametrize("mode", ["broadcast", "disk-delta"])
@pytest.mark.asyncio
async def test_eval_admission_failure_releases_update_lock(mode):
    controller = object.__new__(InferenceController)
    controller.context_lock = ContextLock("test")
    controller.args = SimpleNamespace(update_weight_transfer_mode=mode, eval_num_gpus=4)
    controller._eval_fleet = object()
    with pytest.raises(ValueError, match="requires a supported"):
        await controller.start_update_weights(receiver_pool="eval")
    assert not controller.context_lock.locked


def make_dispatcher(mode="gpu-delta"):
    args = SimpleNamespace(
        update_weight_transfer_mode=mode,
        eval_uses_snapshots=True,
        sglang=SimpleNamespace(models=[SimpleNamespace(name="eval")]),
    )
    info = SimpleNamespace(receiver_pool="eval", snapshot_cell_id_to_hashes={"eval-cell": "incarnation"})
    controller = SimpleNamespace(
        start_update_weights=AsyncMock(return_value=info),
        end_update_weights=AsyncMock(),
        abort_update_weights=AsyncMock(),
    )
    actor = SimpleNamespace(
        update_weights=AsyncMock(return_value=WeightUpdateOutput(weight_version=2, failed_cell_ids=())),
        export_hf=AsyncMock(),
    )
    executor = SimpleNamespace(eval=AsyncMock())
    return EvalDispatcher(args, actor, executor, inference_controller=controller), info


@pytest.mark.parametrize("failed_cell_ids", [(), ("failed-eval-cell",)])
@pytest.mark.asyncio
async def test_eval_sync_runs_before_generation_without_snapshot_export(failed_cell_ids):
    dispatcher, info = make_dispatcher()
    info.snapshot_cell_id_to_hashes.update({cell_id: "failed-incarnation" for cell_id in failed_cell_ids})
    events = []
    entered, release = asyncio.Event(), asyncio.Event()

    async def update(**kwargs):
        assert kwargs == {"info": info, "rollout_id": 9}
        events.append("sync")
        return WeightUpdateOutput(weight_version=2, failed_cell_ids=failed_cell_ids)

    async def end(**kwargs):
        assert kwargs == {
            "snapshot_cell_id_to_hashes": info.snapshot_cell_id_to_hashes,
            "failed_cell_ids": failed_cell_ids,
        }
        events.append("unlock")

    async def evaluate(rollout_id, **kwargs):
        assert rollout_id == 9 and kwargs == {"weight_version": "2"}
        assert events == ["sync", "unlock"]
        entered.set()
        await release.wait()
        events.append("generate")

    dispatcher.actor_model.update_weights.side_effect = update
    dispatcher.inference_controller.end_update_weights.side_effect = end
    dispatcher.rollout_manager.eval.side_effect = evaluate
    task = asyncio.create_task(dispatcher.dispatch(9, hf_dir="unused-initial-checkpoint"))
    try:
        await asyncio.wait_for(entered.wait(), timeout=5)
        assert not task.done()
    finally:
        release.set()
        await task
    assert events == ["sync", "unlock", "generate"]
    dispatcher.inference_controller.start_update_weights.assert_awaited_once_with(receiver_pool="eval")
    dispatcher.actor_model.export_hf.assert_not_awaited()
    assert not dispatcher.pending


@pytest.mark.parametrize(
    "error",
    [RuntimeError("sync failed"), asyncio.CancelledError(), None],
    ids=["failed", "cancelled", "missing-version"],
)
@pytest.mark.asyncio
async def test_failed_or_cancelled_sync_unlocks_once_without_eval_or_retry(error):
    dispatcher, _ = make_dispatcher()
    dispatcher.actor_model.update_weights.side_effect = error
    dispatcher.actor_model.update_weights.return_value = WeightUpdateOutput(weight_version=None, failed_cell_ids=())
    with pytest.raises(
        type(error) if error is not None else RuntimeError, match=None if error is not None else "did not produce"
    ):
        await dispatcher.dispatch(3)
    dispatcher.actor_model.update_weights.assert_awaited_once()
    dispatcher.inference_controller.abort_update_weights.assert_awaited_once()
    dispatcher.inference_controller.end_update_weights.assert_not_awaited()
    dispatcher.rollout_manager.eval.assert_not_awaited()


@pytest.mark.parametrize("mode", ["disk-delta", "broadcast"])
@pytest.mark.asyncio
async def test_unsupported_protocol_keeps_existing_snapshot_eval(mode):
    dispatcher, _ = make_dispatcher(mode)
    dispatcher.args.eval_max_in_flight = 1
    await dispatcher.dispatch(3, hf_dir="existing-hf-export")
    await dispatcher.drain()
    dispatcher.inference_controller.start_update_weights.assert_not_awaited()
    dispatcher.actor_model.update_weights.assert_not_awaited()
    dispatcher.rollout_manager.eval.assert_awaited_once_with(
        3, hf_dir="existing-hf-export", export_time_seconds=None, require_marker=False
    )
