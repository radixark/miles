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
from miles.ray.rollout.eval_fleet import InferenceControllerEvalFleet
from miles.ray.rollout.inference_controller import InferenceController
from miles.utils.context_lock import ContextLock


def make_updater(monkeypatch):
    updater = object.__new__(updater_module.WeightUpdater)
    updater.args = SimpleNamespace(update_weight_transfer_mode="gpu-delta")
    updater.protocol = Mock(spec=UpdateWeightFromGpuDelta)
    updater._uses_gpu_delta = True
    updater.conn_status = ConnStatusManager()
    updater.protocol._publication_version = 4
    updater._receiver_pools = updater_module.ReceiverPools()
    updater._hf_weight_iterator = SimpleNamespace()
    updater._registered_adapters = {"rollout-adapter"}
    protocols = []

    def create(_args, receiver_pool):
        assert receiver_pool == "eval"
        protocol = Mock(spec=UpdateWeightFromGpuDelta)
        protocol._publication_version = 0
        protocols.append(protocol)
        return protocol

    monkeypatch.setattr(pools_module, "get_weight_transfer_protocol", create)
    return updater, protocols


def test_eval_and_rollout_keep_independent_committed_state(monkeypatch):
    updater, created = make_updater(monkeypatch)
    bind_iterator = Mock(wraps=updater._bind_protocol_iterator)
    monkeypatch.setattr(updater, "_bind_protocol_iterator", bind_iterator)
    rollout_protocol, rollout_connection = updater.protocol, updater.conn_status
    rollout_protocol.pop_metrics.return_value = {"rollout": 1}
    rollout_connection.mark_reconnected({"rollout": "incarnation-1"})
    with updater.receiver_pool("rollout"):
        assert updater.protocol is rollout_protocol
    bind_iterator.assert_not_called()
    with updater.receiver_pool("eval"):
        assert updater.protocol is not rollout_protocol
        updater.protocol.pop_metrics.return_value = {"eval": 2}
        assert updater.pop_metrics() == {"eval": 2}
        assert updater.protocol._publication_version == 0
        assert updater._registered_adapters == set()
        updater._registered_adapters.add("eval-adapter")
        assert updater.conn_status.needs_reconnect({"eval": "incarnation-1"})
        updater.conn_status.mark_reconnected({"eval": "incarnation-1"})
        updater.protocol._publication_version = 1
        assert updater._hf_weight_iterator.local_consumer == updater.protocol.send_bucket
        with updater.receiver_pool("eval"):
            assert updater.protocol._publication_version == 1
        bind_iterator.assert_called_once_with()
    assert updater.protocol is rollout_protocol
    assert updater.pop_metrics() == {"rollout": 1}
    assert updater.conn_status is rollout_connection
    assert updater.protocol._publication_version == 4
    assert updater._registered_adapters == {"rollout-adapter"}
    assert updater._hf_weight_iterator.local_consumer == rollout_protocol.send_bucket
    with updater.receiver_pool("rollout"):
        updater.protocol._publication_version = 5
    assert bind_iterator.call_count == 2
    with updater.receiver_pool("eval"):
        assert updater.protocol is created[0]
        assert updater.protocol._publication_version == 1
        assert updater._registered_adapters == {"eval-adapter"}
        assert not updater.conn_status.needs_reconnect({"eval": "incarnation-1"})
        assert updater.conn_status.needs_reconnect({"eval": "incarnation-2"})
    assert updater.protocol._publication_version == 5
    assert len(created) == 1


def test_failed_eval_restores_active_rollout_without_rolling_back_committed_eval(monkeypatch):
    updater, _ = make_updater(monkeypatch)
    original = updater.protocol
    with pytest.raises(RuntimeError, match="post-commit"):
        with updater.receiver_pool("eval"):
            with updater.receiver_pool("eval"):
                updater.protocol._publication_version = 2
                raise RuntimeError("post-commit")
    assert updater.protocol is original
    assert updater.protocol._publication_version == 4
    with updater.receiver_pool("eval"):
        assert updater.protocol._publication_version == 2


def test_trainer_reconfiguration_invalidates_every_pool(monkeypatch):
    updater, _ = make_updater(monkeypatch)
    updater.conn_status.mark_reconnected({"r": "1"})
    with updater.receiver_pool("eval"):
        updater.conn_status.mark_reconnected({"e": "1"})
    updater.mark_trainer_stale()
    assert updater.conn_status.needs_reconnect({"r": "1"})
    with updater.receiver_pool("eval"):
        assert updater.conn_status.needs_reconnect({"e": "1"})


@pytest.mark.parametrize("mode", ["broadcast", "disk-delta"])
def test_unsupported_protocol_cannot_join_eval_pool(monkeypatch, mode):
    updater, _ = make_updater(monkeypatch)
    updater.protocol = object()
    updater.args.update_weight_transfer_mode = mode
    with updater.receiver_pool("rollout"):
        pass
    with pytest.raises(ValueError, match="receiver-pool support"):
        with updater.receiver_pool("eval"):
            pytest.fail("must reject before entering")


@pytest.mark.parametrize("pool_id", ["eval", "unknown"])
def test_disk_factory_rejects_unsupported_pool_before_constructing(monkeypatch, pool_id):
    from miles.backends.training_utils.weight_update.protocol import get_weight_transfer_protocol
    from miles.backends.training_utils.weight_update.protocols import delta

    monkeypatch.setattr(
        delta, "UpdateWeightFromDiskDelta", lambda *_args: pytest.fail("must reject before construction")
    )
    with pytest.raises(ValueError, match="does not support receiver pool"):
        get_weight_transfer_protocol(SimpleNamespace(update_weight_transfer_mode="disk-delta"), pool_id)


@pytest.mark.asyncio
async def test_eval_selection_uses_existing_lock_and_leaves_rollout_default():
    controller = object.__new__(InferenceController)
    controller.context_lock = ContextLock("test")
    controller.args = SimpleNamespace(update_weight_transfer_mode="gpu-delta", eval_num_gpus=4)
    controller._eval_fleet = object()
    rollout = SimpleNamespace(
        update_weights=True, api_clients=["rollout"], engine_gpu_counts=[4], engine_gpu_offsets=[0], server_cells={}
    )
    eval_server = SimpleNamespace(
        update_weights=False, api_clients=["eval"], engine_gpu_counts=[2], engine_gpu_offsets=[8], server_cells={}
    )
    controller.servers = {"default": rollout, "eval": eval_server}
    controller._health_monitoring_pause = AsyncMock()
    controller._ensure_cells_ready = AsyncMock()
    info = await controller.start_update_weights(receiver_pool="eval")
    assert info.receiver_pool == "eval"
    assert info.rollout_engines == ["eval"]
    assert info.engine_gpu_counts == [2] and info.engine_gpu_offsets == [8]
    assert controller.context_lock.locked
    controller._ensure_cells_ready.assert_awaited_once_with(model_id="eval")
    await controller.end_update_weights(info.snapshot_cell_id_to_hashes)
    assert not controller.context_lock.locked
    info = await controller.start_update_weights()
    assert info.receiver_pool == "rollout" and info.rollout_engines == ["rollout"]
    await controller.abort_update_weights()
    assert not controller.context_lock.locked
    assert eval_server.update_weights is False


@pytest.mark.parametrize("mode", ["broadcast", "disk-delta"])
@pytest.mark.asyncio
async def test_eval_admission_failure_releases_update_lock(mode):
    controller = object.__new__(InferenceController)
    controller.context_lock = ContextLock("test")
    controller.args = SimpleNamespace(update_weight_transfer_mode=mode, eval_num_gpus=4)
    controller._eval_fleet = None
    with pytest.raises(ValueError, match="requires a supported"):
        await controller.start_update_weights(receiver_pool="eval")
    assert not controller.context_lock.locked


@pytest.mark.parametrize("versions,reason", [(["2", "2"], None), (["2", "1"], "pin_violation"), ([], "pin_violation")])
@pytest.mark.asyncio
async def test_current_eval_checks_every_engine_without_loading_weights(versions, reason):
    fleet = object.__new__(InferenceControllerEvalFleet)
    clients = [
        SimpleNamespace(get_weight_version=AsyncMock(return_value=value), update_weights_from_disk=AsyncMock())
        for value in versions
    ]
    fleet._fleet_api_clients = AsyncMock(return_value=clients)
    fleet._wait_router_ready = AsyncMock()
    result = await fleet.check_version("2")
    assert result.skip_reason == reason
    fleet._fleet_api_clients.assert_awaited_once_with()
    for client in clients:
        client.get_weight_version.assert_awaited_once()
        client.update_weights_from_disk.assert_not_awaited()
    assert fleet._wait_router_ready.await_count == (1 if reason is None else 0)


@pytest.mark.asyncio
async def test_live_eval_uses_confirmed_pool_without_a_checkpoint(monkeypatch):
    from miles.ray.rollout import rollout_executor as module

    executor = object.__new__(module.RolloutExecutor)
    executor.args = SimpleNamespace()
    executor._eval_lock = asyncio.Lock()
    executor._metric_checker = None
    executor.rollout_id = 4
    executor.last_get_rollout_id_of_model_id = {}
    state = object()
    executor._eval_fleet = SimpleNamespace(current=AsyncMock(return_value=state), pin=AsyncMock())
    captured = []

    def generate(input):
        captured.append(input)
        return SimpleNamespace(data={}, metrics={})

    executor.eval_generate_rollout = generate
    monkeypatch.setattr(module, "save_debug_rollout_data", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module, "log_eval_rollout_data", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(module, "is_complete_hf_export", lambda _path: pytest.fail("must not inspect a checkpoint"))
    await executor._eval_checkpoint(4, None, None, True, weight_version="3")
    executor._eval_fleet.current.assert_awaited_once_with("3")
    executor._eval_fleet.pin.assert_not_awaited()
    assert len(captured) == 1
    assert captured[0].rollout_id == 4
    assert captured[0].weight_version == "3"
    assert captured[0].generate_state is state
    assert captured[0].hf_dir is None


@pytest.mark.asyncio
async def test_unconfirmed_eval_pool_never_generates():
    from miles.ray.rollout import rollout_executor as module
    from miles.rollout.checkpoint_eval import EvalSkip

    executor = object.__new__(module.RolloutExecutor)
    executor.args = SimpleNamespace()
    executor._eval_lock = asyncio.Lock()
    executor._eval_fleet = SimpleNamespace(current=AsyncMock(side_effect=EvalSkip("pin_violation")))
    executor.report_eval_skip = lambda step, reason: (step, reason)
    executor.eval_generate_rollout = Mock(side_effect=AssertionError("unconfirmed pool must not generate"))
    assert await executor._eval_checkpoint(4, None, None, True, weight_version="3") == (4, "pin_violation")
    executor._eval_fleet.current.assert_awaited_once_with("3")
    executor.eval_generate_rollout.assert_not_called()


def make_dispatcher():
    args = SimpleNamespace(update_weight_transfer_mode="gpu-delta", eval_num_gpus=4, eval_uses_snapshots=True)
    info = SimpleNamespace(receiver_pool="eval", snapshot_cell_id_to_hashes={"eval-cell": "incarnation"})
    controller = SimpleNamespace(
        start_update_weights=AsyncMock(return_value=info),
        end_update_weights=AsyncMock(),
        abort_update_weights=AsyncMock(),
    )
    actor = SimpleNamespace(update_weights=AsyncMock(return_value=2), export_hf=AsyncMock())
    executor = SimpleNamespace(eval=AsyncMock())
    return EvalDispatcher(args, actor, executor, inference_controller=controller), info


@pytest.mark.asyncio
async def test_eval_sync_runs_before_generation_without_snapshot_export():
    dispatcher, info = make_dispatcher()
    events = []
    entered, release = asyncio.Event(), asyncio.Event()

    async def update(**kwargs):
        assert kwargs == {"info": info, "rollout_id": 9}
        events.append("sync")
        return 2

    async def end(**kwargs):
        assert kwargs == {"snapshot_cell_id_to_hashes": info.snapshot_cell_id_to_hashes}
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
    dispatcher.actor_model.update_weights.return_value = None
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
    dispatcher, _ = make_dispatcher()
    dispatcher.args.update_weight_transfer_mode = mode
    dispatcher.args.eval_max_in_flight = 1
    await dispatcher.dispatch(3, hf_dir="existing-hf-export")
    await dispatcher.drain()
    dispatcher.inference_controller.start_update_weights.assert_not_awaited()
    dispatcher.actor_model.update_weights.assert_not_awaited()
    dispatcher.rollout_manager.eval.assert_awaited_once_with(
        3, hf_dir="existing-hf-export", export_time_seconds=None, require_marker=False
    )
