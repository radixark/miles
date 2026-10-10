from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest
import torch

pytest.importorskip("modelexpress_rl")

from miles.backends.training_utils.weight_update.protocols import modelexpress as mx

from miles.backends.training_utils.weight_update.updater import WeightUpdater


@pytest.fixture
def setup_update(monkeypatch, request):
    sender = getattr(request, "param", True)
    calls = []
    trainer = MagicMock(server_url="mx:8001", model_name="miles-test")
    trainer.pop_metrics.return_value = {"changed_bytes": 2, "total_bytes": 8, "wire_bytes": 3}
    control = MagicMock()
    versions = []

    def create(**kwargs):
        versions.append(kwargs)
        calls.append("create")
        return SimpleNamespace(version_id=kwargs.get("uid", f"opaque-{len(versions) - 1}"))

    def prepare(*, tensor_iter):
        list(tensor_iter)
        calls.append("capture")

    def stage(*, version, tensors):
        assert tensors
        calls.append("stage")
        return SimpleNamespace(publish=lambda: calls.append("publish"))

    trainer.prepare_delta_base.side_effect = prepare
    trainer.stage_shard.side_effect = stage
    control.create_weight_version.side_effect = create
    control.update_weight_version_state.side_effect = lambda *_: calls.append("ready")
    initialize = MagicMock(return_value=trainer)
    new_group = MagicMock(return_value=object())
    monkeypatch.setattr(mx.ModelExpressTrainerClient, "initialize", initialize)
    monkeypatch.setattr(mx.ModelExpressControlClient, "connect", lambda **kwargs: control)
    monkeypatch.setattr(mx.dist, "get_rank", lambda: 0 if sender else 1)
    monkeypatch.setattr(mx.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(
        mx.dist, "all_gather_object", lambda output, value, **kw: output.__setitem__(slice(None), [True, False])
    )
    monkeypatch.setattr(mx.dist, "new_group", new_group)
    monkeypatch.setattr(mx.dist, "broadcast_object_list", lambda *a, **k: None)
    monkeypatch.setattr(mx.dist, "barrier", lambda *a, **k: None)
    monkeypatch.setattr(mx.dist, "all_reduce", lambda *a, **k: None)
    monkeypatch.setattr(mx, "get_gloo_group", lambda: None)
    monkeypatch.setattr(mx, "get_data_replica_rank_and_size", lambda *a: (0 if sender else 1, 2))
    monkeypatch.setattr("miles.backends.training_utils.weight_update.updater.get_gloo_group", lambda: None)

    def record(name, **kwargs):
        calls.append(name)
        return {"success": True}

    engine = SimpleNamespace(
        pause_generation=AsyncMock(side_effect=lambda **kw: record("pause", **kw)),
        flush_cache=AsyncMock(side_effect=lambda: record("flush")),
        update_weights_from_modelexpress=AsyncMock(side_effect=lambda *a, **kw: record("install", **kw)),
        update_weight_version=AsyncMock(side_effect=lambda **kw: record("version", **kw)),
        continue_generation=AsyncMock(side_effect=lambda: record("resume")),
    )
    iterations = []

    def iter_weights(weights, *, materialize=True, **kwargs):
        iterations.append(materialize)
        if materialize:
            yield [("model.weight", torch.tensor([1.0]))]

    iterator = SimpleNamespace(
        placement=SimpleNamespace(gather_pp=False), weight_update_selector="all", iter_hf_weights=iter_weights
    )
    args = Namespace(
        update_weight_transfer_mode="modelexpress",
        colocate=False,
        pause_generation_mode="abort",
        check_lora_weight_equal=False,
        modelexpress_config={
            "model_name": "miles-test",
            "server_url": "mx:8001",
            "object_storage_uri_prefix": "s3://test/run",
            "object_storage_endpoint_url": "http://minio:9000",
            "object_storage_region_name": "us-west-2",
            "initial_base_version_id": "miles-test-v0",
            "seed_checkpoint_path": "/models",
            "full_hf_checkpoint_interval": 2,
        },
    )
    updater = WeightUpdater(
        args,
        [],
        weights_getter=lambda: {},
        model_name="qwen3",
        quantization_config=None,
        iterator_factory=lambda *a, **k: iterator,
        parallel_state=None,
        is_lora=False,
    )
    updater.connect_rollout_engines([engine])
    return SimpleNamespace(
        updater=updater,
        calls=calls,
        versions=versions,
        trainer=trainer,
        control=control,
        engine=engine,
        iterations=iterations,
        initialize=initialize,
        new_group=new_group,
    )


@pytest.mark.parametrize("pause_mode", ["abort", "retract"])
def test_seed_delta_full_delta_and_cutover_order(setup_update, pause_mode):
    run = setup_update
    assert type(run.updater.protocol) is mx.UpdateWeightFromModelExpressDelta
    run.updater.args.pause_generation_mode = pause_mode
    run.updater.update_weights()
    assert run.updater.weight_version == 0
    assert run.calls == ["create", "capture", "version"]
    for update in range(1, 4):
        run.calls.clear()
        run.updater.update_weights()
        assert run.calls == [
            "create",
            "stage",
            "publish",
            "ready",
            "pause",
            "flush",
            "install",
            "version",
            "resume",
        ]
        run.engine.update_weights_from_modelexpress.assert_awaited_with(f"opaque-{update}", flush_cache=False)
        run.engine.update_weight_version.assert_awaited_with(weight_version=str(update))
        run.engine.pause_generation.assert_awaited_with(mode=pause_mode)
        assert run.engine.flush_cache.await_count == update
    assert run.iterations == [True] * 4  # Exactly one gather per baseline/update.
    assert [v["payload_format"] for v in run.versions] == [
        mx.WeightPayloadFormat.FULL_TENSOR,
        mx.WeightPayloadFormat.XOR_DELTA,
        mx.WeightPayloadFormat.FULL_HF_CHECKPOINT,
        mx.WeightPayloadFormat.XOR_DELTA,
    ]
    assert run.versions[1]["base_version_id"] == "miles-test-v0"
    assert "base_version_id" not in run.versions[2]
    assert run.versions[3]["base_version_id"] == "opaque-2"
    assert run.updater.pop_metrics()["perf/update_weights_wire_bytes"] == 3


def test_failed_install_keeps_engines_paused_and_base_unchanged(setup_update):
    run = setup_update
    run.updater.update_weights()
    run.calls.clear()
    run.engine.update_weights_from_modelexpress.side_effect = None
    run.engine.update_weights_from_modelexpress.return_value = {"success": False, "message": "checksum mismatch"}
    with pytest.raises(RuntimeError, match="Base model weight sync failed"):
        run.updater.update_weights()
    run.engine.continue_generation.assert_not_awaited()
    assert run.updater.protocol._current_version_id == "miles-test-v0"
    assert "version" not in run.calls


def test_failed_flush_prevents_refit_and_version_change(setup_update):
    run = setup_update
    run.updater.update_weights()
    run.engine.update_weight_version.reset_mock()
    run.engine.flush_cache.side_effect = RuntimeError("cache flush failed")
    with pytest.raises(RuntimeError, match="cache flush failed"):
        run.updater.update_weights()
    run.engine.update_weights_from_modelexpress.assert_not_awaited()
    run.engine.update_weight_version.assert_not_awaited()
    run.engine.continue_generation.assert_not_awaited()


def test_unset_full_checkpoint_interval_publishes_only_deltas(setup_update):
    run = setup_update
    run.updater.protocol._full_interval = None
    for _ in range(4):
        run.updater.update_weights()
    assert all(v["payload_format"] == mx.WeightPayloadFormat.XOR_DELTA for v in run.versions[1:])


def test_failed_publication_never_marks_ready_or_pauses(setup_update):
    run = setup_update
    run.updater.update_weights()
    run.trainer.stage_shard.side_effect = None
    run.trainer.stage_shard.return_value.publish.side_effect = RuntimeError("S3 write failed")
    with pytest.raises(RuntimeError, match="S3 write failed"):
        run.updater.update_weights()
    run.control.update_weight_version_state.assert_not_called()
    run.engine.pause_generation.assert_not_awaited()


@pytest.mark.parametrize("setup_update", [False], indirect=True)
def test_non_sender_still_consumes_gather_iterator(setup_update, monkeypatch):
    run = setup_update
    monkeypatch.setattr(
        run.updater.protocol, "_rank_zero_call", MagicMock(side_effect=["miles-test-v0", None, "opaque-1", 0.0])
    )
    run.updater.update_weights()
    run.calls.clear()
    run.updater.update_weights()
    assert run.iterations == [False, False]
    assert run.calls == []
    run.initialize.assert_not_called()
    run.trainer.stage_shard.assert_not_called()
    assert run.updater.protocol._trainer is None
    assert run.updater.protocol._staged is None
    assert run.updater.pop_metrics()["perf/update_weights_wire_bytes"] == 0


def test_reconnect_reuses_publisher_group_and_client(setup_update):
    run = setup_update
    run.updater.connect_rollout_engines([run.engine])
    run.new_group.assert_called_once_with(ranks=[0], backend="gloo")
    run.initialize.assert_called_once()
    assert run.initialize.call_args.args[0].process_group is run.new_group.return_value
    storage = run.initialize.call_args.args[0].object_storage
    assert (storage.uri_prefix, storage.endpoint_url, storage.region_name) == (
        "s3://test/run",
        "http://minio:9000",
        "us-west-2",
    )
    run.trainer.stage_shard.assert_not_called()


def test_multiple_buckets_stage_incrementally_and_publish_once(setup_update):
    run = setup_update
    run.updater.update_weights()
    consumed = []
    staged = SimpleNamespace(publish=MagicMock())

    def stage(*, version, tensors):
        if tensors:
            consumed.append(tensors)
        return staged

    tensors = [torch.tensor([float(n)]) for n in range(6)]
    run.trainer.stage_shard.side_effect = stage
    run.updater._hf_weight_iterator.iter_hf_weights = lambda *a, **k: iter(
        [[(f"weight-{n}", tensor)] for n, tensor in enumerate(tensors)]
    )
    run.updater.update_weights()
    assert run.trainer.stage_shard.call_count == 6
    staged.publish.assert_called_once()
    assert len(consumed) == 6
    for n, bucket in enumerate(consumed):
        name, tensor = bucket[0]
        assert name == f"weight-{n}"
        assert tensor is tensors[n]  # MX owns staging; Miles adds no tensor copies.


def test_bucket_staging_failure_never_publishes_or_pauses(setup_update):
    run = setup_update
    run.updater.update_weights()
    staged = SimpleNamespace(publish=MagicMock())
    run.trainer.stage_shard.side_effect = [staged, RuntimeError("staging failed")]
    run.updater._hf_weight_iterator.iter_hf_weights = lambda *a, **k: iter(
        [[("a", torch.tensor([1.0]))], [("b", torch.tensor([2.0]))]]
    )
    with pytest.raises(RuntimeError, match="staging failed"):
        run.updater.update_weights()
    staged.publish.assert_not_called()
    run.engine.pause_generation.assert_not_awaited()


def test_control_error_is_broadcast_before_raising(setup_update, monkeypatch):
    run = setup_update
    broadcast = MagicMock()
    monkeypatch.setattr(mx.dist, "broadcast_object_list", broadcast)
    run.control.create_weight_version.side_effect = RuntimeError("catalog unavailable")
    with pytest.raises(RuntimeError, match="catalog unavailable"):
        run.updater.update_weights()
    assert broadcast.call_args.args[0] == [None, "catalog unavailable"]
    run.trainer.prepare_delta_base.assert_not_called()


def test_receiver_error_details_reach_all_trainer_ranks(setup_update):
    run = setup_update
    run.updater.update_weights()
    error = RuntimeError("HTTP 400")
    error.__notes__ = ["receiver: incompatible checkpoint"]
    run.engine.update_weights_from_modelexpress.side_effect = error
    with pytest.raises(RuntimeError, match="receiver: incompatible checkpoint"):
        run.updater.update_weights()


def test_rank_zero_failure_preserves_cause_after_broadcast(monkeypatch):
    monkeypatch.setattr(mx.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(mx, "get_gloo_group", lambda: None)
    broadcast = Mock()
    monkeypatch.setattr(mx.dist, "broadcast_object_list", broadcast)
    original_error = ValueError("receiver install failed")

    def install():
        raise original_error

    with pytest.raises(RuntimeError, match="receiver install failed") as caught:
        mx.UpdateWeightFromModelExpressDelta._rank_zero_call(None, install)

    broadcast.assert_called_once_with([None, "receiver install failed"], src=0, group=None)
    assert caught.value.__cause__ is original_error
    assert original_error.__traceback__ is not None


def test_peer_reports_broadcast_failure_without_running_action(monkeypatch):
    monkeypatch.setattr(mx.dist, "get_rank", lambda: 1)
    monkeypatch.setattr(mx, "get_gloo_group", lambda: None)
    monkeypatch.setattr(
        mx.dist,
        "broadcast_object_list",
        lambda result, **kwargs: result.__setitem__(1, "receiver install failed"),
    )
    action = Mock()

    with pytest.raises(RuntimeError, match="receiver install failed") as caught:
        mx.UpdateWeightFromModelExpressDelta._rank_zero_call(None, action)

    action.assert_not_called()
    assert caught.value.__cause__ is None
