"""Only fresh, deferred versions bypass the engine-wide pause frame."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from tests.fast.backends.training_utils.weight_update.test_dist_weight_update_lifecycle import (
    _make_updater,
    _phases,
    _RecordingApiClient,
)

from miles.utils.lora import LORA_ADAPTER_NAME

_MODULE = "miles.backends.training_utils.weight_update.updater"


class _PendingClient(_RecordingApiClient):
    def __getattr__(self, name):
        original = super().__getattr__(name)

        async def method(**kwargs):
            result = await original(**kwargs)
            if name == "register_lora_adapter" and kwargs.get("defer_publish"):
                result["pending"] = True
            return result

        return method


def _updater(calls, client_type=_PendingClient):
    updater = _make_updater([client_type(calls, index) for index in range(2)])
    updater._lora_sync_config = {"r": 8, "lora_alpha": 16}
    updater._hf_weight_iterator.placement = SimpleNamespace(gather_pp=True)
    updater._hf_weight_iterator.iter_hf_weights.side_effect = lambda _weights, **kwargs: iter(
        [
            [(f"{name}:model.layers.0.self_attn.q_proj.lora_A.weight", torch.ones(2, 2))]
            for name, _ in kwargs["adapters"]
        ]
    )
    return updater


def _push(updater, name="A@2"):
    with patch(f"{_MODULE}.dist") as dist, patch(f"{_MODULE}.get_gloo_group"):
        dist.get_rank.return_value = 0
        updater.push_adapter(name, SimpleNamespace(rank=8, alpha=16))


def test_new_version_has_no_pause_flush_or_base_version_update():
    calls = []
    updater = _updater(calls)
    sessions = []
    updater.protocol.send_bucket.side_effect = lambda bucket: sessions.append(
        updater.protocol.weight_update_session_id
    )
    _push(updater)
    assert _phases(calls) == ["register_lora_adapter", "begin_weight_update", "end_weight_update"]
    assert sessions == ["A@2"]
    assert updater.protocol.weight_update_session_id is None
    for _engine, name, kwargs in calls:
        if name == "begin_weight_update":
            assert kwargs == {"selector": "all", "sync_base": False, "new_lora_names": ["A@2"], "session_id": "A@2"}
        if name == "end_weight_update":
            assert kwargs["session_id"] == "A@2"
            assert set(kwargs["expected_lora_checksums"]) == {"A@2"}
            assert kwargs["expected_lora_checksums"]["A@2"]


def test_existing_name_keeps_the_original_pause_frame():
    calls = []
    updater = _updater(calls)
    updater._registered_adapters.add("A@1")
    _push(updater, "A@1")
    assert _phases(calls) == [
        "pause_generation",
        "flush_cache",
        "begin_weight_update",
        "end_weight_update",
        "continue_generation",
    ]
    assert all("new_lora_names" not in kwargs and "session_id" not in kwargs for _, _, kwargs in calls)


def test_fixed_name_single_lora_keeps_pause_even_without_base_sync():
    calls = []
    updater = _updater(calls)
    updater.is_lora = True
    with patch(f"{_MODULE}.dist") as dist, patch(f"{_MODULE}.get_gloo_group"):
        dist.get_rank.return_value = 0
        updater.update_weights()
    assert _phases(calls) == [
        "pause_generation",
        "flush_cache",
        "register_lora_adapter",
        "begin_weight_update",
        "end_weight_update",
        "update_weight_version",
        "continue_generation",
    ]
    assert LORA_ADAPTER_NAME in updater._registered_adapters
    assert all("defer_publish" not in kwargs for _, _, kwargs in calls)


def test_old_engine_without_deferred_publication_fails_closed():
    calls = []
    updater = _updater(calls, client_type=_RecordingApiClient)
    with pytest.raises(RuntimeError, match="must support deferred"):
        _push(updater)
    updater.protocol.send_bucket.assert_not_called()
    assert "begin_weight_update" not in _phases(calls)
    assert "end_weight_update" not in _phases(calls)
    assert updater.protocol.weight_update_session_id is None


def test_failure_of_one_engine_never_reports_a_successful_publication():
    calls = []
    updater = _updater(calls)
    updater.protocol.rollout_engines[1]._failing_method = "end_weight_update"
    with pytest.raises(RuntimeError, match="end_weight_update failed"):
        _push(updater)
    assert "continue_generation" not in _phases(calls)
    assert any(kwargs.get("abort") for _, name, kwargs in calls if name == "end_weight_update")
    assert updater.protocol.weight_update_session_id is None


def test_non_driver_receives_the_prepare_failure_instead_of_waiting_at_a_barrier():
    updater = _updater([])
    operation = Mock()

    def broadcast_error(values, **kwargs):
        values[0] = "registration rejected on rank zero"

    with patch(f"{_MODULE}.dist") as dist, patch(f"{_MODULE}.get_gloo_group"):
        dist.get_rank.return_value = 1
        dist.broadcast_object_list.side_effect = broadcast_error
        with pytest.raises(RuntimeError, match="registration rejected"):
            updater._run_driver_phase(operation)
        operation.assert_not_called()
        dist.barrier.assert_not_called()


def test_sender_bucket_failure_is_shared_before_any_engine_can_commit():
    calls = []
    updater = _updater(calls)
    updater.protocol.send_bucket.side_effect = RuntimeError("bucket rejected")

    def collect_errors(errors, local_error, **kwargs):
        errors[:] = [local_error, None]

    with patch(f"{_MODULE}.dist") as dist, patch(f"{_MODULE}.get_gloo_group"):
        dist.get_rank.return_value = 0
        dist.get_world_size.return_value = 2
        dist.all_gather_object.side_effect = collect_errors
        with pytest.raises(RuntimeError, match="bucket rejected"):
            updater.push_adapter("A@2", SimpleNamespace(rank=8, alpha=16))
    ends = [kwargs for _, name, kwargs in calls if name == "end_weight_update"]
    assert ends and all(kwargs["abort"] for kwargs in ends)
    updater.protocol.finalize.assert_not_called()
