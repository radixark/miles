"""Versioned adapter pushes run as staged sessions: no pause frame, no weight
version, commit gated on the expected checksums and the deferred-publish ack."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from tests.fast.backends.training_utils.weight_update.test_dist_weight_update_lifecycle import (
    _kwargs_of,
    _make_updater,
    _phases,
    _RecordingApiClient,
)

from miles.backends.training_utils.weight_update.hf_weight_iterator import WeightUpdatePlacement
from miles.utils.lora import LORA_ADAPTER_NAME

_MODULE = "miles.backends.training_utils.weight_update.updater"
_SESSION = "miles.backends.training_utils.weight_update.session"


class _PendingAckClient(_RecordingApiClient):
    """Acks defer_publish the way an engine with staged publication does."""

    def __getattr__(self, name):
        original = super().__getattr__(name)

        async def method(**kwargs):
            result = await original(**kwargs)
            if name == "register_lora_adapter" and kwargs.get("defer_publish"):
                result["pending"] = True
            return result

        return method


def _updater(calls, client_type=_PendingAckClient, failing_method=None):
    engines = [client_type(calls, index, failing_method if index == 0 else None) for index in range(2)]
    updater = _make_updater(engines)
    updater._lora_sync_config = {"r": 8, "lora_alpha": 16}
    updater._hf_weight_iterator.placement = WeightUpdatePlacement(gather_pp=True)
    updater._hf_weight_iterator.iter_hf_weights.side_effect = lambda _weights, **kwargs: iter(
        [
            [(f"{name}:model.layers.0.self_attn.q_proj.lora_A.weight", torch.ones(2, 2))]
            for name, _ in kwargs["adapters"]
        ]
    )
    return updater


def _push(updater, name="A@2", lora_path="/ckpt/run/sampler_weights/2"):
    with (
        patch(f"{_MODULE}.dist") as dist,
        patch(f"{_MODULE}.get_gloo_group"),
        patch(f"{_SESSION}.dist") as session_dist,
        patch(f"{_SESSION}.get_gloo_group"),
    ):
        dist.get_rank.return_value = session_dist.get_rank.return_value = 0
        updater.push_adapter(name, SimpleNamespace(rank=8, alpha=16), lora_path)


def test_staged_push_skips_the_pause_frame_and_the_weight_version():
    calls = []
    updater = _updater(calls)
    _push(updater)
    assert _phases(calls) == ["register_lora_adapter", "begin_weight_update", "end_weight_update"]
    for register in _kwargs_of(calls, "register_lora_adapter"):
        assert register["defer_publish"] and register["lora_path"] == "/ckpt/run/sampler_weights/2"
    for kwargs in _kwargs_of(calls, "begin_weight_update"):
        assert kwargs == {"selector": "all", "sync_base": False}
    for kwargs in _kwargs_of(calls, "end_weight_update"):
        assert set(kwargs["expected_lora_checksums"]) == {"A@2"}
        assert kwargs["expected_lora_checksums"]["A@2"]


def test_fixed_name_single_lora_keeps_the_pause_frame():
    calls = []
    updater = _updater(calls)
    updater.is_lora = True
    with (
        patch(f"{_MODULE}.dist") as dist,
        patch(f"{_MODULE}.get_gloo_group"),
        patch(f"{_SESSION}.dist") as session_dist,
        patch(f"{_SESSION}.get_gloo_group"),
    ):
        dist.get_rank.return_value = session_dist.get_rank.return_value = 0
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
    assert all(not kwargs.get("defer_publish") for kwargs in _kwargs_of(calls, "register_lora_adapter"))


def test_missing_pending_ack_fails_closed():
    calls = []
    updater = _updater(calls, client_type=_RecordingApiClient)
    with pytest.raises(RuntimeError, match="deferred LoRA publication"):
        _push(updater)
    updater.protocol.send_bucket.assert_not_called()
    assert "begin_weight_update" not in _phases(calls)


def test_engine_failure_aborts_the_staged_session():
    calls = []
    updater = _updater(calls, failing_method="end_weight_update")
    with pytest.raises(RuntimeError):
        _push(updater)
    assert any(kwargs.get("abort") for kwargs in _kwargs_of(calls, "end_weight_update"))
    assert "update_weight_version" not in _phases(calls)
