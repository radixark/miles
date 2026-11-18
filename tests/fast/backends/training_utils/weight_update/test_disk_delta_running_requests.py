"""A disk-delta reload while the engine still has requests running.

The fake answers like sglang: a retract pause moves running requests back to the
waiting queue, an in_place pause leaves them running, and /update_weights_from_disk
flushes the cache after loading unless the request sends flush_cache=false. A flush
refused because requests are running fails the update with HTTP 400.
"""

from argparse import Namespace
from unittest.mock import MagicMock, patch

import httpx
import pytest

from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient
from miles.backends.training_utils.weight_update.protocols.delta import UpdateWeightFromDiskDelta
from miles.utils.http_utils import GeneralHttpClientProvider

_MODULE = "miles.backends.training_utils.weight_update.protocols.delta"
_SERVER_URL = "http://engine:30000"
_NOT_FLUSHED = (
    "Weights were updated, but the cache was not flushed because requests are still running or queued. "
    "Retract or finish them, then call /flush_cache."
)


class _EngineWithRunningRequests:
    def __init__(self):
        self.running = True
        self.calls: list[tuple[str, dict]] = []

    async def get(self, url, **kwargs):
        return self._answer("GET", url, {})

    async def post(self, url, *, json=None, **kwargs):
        return self._answer("POST", url, json or {})

    def _answer(self, method: str, url: str, payload: dict) -> httpx.Response:
        endpoint = url.rsplit("/", 1)[-1]
        self.calls.append((endpoint, payload))
        if endpoint == "pause_generation" and payload["mode"] == "retract":
            self.running = False
        request = httpx.Request(method, url)
        if endpoint == "update_weights_from_disk" and payload.get("flush_cache", True) and self.running:
            return httpx.Response(400, json={"success": False, "message": _NOT_FLUSHED}, request=request)
        return httpx.Response(200, json={"success": True}, request=request)


def _reload(monkeypatch: pytest.MonkeyPatch, engine: _EngineWithRunningRequests, pause_generation_mode: str) -> None:
    monkeypatch.setattr(GeneralHttpClientProvider, "client", lambda: engine)
    protocol = UpdateWeightFromDiskDelta.__new__(UpdateWeightFromDiskDelta)
    protocol.args = Namespace(
        update_weight_local_checkpoint_dir="/local/ckpt",
        update_weight_disk_dir="/shared/delta",
        pause_generation_mode=pause_generation_mode,
        check_weight_update_equal=False,
    )
    protocol.rollout_engines = [SGLangApiClient(server_url=_SERVER_URL)]
    protocol._post_write_hook = None
    protocol._version_dir = "/shared/delta/v7"
    protocol._snapshot_file_version = 7

    with patch(f"{_MODULE}.dist") as dist_mock, patch(f"{_MODULE}.get_gloo_group", return_value=MagicMock()):
        dist_mock.get_rank.return_value = 0
        protocol._reload_engines(weight_version=42)


def test_in_place_reload_does_not_ask_the_engine_to_flush(monkeypatch):
    """in_place keeps the paused requests running, so the engine would refuse a post-load
    flush and fail the update, leaving generation paused."""
    engine = _EngineWithRunningRequests()

    _reload(monkeypatch, engine, "in_place")

    assert [endpoint for endpoint, _payload in engine.calls] == [
        "pull_weights",
        "pause_generation",
        "update_weights_from_disk",
        "continue_generation",
    ]


def test_retract_reload_keeps_both_flushes(monkeypatch):
    """The same engine accepts a retract reload, which still flushes before and after loading."""
    engine = _EngineWithRunningRequests()

    _reload(monkeypatch, engine, "retract")

    assert [endpoint for endpoint, _payload in engine.calls] == [
        "pull_weights",
        "pause_generation",
        "flush_cache",
        "update_weights_from_disk",
        "continue_generation",
    ]
    assert "flush_cache" not in engine.calls[3][1]
