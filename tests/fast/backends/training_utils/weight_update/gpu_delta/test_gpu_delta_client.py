"""The GPU-delta receiver controls post through the engine client's request helper."""

import pytest

from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient
from miles.backends.training_utils.weight_update.protocols.gpu_delta import client as gpu_delta_client
from miles.utils.http_utils import GeneralHttpClientProvider

SERVER_URL = "http://fake-host:1234"

_PUBLICATION = {
    "session_id": "publication-1",
    "manifest_path": "/checkpoint/gpu-delta/manifest.json",
    "manifest_sha256": "a" * 64,
    "stream_id": "stream-1",
    "base_version": 2,
    "target_version": 3,
    "plan_digest": "b" * 64,
    "participants": [{"engine_id": "engine", "rank_id": "rank"}],
}
_OPTIONS = {
    "release_state": True,
    "pause_mode": "in_place",
    "abort_all_requests": True,
    "weight_version": "actor-42",
    "validate_payload_hash": False,
}


class _Response:
    content = b"{}"

    def raise_for_status(self):
        pass

    def json(self):
        return {}


class _Recorder:
    def __init__(self):
        self.posts = []

    async def post(self, url, **kwargs):
        self.posts.append((url, kwargs["json"]))
        return _Response()


@pytest.fixture
def posts(monkeypatch):
    recorder = _Recorder()
    monkeypatch.setattr(GeneralHttpClientProvider, "client", lambda: recorder)
    return recorder.posts


@pytest.mark.parametrize(
    "call, endpoint, payload",
    [
        (
            lambda c: gpu_delta_client.get_gpu_delta_info(c, engine_id="engine-1"),
            "get_gpu_delta_info",
            {"engine_id": "engine-1"},
        ),
        (lambda c: gpu_delta_client.clear_gpu_delta_state(c), "clear_gpu_delta_state", {}),
        # Unset options stay out of the payload: the server's request schema owns their defaults.
        (
            lambda c: gpu_delta_client.update_weights_from_gpu_delta(c, **_PUBLICATION),
            "update_weights_from_gpu_delta",
            _PUBLICATION,
        ),
        (
            lambda c: gpu_delta_client.update_weights_from_gpu_delta(c, **_PUBLICATION, **_OPTIONS),
            "update_weights_from_gpu_delta",
            _PUBLICATION | _OPTIONS,
        ),
    ],
)
async def test_every_gpu_delta_control_wire_contract(posts, call, endpoint, payload):
    await call(SGLangApiClient(server_url=SERVER_URL))

    assert posts == [(f"{SERVER_URL}/{endpoint}", payload)]


async def test_cell_updaters_submit_the_describe_request_through_the_same_helper(posts):
    client = SGLangApiClient(server_url=SERVER_URL)

    await client._make_request(**gpu_delta_client.info_request("engine-1"))

    assert posts == [(f"{SERVER_URL}/get_gpu_delta_info", {"engine_id": "engine-1"})]
