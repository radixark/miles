"""GPU-delta request options reach the one-shot and staged endpoints."""

import pytest
from tests.fast.backends.sglang_utils import test_sglang_api_client

SERVER_URL = test_sglang_api_client.SERVER_URL
client = test_sglang_api_client.client
recorder = test_sglang_api_client.recorder


@pytest.mark.parametrize("release_state", [True, False])
@pytest.mark.parametrize("flush_cache", [True, False])
@pytest.mark.parametrize("abort_all_requests", [True, False])
async def test_update_weights_from_gpu_delta_forwards_state_lifetime(
    client, recorder, release_state, flush_cache, abort_all_requests
):
    options = {} if release_state else {"release_state": False}
    if not flush_cache:
        options["flush_cache"] = False
    if abort_all_requests:
        options["abort_all_requests"] = True
    await client.update_weights_from_gpu_delta("/checkpoint/gpu-delta/manifest.json", **options)
    verb, url, kwargs = recorder.calls[0]
    assert (verb, url) == ("post", f"{SERVER_URL}/update_weights_from_gpu_delta")
    assert kwargs["json"] == {
        "manifest_path": "/checkpoint/gpu-delta/manifest.json",
        "release_state": release_state,
        "flush_cache": flush_cache,
        "abort_all_requests": abort_all_requests,
    }


@pytest.mark.parametrize("flush_cache", [True, False])
@pytest.mark.parametrize("abort_all_requests", [True, False])
async def test_apply_gpu_delta_forwards_cache_policy(client, recorder, flush_cache, abort_all_requests):
    options = {} if flush_cache else {"flush_cache": False}
    if abort_all_requests:
        options["abort_all_requests"] = True
    await client.apply_gpu_delta("publication-1", **options)
    verb, url, kwargs = recorder.calls[0]
    assert (verb, url) == ("post", f"{SERVER_URL}/apply_gpu_delta")
    assert kwargs["json"] == {
        "session_id": "publication-1",
        "flush_cache": flush_cache,
        "abort_all_requests": abort_all_requests,
    }
