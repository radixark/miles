import asyncio
import inspect

import httpx
import pytest

from miles.backends.sglang_utils import sglang_api_client
from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient
from miles.utils.http_utils import GeneralHttpClientProvider

SERVER_URL = "http://fake-host:1234"


class _FakeResponse:
    def __init__(self, status_code: int = 200, payload: dict | None = None, text: str = ""):
        self.status_code = status_code
        self._payload = payload if payload is not None else {"ok": True}
        self.text = text

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(f"status {self.status_code}", request=None, response=None)

    def json(self):
        return self._payload


class _Recorder:
    def __init__(self):
        self.calls: list[tuple[str, str, dict]] = []
        self.responses: list[_FakeResponse] = []

    def install(self, monkeypatch, responses: list[_FakeResponse] | None = None):
        self.responses = list(responses or [])
        monkeypatch.setattr(GeneralHttpClientProvider, "client", lambda: self)

    async def get(self, url, **kwargs):
        return self._record("get", url, kwargs)

    async def post(self, url, **kwargs):
        return self._record("post", url, kwargs)

    async def delete(self, url, **kwargs):
        return self._record("delete", url, kwargs)

    def _record(self, verb: str, url: str, kwargs: dict) -> _FakeResponse:
        self.calls.append((verb, url, kwargs))
        if self.responses:
            return self.responses.pop(0)
        return _FakeResponse()


@pytest.fixture
def recorder(monkeypatch):
    rec = _Recorder()
    rec.install(monkeypatch)
    return rec


@pytest.fixture
def client():
    return SGLangApiClient(server_url=SERVER_URL)


async def test_post_methods_hit_the_server_url_with_expected_payload(client, recorder):
    """Every POST-based method targets ``<server_url>/<endpoint>`` and sends the documented payload."""
    await client.update_weights_from_tensor(serialized_named_tensors=["a"], load_format="direct", flush_cache=True)
    await client.update_weight_version("run-0001")
    await client.begin_weight_update()
    await client.end_weight_update()

    assert [(verb, url) for verb, url, _ in recorder.calls] == [
        ("post", f"{SERVER_URL}/update_weights_from_tensor"),
        ("post", f"{SERVER_URL}/update_weight_version"),
        ("post", f"{SERVER_URL}/begin_weight_update"),
        ("post", f"{SERVER_URL}/end_weight_update"),
    ]
    assert recorder.calls[0][2]["json"] == {
        "serialized_named_tensors": ["a"],
        "load_format": "direct",
        "flush_cache": True,
        "selector": "all",
    }
    assert recorder.calls[1][2]["json"] == {"new_version": "run-0001", "abort_all_requests": True}


async def test_update_weights_from_tensor_omits_weight_version_when_not_given(client, recorder):
    """``weight_version`` stays out of the payload unless the caller passes one."""
    await client.update_weights_from_tensor(serialized_named_tensors=["a"])

    assert "weight_version" not in recorder.calls[0][2]["json"]


async def test_check_weights_renames_skip_list_to_skip_tensor_list(client, recorder):
    """sglang's CheckWeightsReqInput expects ``skip_tensor_list``, not ``skip_list``."""
    await client.check_weights(action="reset_tensors", skip_list=["lm_head"])

    verb, url, kwargs = recorder.calls[0]
    assert (verb, url) == ("post", f"{SERVER_URL}/weights_checker")
    assert kwargs["json"] == {
        "action": "reset_tensors",
        "allow_quant_error": False,
        "selector": "all",
        "skip_tensor_list": ["lm_head"],
    }


async def test_pull_weights_forwards_the_explicit_checkpoint_dirs(client, recorder):
    """The client holds no args, so both checkpoint dirs arrive as explicit parameters."""
    await client.pull_weights(target_version=7, local_checkpoint_dir="/local", source_dir="/shared")

    assert recorder.calls[0][2]["json"] == {
        "local_checkpoint_dir": "/local",
        "source_dir": "/shared",
        "target_version": 7,
    }


async def test_get_methods_hit_the_documented_endpoints(client, recorder):
    """GET-based methods keep their (non-uniform) endpoint names."""
    await client.health_generate()
    await client.get_server_info()
    await client.get_parallelism_info(rank=3)

    assert [(verb, url) for verb, url, _ in recorder.calls] == [
        ("get", f"{SERVER_URL}/health_generate"),
        ("get", f"{SERVER_URL}/server_info"),
        ("get", f"{SERVER_URL}/parallelism_config"),
    ]
    assert recorder.calls[2][2]["params"] == {"rank": 3}


async def test_get_weight_version_falls_back_to_the_legacy_endpoint(client, monkeypatch):
    """Old sglang builds only serve /get_weight_version, so a non-200 /model_info must fall through."""
    rec = _Recorder()
    rec.install(
        monkeypatch, responses=[_FakeResponse(status_code=404), _FakeResponse(payload={"weight_version": "v3"})]
    )

    assert await client.get_weight_version() == "v3"
    assert [url for _verb, url, _kwargs in rec.calls] == [
        f"{SERVER_URL}/model_info",
        f"{SERVER_URL}/get_weight_version",
    ]


async def test_release_memory_occupation_flushes_the_cache_first(client, recorder):
    """Offload is only safe once the working queue is drained."""
    await client.release_memory_occupation(tags=["weights"])

    assert [(verb, url) for verb, url, _ in recorder.calls] == [
        ("get", f"{SERVER_URL}/flush_cache"),
        ("post", f"{SERVER_URL}/release_memory_occupation"),
    ]


async def test_destroy_weights_update_group_swallows_request_errors(client, monkeypatch):
    """A freshly created engine has no group yet; failing to destroy it must not propagate."""

    class _Raising:
        async def post(self, url, **kwargs):
            raise httpx.ConnectError("no such group")

    monkeypatch.setattr(GeneralHttpClientProvider, "client", lambda: _Raising())

    assert await client.destroy_weights_update_group("group-0") is None


async def test_flush_cache_sleeps_between_pending_request_retries(client, monkeypatch):
    """Regression test for the fully_async weight-update crash: sglang
    returns 400 (not an exception) while requests are still pending, so the
    retry loop must back off on THAT path too, or all 60 "attempts" burn
    through in a fraction of a second -- nowhere near enough time for
    in-flight generation to drain -- and flush_cache raises TimeoutError
    almost immediately after pause_generation instead of after ~60s."""
    sleep_calls = []

    async def fake_sleep(seconds):
        sleep_calls.append(seconds)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    _Recorder().install(monkeypatch, responses=[_FakeResponse(status_code=400) for _ in range(60)])

    with pytest.raises(TimeoutError, match="Timeout while flushing cache"):
        await client.flush_cache()

    assert len(sleep_calls) == 60, (
        f"expected the loop to back off on every one of its 60 attempts, got {len(sleep_calls)} sleeps "
        "-- a 400 response (pending requests) must not skip the retry delay"
    )


async def test_flush_cache_retries_a_refused_connection(client, monkeypatch):
    """A server that is briefly unreachable mid-offload must not fail the whole weight update."""

    class _RefusingThenServing:
        def __init__(self):
            self.attempts = 0

        async def get(self, url, **kwargs):
            self.attempts += 1
            if self.attempts == 1:
                raise httpx.ConnectError("connection refused")
            return _FakeResponse()

    serving = _RefusingThenServing()
    monkeypatch.setattr(GeneralHttpClientProvider, "client", lambda: serving)
    monkeypatch.setattr(asyncio, "sleep", _noop_sleep)

    await client.flush_cache()

    assert serving.attempts == 2


_REMAINING_POST_CASES = [
    (
        lambda c: c.load_lora_adapter_from_tensors(
            lora_name="l", config_dict={"r": 8}, serialized_named_tensors=["t"]
        ),
        "load_lora_adapter_from_tensors",
        {"lora_name": "l", "config_dict": {"r": 8}, "serialized_named_tensors": ["t"], "pinned": False},
    ),
    (
        lambda c: c.load_lora_adapter_from_distributed(
            lora_name="l", config_dict={"r": 8}, names=["w"], dtypes=["torch.bfloat16"], shapes=[[1]], group_name="g"
        ),
        "load_lora_adapter_from_distributed",
        {
            "lora_name": "l",
            "config_dict": {"r": 8},
            "names": ["w"],
            "dtypes": ["bfloat16"],
            "shapes": [[1]],
            "group_name": "g",
            "pinned": False,
            "upsert": False,
        },
    ),
    (
        lambda c: c.register_lora_adapter("l", {"r": 8}),
        "register_lora_adapter",
        {"lora_name": "l", "config_dict": {"r": 8}, "pinned": False},
    ),
    (
        lambda c: c.resume_memory_occupation(tags=["weights"]),
        "resume_memory_occupation",
        {"tags": ["weights"]},
    ),
    (
        lambda c: c.update_weights_from_disk(model_path="/ckpt", load_format="direct", weight_version="v1"),
        "update_weights_from_disk",
        {"model_path": "/ckpt", "load_format": "direct", "weight_version": "v1"},
    ),
    (
        lambda c: c.init_weights_update_group("addr", 1234, 8, 9, "g", "nccl"),
        "init_weights_update_group",
        {
            "master_address": "addr",
            "master_port": 1234,
            "rank_offset": 8,
            "world_size": 9,
            "group_name": "g",
            "backend": "nccl",
        },
    ),
    (lambda c: c.destroy_weights_update_group("g"), "destroy_weights_update_group", {"group_name": "g"}),
    (
        lambda c: c.update_weights_from_distributed(
            names=["w"], dtypes=["torch.bfloat16"], shapes=[[1]], group_name="g", flush_cache=True
        ),
        "update_weights_from_distributed",
        {"names": ["w"], "dtypes": ["bfloat16"], "shapes": [[1]], "group_name": "g", "flush_cache": True},
    ),
    (lambda c: c.pause_generation(mode="abort"), "pause_generation", {"mode": "abort"}),
    (lambda c: c.continue_generation(), "continue_generation", {}),
    (
        lambda c: c.start_profile(output_dir="/out", num_steps=3),
        "start_profile",
        {
            "output_dir": "/out",
            "start_step": None,
            "num_steps": 3,
            "activities": None,
            "profile_by_stage": False,
            "with_stack": None,
            "record_shapes": None,
        },
    ),
    (lambda c: c.stop_profile(), "stop_profile", {}),
]


@pytest.mark.parametrize("call, endpoint, expected_payload", _REMAINING_POST_CASES)
async def test_every_remaining_post_method_wire_contract(client, recorder, call, endpoint, expected_payload):
    """Each remaining POST method posts its documented payload to its own endpoint."""
    await call(client)

    assert recorder.calls == [("post", f"{SERVER_URL}/{endpoint}", {"json": expected_payload})]


async def test_get_remote_instance_transfer_engine_info_unwraps_the_response(client, monkeypatch):
    """The method returns the inner field, not the whole JSON body."""
    rec = _Recorder()
    rec.install(monkeypatch, responses=[_FakeResponse(payload={"remote_instance_transfer_engine_info": {"a": 1}})])

    assert await client.get_remote_instance_transfer_engine_info(rank=2) == {"a": 1}
    assert rec.calls[0][2]["params"] == {"rank": 2}


async def test_every_public_method_is_a_coroutine_function():
    """The client is async-only: a caller that forgets to await must never silently fire nothing."""
    methods = [
        name for name in dir(SGLangApiClient) if not name.startswith("__") and callable(getattr(SGLangApiClient, name))
    ]
    non_async = [name for name in methods if not inspect.iscoroutinefunction(getattr(SGLangApiClient, name))]

    assert non_async == []


class TestWaitServerHealthy:
    """``wait_server_healthy`` polls until the server answers, and gives up if the process dies."""

    async def test_it_polls_health_then_flush_cache(self, monkeypatch):
        """Readiness means both the health endpoint and a drained working queue."""
        rec = _Recorder()
        rec.install(monkeypatch, responses=[_FakeResponse(status_code=503), _FakeResponse(), _FakeResponse()])
        monkeypatch.setattr(asyncio, "sleep", _noop_sleep)

        await sglang_api_client.wait_server_healthy(server_url=SERVER_URL, api_key="k", is_process_alive=lambda: True)

        assert [url for _verb, url, _kwargs in rec.calls] == [
            f"{SERVER_URL}/health_generate",
            f"{SERVER_URL}/health_generate",
            f"{SERVER_URL}/flush_cache",
        ]

    async def test_it_raises_once_the_server_process_is_gone(self, monkeypatch):
        """Polling a dead process forever would hang engine startup instead of reporting it."""

        class _Refusing:
            async def get(self, url, **kwargs):
                raise httpx.ConnectError("connection refused")

        monkeypatch.setattr(GeneralHttpClientProvider, "client", lambda: _Refusing())
        monkeypatch.setattr(asyncio, "sleep", _noop_sleep)

        with pytest.raises(Exception, match="Server process terminated unexpectedly"):
            await sglang_api_client.wait_server_healthy(
                server_url=SERVER_URL, api_key="k", is_process_alive=lambda: False
            )


async def _noop_sleep(seconds):
    return None
