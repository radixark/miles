import asyncio
import inspect
import json

import httpx
import pytest

from miles.backends.sglang_utils import sglang_api_client
from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient
from miles.utils.http_utils import GeneralHttpClientProvider

SERVER_URL = "http://fake-host:1234"


class _FakeResponse:
    def __init__(self, status_code: int = 200, payload: dict | None = None, text: str = "", body: bytes | None = None):
        self.status_code = status_code
        self._payload = payload if payload is not None else {"ok": True}
        self.text = text
        self.content = json.dumps(self._payload).encode() if body is None else body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(f"status {self.status_code}", request=None, response=None)

    def json(self):
        if not self.content:
            raise ValueError("a response with no body carries no json")
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
    assert recorder.calls[1][2]["json"] == {"new_version": "run-0001", "abort_all_requests": False}


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
    (lambda c: c.unload_lora_adapter("l"), "unload_lora_adapter", {"lora_name": "l"}),
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
        {
            "names": ["w"],
            "dtypes": ["bfloat16"],
            "shapes": [[1]],
            "group_name": "g",
            "flush_cache": True,
            "selector": "all",
        },
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


class TestProbeServerHealthy:
    """``probe_server_healthy`` is a single bounded shot, because it runs inside the locked tick sweep."""

    async def test_a_healthy_server_probes_true(self, monkeypatch):
        """A ready engine is what moves the cell out of the initializing state."""
        rec = _Recorder()
        rec.install(monkeypatch, responses=[_FakeResponse()])

        assert await sglang_api_client.probe_server_healthy(server_url=SERVER_URL, api_key="k") is True
        assert [url for _verb, url, _kwargs in rec.calls] == [f"{SERVER_URL}/health_generate"]

    async def test_a_server_that_is_still_loading_probes_false(self, monkeypatch):
        """A non-200 means not ready yet, not a failure to report upwards."""
        rec = _Recorder()
        rec.install(monkeypatch, responses=[_FakeResponse(status_code=503)])

        assert await sglang_api_client.probe_server_healthy(server_url=SERVER_URL, api_key="k") is False

    async def test_a_server_that_is_not_listening_probes_false(self, monkeypatch):
        """The port only opens minutes after launch, so a refused connection is the normal case."""

        class _Refusing:
            async def get(self, url, **kwargs):
                raise httpx.ConnectError("connection refused")

        monkeypatch.setattr(GeneralHttpClientProvider, "client", lambda: _Refusing())

        assert await sglang_api_client.probe_server_healthy(server_url=SERVER_URL, api_key="k") is False

    async def test_the_probe_is_bounded_so_a_wedged_engine_cannot_stall_the_sweep(self, monkeypatch):
        """The shared http client has no read timeout, so this call must carry its own."""
        rec = _Recorder()
        rec.install(monkeypatch, responses=[_FakeResponse()])

        await sglang_api_client.probe_server_healthy(server_url=SERVER_URL, api_key="k")

        assert rec.calls[0][2]["timeout"] == 5.0

    async def test_a_socket_level_os_error_probes_false(self, monkeypatch):
        """A bare ``OSError`` (DNS/socket failure outside httpx's hierarchy) is an unhealthy probe, not a crash."""

        class _Failing:
            async def get(self, url, **kwargs):
                raise OSError("name resolution failed")

        monkeypatch.setattr(GeneralHttpClientProvider, "client", lambda: _Failing())

        assert await sglang_api_client.probe_server_healthy(server_url=SERVER_URL, api_key="k") is False

    async def test_a_caller_supplied_timeout_replaces_the_default(self, monkeypatch):
        """A caller that knows its own sweep budget must be able to tighten the bound."""
        rec = _Recorder()
        rec.install(monkeypatch, responses=[_FakeResponse()])

        await sglang_api_client.probe_server_healthy(server_url=SERVER_URL, api_key="k", timeout=0.25)

        assert rec.calls[0][2]["timeout"] == 0.25

    async def test_it_authenticates_like_every_other_call(self, monkeypatch):
        """A probe rejected for missing auth would look exactly like an engine that never starts."""
        rec = _Recorder()
        rec.install(monkeypatch, responses=[_FakeResponse()])

        await sglang_api_client.probe_server_healthy(server_url=SERVER_URL, api_key="secret")

        assert rec.calls[0][2]["headers"]["Authorization"] == "Bearer secret"


class TestWaitServerHealthy:
    """``wait_server_healthy`` polls until the server answers on health and flush_cache."""

    async def test_it_polls_health_then_flush_cache(self, monkeypatch):
        """Readiness means both the health endpoint and a drained working queue."""
        rec = _Recorder()
        rec.install(monkeypatch, responses=[_FakeResponse(status_code=503), _FakeResponse(), _FakeResponse()])
        monkeypatch.setattr(asyncio, "sleep", _noop_sleep)

        await sglang_api_client.wait_server_healthy(server_url=SERVER_URL, api_key="k")

        assert [url for _verb, url, _kwargs in rec.calls] == [
            f"{SERVER_URL}/health_generate",
            f"{SERVER_URL}/health_generate",
            f"{SERVER_URL}/flush_cache",
        ]

    async def test_both_phases_retry_transport_errors_and_non_200_responses(self, monkeypatch):
        """Health and flush_cache each back off and retry on an httpx error as well as on a non-200 body."""

        class _Sequenced:
            def __init__(self, outcomes):
                self.outcomes = list(outcomes)
                self.urls: list[str] = []

            async def get(self, url, **kwargs):
                self.urls.append(url)
                outcome = self.outcomes.pop(0)
                if isinstance(outcome, Exception):
                    raise outcome
                return outcome

        sequenced = _Sequenced(
            [
                httpx.ConnectError("health refused"),
                _FakeResponse(status_code=503),
                _FakeResponse(),
                httpx.ConnectError("flush refused"),
                _FakeResponse(status_code=500),
                _FakeResponse(),
            ]
        )
        sleep_calls = []

        async def recording_sleep(seconds):
            sleep_calls.append(seconds)

        monkeypatch.setattr(GeneralHttpClientProvider, "client", lambda: sequenced)
        monkeypatch.setattr(asyncio, "sleep", recording_sleep)

        await sglang_api_client.wait_server_healthy(server_url=SERVER_URL, api_key="k")

        assert sequenced.urls == [
            f"{SERVER_URL}/health_generate",
            f"{SERVER_URL}/health_generate",
            f"{SERVER_URL}/health_generate",
            f"{SERVER_URL}/flush_cache",
            f"{SERVER_URL}/flush_cache",
            f"{SERVER_URL}/flush_cache",
        ]
        assert sleep_calls == [2, 2, 2, 2]


class TestMakeRequest:
    """``_make_request`` is the shared POST path behind most of the client's methods."""

    async def test_it_returns_the_decoded_json_body(self, client, monkeypatch):
        """Callers read fields out of the result, so the raw response object is not an acceptable return value."""
        _Recorder().install(monkeypatch, responses=[_FakeResponse(payload={"success": True, "message": "done"})])

        assert await client.begin_weight_update() == {"success": True, "message": "done"}

    async def test_a_failed_request_raises_with_the_server_response_body_attached(self, client, monkeypatch):
        """sglang puts the real reason in the body, so the note must carry it into the traceback."""
        _Recorder().install(monkeypatch, responses=[_FakeResponse(status_code=500, text="cuda oom")])

        with pytest.raises(httpx.HTTPStatusError) as excinfo:
            await client.begin_weight_update()

        assert excinfo.value.__notes__ == ["response.text='cuda oom'"]


class TestHealthGenerate:
    """``health_generate`` is the client-bound counterpart of the module-level probe."""

    async def test_it_returns_true_and_forwards_a_custom_timeout(self, client, monkeypatch):
        """The shared http client has no read timeout, so the caller's bound must reach the wire."""
        rec = _Recorder()
        rec.install(monkeypatch, responses=[_FakeResponse()])

        assert await client.health_generate(timeout=1.5) is True
        assert rec.calls[0][2]["timeout"] == 1.5

    async def test_a_non_success_status_propagates(self, client, monkeypatch):
        """An unhealthy engine must raise rather than report ``True``."""
        _Recorder().install(monkeypatch, responses=[_FakeResponse(status_code=503)])

        with pytest.raises(httpx.HTTPStatusError):
            await client.health_generate()


class TestInformationGetters:
    """The three read-only GET methods each carry their own endpoint, params and bound."""

    async def test_remote_instance_transfer_engine_info_targets_its_endpoint_with_a_bound(self, client, monkeypatch):
        """Endpoint name and the 5-second bound are part of this method's wire contract."""
        rec = _Recorder()
        rec.install(monkeypatch, responses=[_FakeResponse(payload={"remote_instance_transfer_engine_info": {"a": 1}})])

        await client.get_remote_instance_transfer_engine_info(rank=2)

        verb, url, kwargs = rec.calls[0]
        assert (verb, url) == ("get", f"{SERVER_URL}/get_remote_instance_transfer_engine_info")
        assert kwargs == {"params": {"rank": 2}, "timeout": 5.0}

    async def test_parallelism_info_returns_the_whole_json_body(self, client, monkeypatch):
        """Unlike the transfer-engine getter, this one does not unwrap a field."""
        rec = _Recorder()
        rec.install(monkeypatch, responses=[_FakeResponse(payload={"tp_size": 4, "dp_size": 2})])

        assert await client.get_parallelism_info(rank=1) == {"tp_size": 4, "dp_size": 2}
        assert rec.calls[0][2] == {"params": {"rank": 1}, "timeout": 5.0}

    async def test_server_info_returns_the_whole_json_body_without_params(self, client, monkeypatch):
        """``/server_info`` is rank-independent, so it must not send a rank parameter."""
        rec = _Recorder()
        rec.install(monkeypatch, responses=[_FakeResponse(payload={"version": "0.4.0"})])

        assert await client.get_server_info() == {"version": "0.4.0"}
        assert "params" not in rec.calls[0][2]
        assert rec.calls[0][2]["timeout"] == 5.0


_DIRECT_HTTP_METHODS = [
    ("get_remote_instance_transfer_engine_info", lambda c: c.get_remote_instance_transfer_engine_info(rank=0)),
    ("get_parallelism_info", lambda c: c.get_parallelism_info(rank=0)),
    ("get_server_info", lambda c: c.get_server_info()),
    ("pause_generation", lambda c: c.pause_generation()),
    ("continue_generation", lambda c: c.continue_generation()),
    ("start_profile", lambda c: c.start_profile()),
    ("stop_profile", lambda c: c.stop_profile()),
]


class TestMethodsThatBypassMakeRequest:
    @pytest.mark.parametrize("name, call", _DIRECT_HTTP_METHODS, ids=[name for name, _call in _DIRECT_HTTP_METHODS])
    async def test_methods_that_bypass_make_request_still_propagate_non_success_status(
        self, client, monkeypatch, name, call
    ):
        """These methods build their own request, so each must keep its own ``raise_for_status``."""
        _Recorder().install(monkeypatch, responses=[_FakeResponse(status_code=500)])

        with pytest.raises(httpx.HTTPStatusError):
            await call(client)


class TestLoadLoraAdapterFromTensors:
    """The two transports are mutually exclusive and each shapes the payload differently."""

    async def test_supplying_both_transports_is_rejected(self, client, recorder):
        """An ambiguous call must fail loudly instead of silently dropping one transport."""
        with pytest.raises(ValueError, match="exactly one of"):
            await client.load_lora_adapter_from_tensors(
                lora_name="l",
                config_dict={"r": 8},
                serialized_tensors="blob",
                serialized_named_tensors=["t"],
            )

        assert recorder.calls == []

    async def test_supplying_neither_transport_is_rejected(self, client, recorder):
        """A call with no weights at all must fail before it reaches the server."""
        with pytest.raises(ValueError, match="exactly one of"):
            await client.load_lora_adapter_from_tensors(lora_name="l", config_dict={"r": 8})

        assert recorder.calls == []

    async def test_the_serialized_tensors_transport_forwards_every_optional_field(self, client, recorder):
        """The whole-adapter transport plus all optional fields must reach the wire verbatim."""
        await client.load_lora_adapter_from_tensors(
            lora_name="l",
            config_dict={"r": 8},
            serialized_tensors="blob",
            load_format="direct",
            pinned=True,
            added_tokens_config={"tok": 1},
            upsert=True,
            expected_checksums={"w": "abc"},
        )

        assert recorder.calls == [
            (
                "post",
                f"{SERVER_URL}/load_lora_adapter_from_tensors",
                {
                    "json": {
                        "lora_name": "l",
                        "config_dict": {"r": 8},
                        "pinned": True,
                        "serialized_tensors": "blob",
                        "upsert": True,
                        "load_format": "direct",
                        "added_tokens_config": {"tok": 1},
                        "expected_checksums": {"w": "abc"},
                    }
                },
            )
        ]


class TestLoadLoraAdapterFromDistributed:
    async def test_load_lora_adapter_from_distributed_forwards_non_default_adapter_options(self, client, recorder):
        """``pinned``, ``upsert`` and ``added_tokens_config`` all belong in the distributed payload."""
        await client.load_lora_adapter_from_distributed(
            lora_name="l",
            config_dict={"r": 8},
            names=["w"],
            dtypes=["torch.bfloat16"],
            shapes=[[1]],
            group_name="g",
            pinned=True,
            added_tokens_config={"tok": 1},
            upsert=True,
        )

        assert recorder.calls[0][2]["json"] == {
            "lora_name": "l",
            "config_dict": {"r": 8},
            "names": ["w"],
            "dtypes": ["bfloat16"],
            "shapes": [[1]],
            "group_name": "g",
            "pinned": True,
            "upsert": True,
            "added_tokens_config": {"tok": 1},
        }


class TestFlushCacheTimeoutMessage:
    """The timeout is the only diagnostic the caller gets, so it must name the last failure."""

    async def test_it_reports_the_last_non_200_response_body(self, client, monkeypatch):
        """A 400 carries the reason (pending requests) in its body, not in an exception."""
        monkeypatch.setattr(asyncio, "sleep", _noop_sleep)
        _Recorder().install(
            monkeypatch, responses=[_FakeResponse(status_code=400, text="3 requests pending") for _ in range(60)]
        )

        with pytest.raises(TimeoutError, match="3 requests pending"):
            await client.flush_cache()

    async def test_it_reports_the_last_exception_message(self, client, monkeypatch):
        """When the server is unreachable the exception text is the only clue available."""

        class _AlwaysRaising:
            async def get(self, url, **kwargs):
                raise httpx.ConnectError("connection refused")

        monkeypatch.setattr(GeneralHttpClientProvider, "client", lambda: _AlwaysRaising())
        monkeypatch.setattr(asyncio, "sleep", _noop_sleep)

        with pytest.raises(TimeoutError, match="connection refused"):
            await client.flush_cache()


class TestGetWeightVersion:
    """The client probes the new endpoint first and only then the legacy one."""

    async def test_it_prefers_model_info_and_never_touches_the_legacy_endpoint(self, client, monkeypatch):
        """A modern sglang answers /model_info, so the legacy call must not be made at all."""
        rec = _Recorder()
        rec.install(monkeypatch, responses=[_FakeResponse(payload={"weight_version": "v9"})])

        assert await client.get_weight_version() == "v9"
        assert [url for _verb, url, _kwargs in rec.calls] == [f"{SERVER_URL}/model_info"]

    async def test_it_raises_when_neither_endpoint_answers(self, client, monkeypatch):
        """Returning ``None`` here would silently mark every weight version as unknown."""
        rec = _Recorder()
        rec.install(monkeypatch, responses=[_FakeResponse(status_code=404), _FakeResponse(status_code=404)])

        with pytest.raises(httpx.HTTPStatusError):
            await client.get_weight_version()

        assert [url for _verb, url, _kwargs in rec.calls] == [
            f"{SERVER_URL}/model_info",
            f"{SERVER_URL}/get_weight_version",
        ]


class TestWeightControlPayloads:
    """Optional weight-control fields must be included when given and omitted when not."""

    async def test_tensor_update_forwards_a_weight_version_and_a_non_default_selector(self, client, recorder):
        """Multi-model engines address one submodel at a time and stamp the resulting version."""
        await client.update_weights_from_tensor(
            serialized_named_tensors=["a"], weight_version="run-7", selector="draft"
        )

        assert recorder.calls[0][2]["json"] == {
            "serialized_named_tensors": ["a"],
            "load_format": None,
            "flush_cache": False,
            "selector": "draft",
            "weight_version": "run-7",
        }

    async def test_distributed_update_forwards_a_weight_version_and_a_non_default_selector(self, client, recorder):
        """The distributed path carries the same optional fields as the tensor path."""
        await client.update_weights_from_distributed(
            names=["w"],
            dtypes=["torch.bfloat16"],
            shapes=[[1]],
            group_name="g",
            weight_version="run-7",
            selector="draft",
        )

        assert recorder.calls[0][2]["json"] == {
            "names": ["w"],
            "dtypes": ["bfloat16"],
            "shapes": [[1]],
            "group_name": "g",
            "flush_cache": False,
            "selector": "draft",
            "weight_version": "run-7",
        }

    async def test_disk_update_omits_the_optional_fields_when_not_given(self, client, recorder):
        """sglang rejects an explicit ``load_format=None``, so the key must be absent, not null."""
        await client.update_weights_from_disk(model_path="/ckpt")

        assert recorder.calls[0][2]["json"] == {"model_path": "/ckpt"}

    async def test_check_weights_omits_the_skip_list_when_not_given(self, client, recorder):
        """Without a skip list the request must not carry an empty ``skip_tensor_list``."""
        await client.check_weights(action="check", allow_quant_error=True, selector="draft")

        assert recorder.calls[0][2]["json"] == {
            "action": "check",
            "allow_quant_error": True,
            "selector": "draft",
        }

    async def test_begin_weight_update_forwards_a_custom_selector(self, client, recorder):
        """The session must open on the same submodel the update will write to."""
        await client.begin_weight_update(selector="draft")

        assert recorder.calls[0][2]["json"] == {"selector": "draft"}

    async def test_end_weight_update_posts_an_empty_payload(self, client, recorder):
        """Closing the session takes no selector: post-processing covers the full model."""
        await client.end_weight_update()

        assert recorder.calls[0][2]["json"] == {}

    async def test_update_weight_version_keeps_in_flight_requests_by_default(self, client, recorder):
        """main never aborted generation on a version bump, and the broadcast path relies on that default."""
        await client.update_weight_version("run-7")

        assert recorder.calls[0][2]["json"] == {"new_version": "run-7", "abort_all_requests": False}

    async def test_update_weight_version_can_abort_all_requests_explicitly(self, client, recorder):
        """An explicit opt-in must still reach the server for callers that do want the abort."""
        await client.update_weight_version("run-7", abort_all_requests=True)

        assert recorder.calls[0][2]["json"] == {"new_version": "run-7", "abort_all_requests": True}

    async def test_release_memory_occupation_forwards_its_tags(self, client, recorder):
        """Multi-stage offload releases weights and kv_cache separately."""
        await client.release_memory_occupation(tags=["weights", "kv_cache"])

        assert recorder.calls[1][2]["json"] == {"tags": ["weights", "kv_cache"]}


class TestStartProfile:
    async def test_start_profile_forwards_every_non_default_profile_option(self, client, recorder):
        """Every profiling knob is optional, so each one must survive the trip to the wire."""
        await client.start_profile(
            output_dir="/out",
            start_step=5,
            num_steps=3,
            activities=["CPU", "GPU"],
            profile_by_stage=True,
            with_stack=True,
            record_shapes=True,
        )

        assert recorder.calls[0][2]["json"] == {
            "output_dir": "/out",
            "start_step": 5,
            "num_steps": 3,
            "activities": ["CPU", "GPU"],
            "profile_by_stage": True,
            "with_stack": True,
            "record_shapes": True,
        }


async def _noop_sleep(seconds):
    return None


class TestABodylessSuccess:
    async def test_a_success_carrying_no_body_answers_none_instead_of_raising(self, client, monkeypatch):
        """The engine answers some endpoints with 200 and an empty body, and parsing that as json raises."""
        _Recorder().install(monkeypatch, responses=[_FakeResponse(body=b"")])

        assert await client.abort_all_requests() is None

    async def test_a_success_carrying_a_body_is_still_parsed(self, client, monkeypatch):
        """Every other endpoint answers with json the callers read, so the empty-body case must not swallow it."""
        _Recorder().install(monkeypatch, responses=[_FakeResponse(payload={"ok": True})])

        assert await client.abort_all_requests() == {"ok": True}

    async def test_aborting_every_request_reaches_the_engines_abort_endpoint(self, client, recorder):
        """A take-over aborts what the previous script left generating, and no other endpoint does that."""
        await client.abort_all_requests()

        assert recorder.calls[0][0] == "post"
        assert recorder.calls[0][1] == f"{SERVER_URL}/abort_request"
        assert recorder.calls[0][2]["json"] == {"abort_all": True}

    async def test_aborting_every_request_carries_the_budget_the_caller_gives(self, client, recorder):
        """A wedged tokenizer manager is exactly what a take-over aborts, and this client waits forever by default."""
        await client.abort_all_requests(timeout=12.0)

        assert recorder.calls[0][2]["timeout"] == 12.0

    async def test_aborting_every_request_without_a_budget_leaves_the_client_default(self, client, recorder):
        """Every other endpoint of this client is unbounded, so naming a timeout stays the caller's choice."""
        await client.abort_all_requests()

        assert "timeout" not in recorder.calls[0][2]
