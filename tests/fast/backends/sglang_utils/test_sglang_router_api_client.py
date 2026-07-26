import httpx
import pytest

sglang_router = pytest.importorskip("sglang_router")

from miles.backends.sglang_utils.sglang_router_api_client import SGLangRouterApiClient  # noqa: E402
from miles.utils.http_utils import GeneralHttpClientProvider  # noqa: E402

ROUTER_URL = "http://router-host:9000"
WORKER_URL = "http://fake-host:1234"


class _FakeResponse:
    def __init__(self, payload: dict | None = None, status_code: int = 200):
        self._payload = payload if payload is not None else {"ok": True}
        self.status_code = status_code

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
    return SGLangRouterApiClient(router_url=ROUTER_URL)


async def test_add_worker_uses_the_query_string_api_when_legacy(client, recorder):
    """Routers <= 0.2.1 and the miles router only understand /add_worker?url=."""
    await client.add_worker(worker_url=WORKER_URL, worker_type="regular", use_legacy_api=True)

    assert recorder.calls == [("post", f"{ROUTER_URL}/add_worker?url={WORKER_URL}", {})]


async def test_add_worker_rejects_pd_disaggregation_on_the_legacy_api(client, recorder):
    """The legacy API has no worker_type concept, so prefill/decode workers must be refused."""
    with pytest.raises(AssertionError, match="pd disaggregation is not supported"):
        await client.add_worker(worker_url=WORKER_URL, worker_type="prefill", use_legacy_api=True)


async def test_add_worker_posts_the_worker_payload_on_the_modern_api(client, recorder):
    """Modern routers take a JSON body on /workers."""
    await client.add_worker(worker_url=WORKER_URL, worker_type="regular", use_legacy_api=False)

    assert len(recorder.calls) == 1
    verb, url, kwargs = recorder.calls[0]
    assert (verb, url) == ("post", f"{ROUTER_URL}/workers")
    assert kwargs["json"] == {"url": WORKER_URL, "worker_type": "regular"}


async def test_add_worker_includes_the_bootstrap_port_for_prefill_workers(client, recorder):
    """PD disaggregation needs the prefill worker's bootstrap port registered with the router."""
    await client.add_worker(worker_url=WORKER_URL, worker_type="prefill", use_legacy_api=False, bootstrap_port=8998)

    assert len(recorder.calls) == 1
    assert recorder.calls[0][2]["json"] == {
        "url": WORKER_URL,
        "worker_type": "prefill",
        "bootstrap_port": 8998,
    }


async def test_remove_worker_uses_the_query_string_api_when_legacy(client, recorder):
    """Legacy routers unregister via /remove_worker?url=."""
    await client.remove_worker(worker_url=WORKER_URL, use_legacy_api=True)

    assert recorder.calls == [("post", f"{ROUTER_URL}/remove_worker?url={WORKER_URL}", {})]


async def test_remove_worker_deletes_by_url_on_pre_0_3_routers(client, recorder, monkeypatch):
    """Routers in [0.2.2, 0.3.0) address workers by percent-encoded url."""
    monkeypatch.setattr(sglang_router, "__version__", "0.2.5")

    await client.remove_worker(worker_url=WORKER_URL, use_legacy_api=False)

    assert recorder.calls == [("delete", f"{ROUTER_URL}/workers/http%3A%2F%2Ffake-host%3A1234", {})]


async def test_remove_worker_resolves_the_worker_id_on_modern_routers(client, monkeypatch):
    """Routers >= 0.3.0 address workers by id, so the url must be resolved first."""
    monkeypatch.setattr(sglang_router, "__version__", "0.3.1")
    rec = _Recorder()
    rec.install(monkeypatch, responses=[_FakeResponse({"workers": [{"url": WORKER_URL, "id": "w-7"}]})])

    await client.remove_worker(worker_url=WORKER_URL, use_legacy_api=False)

    assert [(verb, url) for verb, url, _kwargs in rec.calls] == [
        ("get", f"{ROUTER_URL}/workers"),
        ("delete", f"{ROUTER_URL}/workers/w-7"),
    ]


async def test_remove_worker_tolerates_an_unknown_worker(client, monkeypatch):
    """Shutdown must not fail when the router no longer knows the worker."""
    monkeypatch.setattr(sglang_router, "__version__", "0.3.1")
    rec = _Recorder()
    rec.install(monkeypatch, responses=[_FakeResponse({"workers": []})])

    await client.remove_worker(worker_url=WORKER_URL, use_legacy_api=False)

    assert [verb for verb, _url, _kwargs in rec.calls] == ["get"]


async def test_add_worker_propagates_router_errors(client, monkeypatch):
    """A router that rejects the registration must surface, not be swallowed."""
    rec = _Recorder()
    rec.install(monkeypatch, responses=[_FakeResponse(status_code=500)])

    with pytest.raises(httpx.HTTPStatusError):
        await client.add_worker(worker_url=WORKER_URL, worker_type="regular", use_legacy_api=True)


async def test_remove_worker_propagates_router_errors(client, monkeypatch):
    """Unregistration errors must surface on the legacy path too."""
    rec = _Recorder()
    rec.install(monkeypatch, responses=[_FakeResponse(status_code=500)])

    with pytest.raises(httpx.HTTPStatusError):
        await client.remove_worker(worker_url=WORKER_URL, use_legacy_api=True)


@pytest.mark.parametrize("worker_type", ["prefill", "decode"])
async def test_add_worker_rejects_pd_worker_types_before_any_request(client, monkeypatch, worker_type):
    """The legacy API has no worker_type, so a pd worker must be refused without registering it."""
    rec = _Recorder()
    rec.install(monkeypatch)

    with pytest.raises(AssertionError, match="pd disaggregation is not supported"):
        await client.add_worker(worker_url=WORKER_URL, worker_type=worker_type, use_legacy_api=True)

    assert rec.calls == []


@pytest.mark.parametrize("worker_type", ["regular", "decode"])
async def test_add_worker_omits_bootstrap_port_for_non_prefill_workers(client, recorder, worker_type):
    """Only a prefill worker exposes a bootstrap port for the decode side to dial."""
    await client.add_worker(worker_url=WORKER_URL, worker_type=worker_type, use_legacy_api=False, bootstrap_port=8998)

    assert len(recorder.calls) == 1
    assert recorder.calls[0][2]["json"] == {"url": WORKER_URL, "worker_type": worker_type}


@pytest.mark.parametrize(
    "version, expected_verb", [("0.2.2", "delete"), ("0.2.9", "delete"), ("0.3.0", "get"), ("0.3.1", "get")]
)
async def test_remove_worker_switches_api_exactly_at_0_3_0(client, monkeypatch, version, expected_verb):
    """0.3.0 is where workers stop being addressable by url and must be looked up by id."""
    monkeypatch.setattr(sglang_router, "__version__", version)
    rec = _Recorder()
    rec.install(monkeypatch, responses=[_FakeResponse({"workers": [{"url": WORKER_URL, "id": "w-1"}]})])

    await client.remove_worker(worker_url=WORKER_URL, use_legacy_api=False)

    assert rec.calls[0][0] == expected_verb


async def test_add_worker_propagates_a_failing_modern_response(client, monkeypatch):
    """A router rejecting the registration must surface, not be swallowed."""
    rec = _Recorder()
    rec.install(monkeypatch, responses=[_FakeResponse(status_code=500)])

    with pytest.raises(httpx.HTTPStatusError):
        await client.add_worker(worker_url=WORKER_URL, worker_type="regular", use_legacy_api=False)


async def test_remove_worker_propagates_a_failing_url_addressed_delete(client, monkeypatch):
    """Unregistration errors surface on the url-addressed path too."""
    monkeypatch.setattr(sglang_router, "__version__", "0.2.5")
    rec = _Recorder()
    rec.install(monkeypatch, responses=[_FakeResponse(status_code=500)])

    with pytest.raises(httpx.HTTPStatusError):
        await client.remove_worker(worker_url=WORKER_URL, use_legacy_api=False)


async def test_remove_worker_propagates_a_failing_id_addressed_delete(client, monkeypatch):
    """The id-addressed delete is outside the lookup's broad except, so its status must surface."""
    monkeypatch.setattr(sglang_router, "__version__", "0.3.1")
    rec = _Recorder()
    rec.install(
        monkeypatch,
        responses=[_FakeResponse({"workers": [{"url": WORKER_URL, "id": "w-1"}]}), _FakeResponse(status_code=500)],
    )

    with pytest.raises(httpx.HTTPStatusError):
        await client.remove_worker(worker_url=WORKER_URL, use_legacy_api=False)


async def test_remove_worker_warns_about_an_unknown_worker(client, monkeypatch, caplog):
    """Shutdown tolerates a worker the router forgot, but says so."""
    monkeypatch.setattr(sglang_router, "__version__", "0.3.1")
    rec = _Recorder()
    rec.install(monkeypatch, responses=[_FakeResponse({"workers": []})])

    with caplog.at_level("WARNING"):
        await client.remove_worker(worker_url=WORKER_URL, use_legacy_api=False)

    assert "not found in router" in caplog.text


async def test_remove_worker_tolerates_a_broken_worker_lookup(client, monkeypatch, caplog):
    """A router that cannot list its workers must not block the engine's shutdown."""
    monkeypatch.setattr(sglang_router, "__version__", "0.3.1")

    class _Refusing:
        async def get(self, url, **kwargs):
            raise httpx.ConnectError("router down")

    monkeypatch.setattr(GeneralHttpClientProvider, "client", lambda: _Refusing())

    with caplog.at_level("WARNING"):
        await client.remove_worker(worker_url=WORKER_URL, use_legacy_api=False)

    assert "Failed to fetch workers list" in caplog.text
