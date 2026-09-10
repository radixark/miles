"""Tests for OpenAIEndpointTracer (session-server client side).

The sample-assembly and TITO multi-turn merge tests live in
tests/fast/rollout/session/test_samples.py (assembly) and
test_samples_codec.py (wire codec), next to the functions.
The collect_samples tests here lock the client's HTTP behavior deltas vs the
old collect_records path: single POST with no retries, non-2xx raises with the
body text, timeout raises (instead of silently ABORTing), and the session
DELETE is scheduled only after the caller accepts a decoded snapshot.
"""

import asyncio
from types import SimpleNamespace

import httpx
import pytest

import miles.rollout.generate_utils.openai_endpoint_utils as endpoint
import miles.utils.http_utils as http_utils
from miles.rollout.generate_utils.openai_endpoint_utils import OpenAIEndpointTracer, SessionCollectError
from miles.rollout.session.samples.codec import COMPUTED_FIELDS, COMPUTED_FIELDS_V2, encode_samples
from miles.rollout.session.types import SESSION_GENERATION_HEADER, SESSION_RECORD_ERROR_CODE
from miles.utils.http_utils import post_bytes_no_retry
from miles.utils.types import Sample


@pytest.mark.asyncio
async def test_create_reads_session_server_instance_id_from_args(monkeypatch):
    calls: list[tuple[str, str]] = []

    async def fake_post(url: str, payload: dict, action: str = "post"):
        calls.append((action, url))
        assert action == "post"
        assert url == "http://127.0.0.1:12345/sessions"
        return {"session_id": "session-123"}

    monkeypatch.setattr("miles.rollout.generate_utils.openai_endpoint_utils.post", fake_post)

    args = SimpleNamespace(
        session_server_addrs=["127.0.0.1:12345"],
        session_server_instance_ids={"127.0.0.1:12345": "server-instance-123"},
    )
    tracer = await OpenAIEndpointTracer.create(args)

    assert tracer.base_url == "http://127.0.0.1:12345/sessions/session-123"
    assert tracer.session_server_id == "127.0.0.1:12345"
    assert tracer.session_server_instance_id == "server-instance-123"
    # No /health probe: the id is read locally, create() issues only the POST.
    assert calls == [("post", "http://127.0.0.1:12345/sessions")]


@pytest.mark.asyncio
async def test_create_without_instance_id_on_args(monkeypatch):
    async def fake_post(url: str, payload: dict, action: str = "post"):
        return {"session_id": "session-123"}

    monkeypatch.setattr("miles.rollout.generate_utils.openai_endpoint_utils.post", fake_post)

    args = SimpleNamespace(session_server_addrs=["127.0.0.1:12345"])
    tracer = await OpenAIEndpointTracer.create(args)

    assert tracer.session_server_instance_id is None


@pytest.mark.asyncio
async def test_create_distributes_sessions_across_port_range(monkeypatch):
    """With a multi-port range, sessions land on more than one instance, and every
    request of a session (create, samples POST, DELETE) hits the port chosen
    at create time — the URL is the router."""
    calls: list[tuple[str, str]] = []

    async def fake_post(url: str, payload: dict, action: str = "post"):
        calls.append((action, url))
        if action == "post" and url.endswith("/sessions"):
            return {"session_id": f"session-{len(calls)}"}
        return {}

    async def fake_request(url, payload, *, method, timeout, headers=None):
        calls.append((method, url))
        return (
            httpx.Response(204)
            if method == "DELETE"
            else httpx.Response(
                200, content=encode_samples([], {}, "no_records"), headers={SESSION_GENERATION_HEADER: "1"}
            )
        )

    monkeypatch.setattr("miles.rollout.generate_utils.openai_endpoint_utils.post", fake_post)
    monkeypatch.setattr(endpoint, "request_no_retry", fake_request)

    ports = [12345, 12346, 12347, 12348]
    args = SimpleNamespace(session_server_addrs=[f"127.0.0.1:{port}" for port in ports])

    chosen_ports = set()
    for _ in range(32):
        calls.clear()
        tracer = await OpenAIEndpointTracer.create(args)
        port = int(tracer.session_server_id.rsplit(":", 1)[1])
        assert port in ports
        chosen_ports.add(port)

        collected = await tracer.collect_samples(Sample(), max_seq_len=None)
        tracer.schedule_cleanup(collected.generation)
        await asyncio.gather(*endpoint._cleanup_tasks)
        prefix = f"http://127.0.0.1:{port}"
        assert [url for _, url in calls] == [
            f"{prefix}/sessions",
            f"{tracer.base_url}/samples",
            tracer.base_url,
        ]
        assert tracer.base_url.startswith(f"{prefix}/sessions/")

    # 32 uniform picks over 4 ports miss a given port with p = (3/4)^32 ≈ 1e-4.
    assert len(chosen_ports) > 1


class TestOpenAIEndpointTracerCreate:
    @pytest.mark.asyncio
    async def test_create_routes_to_selected_address_across_multiple_hosts(self, monkeypatch):
        """create() sends the session POST to the whole selected host:port and reads that host's instance id, even when both hosts share a port."""
        posted: list[str] = []

        async def fake_post(url: str, payload: dict, action: str = "post"):
            posted.append(url)
            return {"session_id": "session-abc"}

        monkeypatch.setattr("miles.rollout.generate_utils.openai_endpoint_utils.post", fake_post)
        monkeypatch.setattr("miles.rollout.generate_utils.openai_endpoint_utils.random.choice", lambda addrs: addrs[1])

        args = SimpleNamespace(
            session_server_addrs=["10.0.0.1:5005", "10.0.0.2:5005"],
            session_server_instance_ids={"10.0.0.1:5005": "instance-a", "10.0.0.2:5005": "instance-b"},
        )
        tracer = await OpenAIEndpointTracer.create(args)

        assert posted == ["http://10.0.0.2:5005/sessions"]
        assert tracer.session_server_id == "10.0.0.2:5005"
        assert tracer.base_url == "http://10.0.0.2:5005/sessions/session-abc"
        assert tracer.session_server_instance_id == "instance-b"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("addrs_kwargs", [{}, {"session_server_addrs": None}, {"session_server_addrs": []}])
    async def test_create_without_session_server_addrs_raises_before_post(self, monkeypatch, addrs_kwargs):
        """create() raises a RuntimeError pointing at --use-session-server and issues no HTTP request when session_server_addrs is absent, null or empty."""
        posted: list[str] = []

        async def fake_post(url: str, payload: dict, action: str = "post"):
            posted.append(url)
            return {"session_id": "session-abc"}

        monkeypatch.setattr("miles.rollout.generate_utils.openai_endpoint_utils.post", fake_post)

        with pytest.raises(RuntimeError, match="session_server_addrs is not set"):
            await OpenAIEndpointTracer.create(SimpleNamespace(**addrs_kwargs))

        assert posted == []


# ── collect_samples client behavior ──


def _tracer() -> OpenAIEndpointTracer:
    return OpenAIEndpointTracer(router_url="http://127.0.0.1:12345", session_id="sid-1")


def _computed_reply_payload() -> bytes:
    sample = Sample()
    sample.tokens = [1, 2, 10]
    sample.response = "r"
    sample.response_length = 1
    sample.loss_mask = [1]
    sample.rollout_log_probs = [-0.5]
    sample.status = Sample.Status.COMPLETED
    return encode_samples([sample], {"max_trim_tokens": 1}, None)


class _CollectCalls:
    def __init__(self, monkeypatch, *, post_outcome, delete_outcome=None):
        self.calls = []
        self.headers = []

        async def request(url, payload, *, method, timeout, headers=None):
            self.calls.append(f"{method} {url}")
            self.headers.append(headers)
            if method == "POST":
                assert payload == {"max_seq_len": 7}
                if isinstance(post_outcome, BaseException):
                    raise post_outcome
                return (
                    post_outcome
                    if isinstance(post_outcome, httpx.Response)
                    else httpx.Response(200, content=post_outcome, headers={SESSION_GENERATION_HEADER: "17"})
                )
            if isinstance(delete_outcome, Exception):
                raise delete_outcome
            return httpx.Response(delete_outcome or 204)

        monkeypatch.setattr(endpoint, "request_no_retry", request)


@pytest.mark.asyncio
async def test_collect_decodes_then_caller_schedules_guarded_cleanup(monkeypatch):
    calls = _CollectCalls(monkeypatch, post_outcome=_computed_reply_payload())
    tracer = _tracer()
    result = await tracer.collect_samples(Sample(), max_seq_len=7)
    assert len(calls.calls) == 1
    (sample,) = result.reply.samples
    assert sample.tokens == [1, 2, 10] and sample.status == Sample.Status.COMPLETED
    assert result.reply.session_metadata == {"max_trim_tokens": 1}
    tracer.schedule_cleanup(result.generation)
    assert len(calls.calls) == 1
    await asyncio.gather(*endpoint._cleanup_tasks)
    assert calls.calls[-1] == "DELETE http://127.0.0.1:12345/sessions/sid-1"
    assert calls.headers[-1] == {SESSION_GENERATION_HEADER: "17"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "outcome,exception",
    [
        (httpx.Response(422, text="trim_count exceeds allowed"), RuntimeError),
        (httpx.Response(503, json={"error": "other failure"}), RuntimeError),
        (httpx.Response(503, text="not JSON"), RuntimeError),
        (httpx.Response(503, json={"error": {"code": SESSION_RECORD_ERROR_CODE}}), SessionCollectError),
        (TimeoutError(), TimeoutError),
        (httpx.ReadError("broken"), httpx.ReadError),
        (asyncio.CancelledError(), asyncio.CancelledError),
        (b"bad payload", Exception),
    ],
)
async def test_collect_failure_never_deletes(monkeypatch, outcome, exception):
    calls = _CollectCalls(monkeypatch, post_outcome=outcome)
    with pytest.raises(exception) as raised:
        await _tracer().collect_samples(Sample(), max_seq_len=7)
    if exception is RuntimeError:
        assert type(raised.value) is RuntimeError
    assert len(calls.calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", [204, 404, 412, 500, RuntimeError("delete boom"), TimeoutError()])
async def test_cleanup_outcomes_do_not_change_collected_output(monkeypatch, outcome):
    calls = _CollectCalls(monkeypatch, post_outcome=_computed_reply_payload(), delete_outcome=outcome)
    tracer = _tracer()
    result = await tracer.collect_samples(Sample(), max_seq_len=7)
    tracer.schedule_cleanup(result.generation)
    await asyncio.gather(*endpoint._cleanup_tasks)
    assert len(result.reply.samples) == 1 and len(calls.calls) == 2
    assert not endpoint._cleanup_tasks


@pytest.mark.asyncio
@pytest.mark.parametrize("header", [None, "bad", "0", "-1"])
async def test_missing_cleanup_guard_never_sends_delete(monkeypatch, header):
    response = httpx.Response(
        200, content=_computed_reply_payload(), headers={} if header is None else {SESSION_GENERATION_HEADER: header}
    )
    calls = _CollectCalls(monkeypatch, post_outcome=response)
    tracer = _tracer()
    collected = await tracer.collect_samples(Sample(), max_seq_len=7)
    tracer.schedule_cleanup(collected.generation)
    assert collected.generation is None and len(calls.calls) == 1


@pytest.mark.asyncio
async def test_cleanup_saturation_and_scheduling_failure_are_nonthrowing(monkeypatch):
    calls = _CollectCalls(monkeypatch, post_outcome=_computed_reply_payload())
    monkeypatch.setattr(endpoint, "_MAX_CLEANUPS", 0)
    _tracer().schedule_cleanup(17)
    assert not endpoint._cleanup_tasks
    monkeypatch.setattr(endpoint, "_MAX_CLEANUPS", 32)

    def failed(coroutine):
        raise RuntimeError("loop unavailable")

    monkeypatch.setattr(asyncio, "create_task", failed)
    _tracer().schedule_cleanup(17)
    assert not endpoint._cleanup_tasks and not calls.calls


# ── post_bytes_no_retry primitive ──


class _FakeResponse:
    def __init__(self, status_code: int, content: bytes = b"", text: str = ""):
        self.status_code = status_code
        self.content = content
        self.text = text


class _FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.post_count = 0

    async def post(self, url, json=None):
        self.post_count += 1
        outcome = self.responses.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@pytest.mark.asyncio
async def test_post_bytes_no_retry_returns_raw_bytes(monkeypatch):
    client = _FakeClient([_FakeResponse(200, content=b"\x00\x01binary")])
    monkeypatch.setattr(http_utils, "_http_client", client)
    assert await post_bytes_no_retry("http://x/samples", {}, timeout=5) == b"\x00\x01binary"
    assert client.post_count == 1


@pytest.mark.asyncio
async def test_post_bytes_no_retry_does_not_retry_and_carries_body(monkeypatch):
    # Two queued outcomes; a retrying client would consume both. It must not.
    client = _FakeClient([_FakeResponse(422, text="cursor 3 != len(accumulated_token_ids) 4"), RuntimeError("late")])
    monkeypatch.setattr(http_utils, "_http_client", client)
    with pytest.raises(RuntimeError, match="422.*cursor 3"):
        await post_bytes_no_retry("http://x/samples", {}, timeout=5)
    assert client.post_count == 1


@pytest.mark.asyncio
async def test_post_bytes_no_retry_transport_error_propagates_once(monkeypatch):
    client = _FakeClient([ConnectionError("boom"), RuntimeError("late")])
    monkeypatch.setattr(http_utils, "_http_client", client)
    with pytest.raises(ConnectionError, match="boom"):
        await post_bytes_no_retry("http://x/samples", {}, timeout=5)
    assert client.post_count == 1


# ── v2 wire (--use-session-server v2): metadata channel + extended fields ──


@pytest.mark.asyncio
async def test_collect_samples_v2_payload_carries_metadata_and_decodes_extras(monkeypatch):
    """v2 pin: the collect body gains the "metadata" key only when the caller
    passes agent metadata, and the v2 field tuple overlays reward + merged
    metadata; the v1 pin above (`payload == {"max_seq_len": 7}`) stays."""
    from miles.rollout.session.samples.codec import COMPUTED_FIELDS_V2

    sample = Sample()
    sample.tokens = [1, 2, 10]
    sample.response = "r"
    sample.response_length = 1
    sample.loss_mask = [1]
    sample.rollout_log_probs = [-0.5]
    sample.status = Sample.Status.COMPLETED
    sample.reward = 0.75
    sample.metadata = {"leaf": {"node_id": 1}}
    payload = encode_samples([sample], {"max_trim_tokens": 1}, None, fields=COMPUTED_FIELDS_V2)

    seen = []

    async def fake_request(url, body, *, method, timeout, headers=None):
        seen.append(body)
        return httpx.Response(200, content=payload, headers={SESSION_GENERATION_HEADER: "2"})

    async def fake_post(url, body, action="post"):
        assert action == "delete"
        return {}

    monkeypatch.setattr(endpoint, "request_no_retry", fake_request)
    monkeypatch.setattr("miles.rollout.generate_utils.openai_endpoint_utils.post", fake_post)

    tracer = OpenAIEndpointTracer(
        router_url="http://127.0.0.1:12345", session_id="sid-1", samples_wire_fields=COMPUTED_FIELDS_V2
    )
    input_sample = Sample()
    input_sample.metadata = {"env": "keep-me"}
    result = await tracer.collect_samples(input_sample, max_seq_len=7, agent_metadata={"reward": 0.75})

    assert seen == [{"max_seq_len": 7, "metadata": {"reward": 0.75}}]
    (decoded,) = result.reply.samples
    assert decoded.reward == 0.75
    assert decoded.metadata == {"env": "keep-me", "leaf": {"node_id": 1}}


@pytest.mark.asyncio
async def test_create_selects_wire_fields_by_session_server_version(monkeypatch):
    async def fake_post(url: str, payload: dict, action: str = "post"):
        return {"session_id": "sid-x"}

    monkeypatch.setattr("miles.rollout.generate_utils.openai_endpoint_utils.post", fake_post)

    def args(version):
        return SimpleNamespace(session_server_addrs=["127.0.0.1:7000"], use_session_server=version)

    assert (await OpenAIEndpointTracer.create(args(True))).samples_wire_fields == COMPUTED_FIELDS
    assert (await OpenAIEndpointTracer.create(args("v2"))).samples_wire_fields == COMPUTED_FIELDS_V2


@pytest.mark.asyncio
async def test_request_no_retry_preserves_response_and_bounds_total_time(monkeypatch):
    seen = []

    async def handle(request):
        seen.append(request)
        if request.method == "DELETE":
            await asyncio.Event().wait()
        return httpx.Response(503, content=b"disk", headers={"x-test": "present"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        monkeypatch.setattr(http_utils, "_http_client", client)
        response = await http_utils.request_no_retry("http://x/samples", {}, method="POST", timeout=1)
        assert response.status_code == 503 and response.content == b"disk" and response.headers["x-test"] == "present"
        with pytest.raises(TimeoutError):
            await http_utils.request_no_retry(
                "http://x/session", {}, method="DELETE", timeout=0.01, headers={SESSION_GENERATION_HEADER: "17"}
            )
    assert [r.method for r in seen] == ["POST", "DELETE"]
    assert seen[-1].headers[SESSION_GENERATION_HEADER] == "17"


@pytest.mark.asyncio
async def test_slow_cleanup_is_bounded_and_does_not_block_scheduling(monkeypatch):
    entered, release = asyncio.Event(), asyncio.Event()
    calls = []

    async def slow(url, body, *, method, timeout, headers=None):
        calls.append(url)
        entered.set()
        await release.wait()
        return httpx.Response(204)

    monkeypatch.setattr(endpoint, "request_no_retry", slow)
    monkeypatch.setattr(endpoint, "_MAX_CLEANUPS", 1)
    _tracer().schedule_cleanup(17)
    await entered.wait()
    _tracer().schedule_cleanup(18)
    assert len(endpoint._cleanup_tasks) == 1 and len(calls) == 1
    release.set()
    await asyncio.gather(*endpoint._cleanup_tasks)
    assert not endpoint._cleanup_tasks
