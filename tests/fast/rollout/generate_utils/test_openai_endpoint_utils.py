"""Tests for OpenAIEndpointTracer (session-server client side).

The sample-assembly and TITO multi-turn merge tests live in
tests/fast/rollout/session/test_samples.py (assembly) and
test_samples_codec.py (wire codec), next to the functions.
The collect_samples tests here lock the client's HTTP behavior deltas vs the
old collect_records path: single POST with no retries, non-2xx raises with the
body text, timeout raises (instead of silently ABORTing), and the session
DELETE is attempted on every path.
"""

import asyncio
from types import SimpleNamespace

import pytest

import miles.utils.http_utils as http_utils
from miles.rollout.generate_utils.openai_endpoint_utils import OpenAIEndpointTracer
from miles.rollout.session.samples.codec import encode_samples_reply
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
        session_server_ip="127.0.0.1",
        session_server_ports=[12345],
        session_server_instance_ids={12345: "server-instance-123"},
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

    args = SimpleNamespace(session_server_ip="127.0.0.1", session_server_ports=[12345])
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

    async def fake_post_bytes(url, payload, *, timeout):
        calls.append(("post_bytes", url))
        return encode_samples_reply([], {}, "no_records")

    monkeypatch.setattr("miles.rollout.generate_utils.openai_endpoint_utils.post", fake_post)
    monkeypatch.setattr("miles.rollout.generate_utils.openai_endpoint_utils.post_bytes_no_retry", fake_post_bytes)

    ports = [12345, 12346, 12347, 12348]
    args = SimpleNamespace(session_server_ip="127.0.0.1", session_server_ports=ports)

    chosen_ports = set()
    for _ in range(32):
        calls.clear()
        tracer = await OpenAIEndpointTracer.create(args)
        port = int(tracer.session_server_id.rsplit(":", 1)[1])
        assert port in ports
        chosen_ports.add(port)

        await tracer.collect_samples(Sample(), max_seq_len=None)
        prefix = f"http://127.0.0.1:{port}"
        assert [url for _, url in calls] == [
            f"{prefix}/sessions",
            f"{tracer.base_url}/samples",
            tracer.base_url,
        ]
        assert tracer.base_url.startswith(f"{prefix}/sessions/")

    # 32 uniform picks over 4 ports miss a given port with p = (3/4)^32 ≈ 1e-4.
    assert len(chosen_ports) > 1


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
    return encode_samples_reply([sample], {"max_trim_tokens": 1}, None)


class _CollectCalls:
    """Patches the two HTTP primitives collect_samples uses and records order."""

    def __init__(self, monkeypatch, *, post_outcome, delete_outcome=None):
        self.calls: list[str] = []

        async def fake_post_bytes(url, payload, *, timeout):
            self.calls.append(f"POST {url}")
            assert payload == {"max_seq_len": 7}
            if isinstance(post_outcome, Exception):
                raise post_outcome
            return post_outcome

        async def fake_post(url, payload, action="post"):
            assert action == "delete"
            self.calls.append(f"DELETE {url}")
            if isinstance(delete_outcome, Exception):
                raise delete_outcome
            return {}

        monkeypatch.setattr("miles.rollout.generate_utils.openai_endpoint_utils.post_bytes_no_retry", fake_post_bytes)
        monkeypatch.setattr("miles.rollout.generate_utils.openai_endpoint_utils.post", fake_post)


@pytest.mark.asyncio
async def test_collect_samples_single_post_then_delete(monkeypatch):
    calls = _CollectCalls(monkeypatch, post_outcome=_computed_reply_payload())
    result = await _tracer().collect_samples(Sample(), max_seq_len=7)

    assert calls.calls == [
        "POST http://127.0.0.1:12345/sessions/sid-1/samples",
        "DELETE http://127.0.0.1:12345/sessions/sid-1",
    ]
    (sample,) = result.samples
    assert sample.tokens == [1, 2, 10] and sample.status == Sample.Status.COMPLETED
    assert result.session_metadata == {"max_trim_tokens": 1}


@pytest.mark.asyncio
async def test_collect_samples_non_2xx_raises_with_body_and_still_deletes(monkeypatch):
    calls = _CollectCalls(monkeypatch, post_outcome=RuntimeError("422: trim_count 2 exceeds allowed=1"))
    with pytest.raises(RuntimeError, match="trim_count 2 exceeds allowed=1"):
        await _tracer().collect_samples(Sample(), max_seq_len=7)
    assert calls.calls[-1] == "DELETE http://127.0.0.1:12345/sessions/sid-1"


@pytest.mark.asyncio
async def test_collect_samples_timeout_raises_and_still_deletes(monkeypatch):
    # The old collect_records swallowed the timeout and returned empty records
    # (silently ABORTing the sample); the samples path must raise it.
    calls = _CollectCalls(monkeypatch, post_outcome=asyncio.TimeoutError())
    with pytest.raises(asyncio.TimeoutError):
        await _tracer().collect_samples(Sample(), max_seq_len=7)
    assert calls.calls[-1] == "DELETE http://127.0.0.1:12345/sessions/sid-1"


@pytest.mark.asyncio
async def test_collect_samples_delete_failure_is_tolerated(monkeypatch):
    _CollectCalls(monkeypatch, post_outcome=_computed_reply_payload(), delete_outcome=RuntimeError("delete boom"))
    result = await _tracer().collect_samples(Sample(), max_seq_len=7)
    assert len(result.samples) == 1


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
