"""Tests for OpenAIEndpointTracer (session-server client side).

The sample-assembly and TITO multi-turn merge tests live in
tests/fast/rollout/session/test_samples.py, next to the functions.
"""

from types import SimpleNamespace

import pytest

from miles.rollout.generate_utils.openai_endpoint_utils import OpenAIEndpointTracer


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
    request of a session (create, chat, GET, DELETE) hits the port chosen
    at create time — the URL is the router."""
    calls: list[tuple[str, str]] = []

    async def fake_post(url: str, payload: dict, action: str = "post"):
        calls.append((action, url))
        if action == "post" and url.endswith("/sessions"):
            return {"session_id": f"session-{len(calls)}"}
        return {"session_id": url.rsplit("/", 1)[1], "records": [], "metadata": {}}

    monkeypatch.setattr("miles.rollout.generate_utils.openai_endpoint_utils.post", fake_post)

    ports = [12345, 12346, 12347, 12348]
    args = SimpleNamespace(session_server_ip="127.0.0.1", session_server_ports=ports)

    chosen_ports = set()
    for _ in range(32):
        calls.clear()
        tracer = await OpenAIEndpointTracer.create(args)
        port = int(tracer.session_server_id.rsplit(":", 1)[1])
        assert port in ports
        chosen_ports.add(port)

        await tracer.collect_records()
        prefix = f"http://127.0.0.1:{port}"
        assert [url for _, url in calls] == [
            f"{prefix}/sessions",
            tracer.base_url,
            tracer.base_url,
        ]
        assert tracer.base_url.startswith(f"{prefix}/sessions/")

    # 32 uniform picks over 4 ports miss a given port with p = (3/4)^32 ≈ 1e-4.
    assert len(chosen_ports) > 1
