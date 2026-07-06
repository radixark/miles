"""Unit tests for the session worker's op dispatch / reply encoding (no process spawn).

Drives ``SessionWorker.handle`` in-process against a real ``SessionCore`` (real
tokenizer) and a mock backend. The full chat-through-a-spawned-worker path is
covered by the multi-process equivalence tests.
"""

import json
import uuid
from types import SimpleNamespace

import pytest

from miles.rollout.session.core import build_session_core
from miles.rollout.session.ipc import (
    OP_CREATE,
    OP_DELETE,
    OP_GET,
    OP_HEALTH,
    OP_PROXY,
    decode_envelope,
    encode_request,
)
from miles.rollout.session.worker import SessionWorker


class _MockBackend:
    """Returns a canned JSON response for the proxy op; never called by CRUD ops."""

    async def do_proxy(self, request, path, *, body, headers):
        return {
            "request_body": body,
            "response_body": b'{"proxied":true}',
            "status_code": 200,
            "headers": {"content-type": "application/json"},
        }


@pytest.fixture(scope="module")
def worker():
    args = SimpleNamespace(
        miles_router_timeout=30,
        hf_checkpoint="Qwen/Qwen3-0.6B",
        chat_template_path=None,
        apply_chat_template_kwargs={"enable_thinking": False},
        tito_model="default",
        tito_allowed_append_roles=["tool"],
        session_server_instance_id=uuid.uuid4().hex,
    )
    return SessionWorker(build_session_core(_MockBackend(), args))


def _reply(payload):
    meta, body = decode_envelope(payload)
    return meta["status"], meta["headers"], body


async def test_health(worker):
    status, _, body = _reply(await worker.handle(encode_request(OP_HEALTH)))
    assert status == 200
    assert json.loads(body)["status"] == "ok"


async def test_create_get_delete_lifecycle(worker):
    sid = uuid.uuid4().hex

    status, _, body = _reply(await worker.handle(encode_request(OP_CREATE, session_id=sid)))
    assert status == 200
    assert json.loads(body)["session_id"] == sid

    status, _, body = _reply(await worker.handle(encode_request(OP_GET, session_id=sid)))
    assert status == 200
    assert json.loads(body)["session_id"] == sid
    assert json.loads(body)["records"] == []

    status, _, body = _reply(await worker.handle(encode_request(OP_DELETE, session_id=sid)))
    assert status == 204
    assert body == b""

    # get after delete → SessionNotFoundError mapped to a 404 error envelope
    status, _, body = _reply(await worker.handle(encode_request(OP_GET, session_id=sid)))
    assert status == 404
    assert "not found" in json.loads(body)["error"]


async def test_get_missing_session_returns_404(worker):
    status, _, body = _reply(await worker.handle(encode_request(OP_GET, session_id=uuid.uuid4().hex)))
    assert status == 404
    assert "not found" in json.loads(body)["error"]


async def test_proxy_passthrough(worker):
    status, headers, body = _reply(
        await worker.handle(encode_request(OP_PROXY, session_id=uuid.uuid4().hex, path="v1/models", method="GET"))
    )
    assert status == 200
    assert json.loads(body) == {"proxied": True}
    assert headers.get("content-type") == "application/json"


async def test_create_duplicate_id_raises(worker):
    # The collision guard makes a routing bug loud (becomes a 502 over IPC), rather
    # than silently clobbering an existing session.
    sid = uuid.uuid4().hex
    await worker.handle(encode_request(OP_CREATE, session_id=sid))
    with pytest.raises(ValueError):
        await worker.handle(encode_request(OP_CREATE, session_id=sid))
