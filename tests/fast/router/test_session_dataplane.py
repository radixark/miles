"""Data-plane integration: router + N worker shards over real socketpair IPC.

Wires the router to N ``SessionWorker`` shards (each its own ``SessionRegistry``)
over real ``socket.socketpair`` IPC channels in one event loop, and drives the
router app via httpx ASGITransport. This exercises hash routing, IPC framing,
worker dispatch, TITO state per shard, and equivalence with the single-process
server — without OS-process spawning (that lifecycle is the supervisor's job,
covered separately). Workers as in-loop IPC handlers still use distinct
registries, so cross-shard routing is genuinely tested.
"""

import json
import socket
import uuid
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pytest

from miles.rollout.session.core import SessionCore
from miles.rollout.session.ipc import open_unix_channel
from miles.rollout.session.linear_trajectory import SessionRegistry
from miles.rollout.session.router import build_router_app
from miles.rollout.session.server import SessionServer
from miles.rollout.session.sharding import worker_index_for_session
from miles.rollout.session.worker import ProxyBackend, SessionWorker
from miles.utils.chat_template_utils import get_tito_tokenizer
from miles.utils.processing_utils import load_tokenizer
from miles.utils.test_utils.mock_sglang_server import MockSGLangServer, ProcessResult, with_mock_server

N_WORKERS = 3


def _process_fn(prompt: str) -> ProcessResult:
    return ProcessResult(text=f"echo: {prompt}", finish_reason="stop")


def _make_patched_chat_response():
    # Add meta_info.output_token_logprobs the session core requires (same as test_sessions).
    original = MockSGLangServer._compute_chat_completions_response

    def patched(self, payload: dict) -> dict:
        response = original(self, payload)
        choice = response["choices"][0]
        logprobs_content = choice["logprobs"]["content"]
        otl = [(item["logprob"], self.tokenizer.convert_tokens_to_ids(item["token"])) for item in logprobs_content]
        choice["meta_info"] = {"output_token_logprobs": otl, "completion_tokens": len(otl)}
        return response

    return patched


def _make_args():
    return SimpleNamespace(
        miles_router_timeout=30,
        hf_checkpoint="Qwen/Qwen3-0.6B",
        chat_template_path=None,
        apply_chat_template_kwargs={"enable_thinking": False},
        tito_model="default",
        tito_allowed_append_roles=["tool"],
        session_server_instance_id=uuid.uuid4().hex,
    )


@pytest.fixture
async def env():
    with patch.object(MockSGLangServer, "_compute_chat_completions_response", new=_make_patched_chat_response()):
        with with_mock_server(process_fn=_process_fn) as backend:
            args = _make_args()
            # One shared tokenizer for the N worker shards (fast); each shard gets its own registry.
            tokenizer = load_tokenizer(args.hf_checkpoint, chat_template_path=None, trust_remote_code=True)
            tito = get_tito_tokenizer(
                tokenizer,
                tokenizer_type="default",
                chat_template_kwargs={"enable_thinking": False},
                allowed_append_roles=["tool"],
            )
            backends, worker_channels, router_channels = [], [], []
            for _ in range(N_WORKERS):
                a, b = socket.socketpair()
                be = ProxyBackend(backend.url, timeout=30)
                backends.append(be)
                core = SessionCore(
                    be, SessionRegistry(args, tokenizer, tito_tokenizer=tito), args, args.session_server_instance_id
                )
                worker = SessionWorker(core)
                worker_channels.append(await open_unix_channel(b, request_handler=worker.handle))
                router_channels.append(await open_unix_channel(a))

            app = build_router_app(router_channels, args.session_server_instance_id)
            client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://router")

            # Single-process server for equivalence comparison (same mock backend).
            sp = SessionServer(args, backend_url=backend.url)
            sp_client = httpx.AsyncClient(transport=httpx.ASGITransport(app=sp.app), base_url="http://sp")

            try:
                yield SimpleNamespace(client=client, sp_client=sp_client, n=N_WORKERS)
            finally:
                await client.aclose()
                await sp_client.aclose()
                for ch in worker_channels + router_channels:
                    await ch.aclose()
                for be in backends:
                    await be.aclose()
                await sp.client.aclose()


async def test_health_ok(env):
    r = await env.client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


async def test_lifecycle_and_routing_determinism(env):
    """create -> chat -> get -> delete for many ids; get/delete succeeding proves
    every id-bearing op routes to the same shard that holds the session."""
    ids = [(await env.client.post("/sessions")).json()["session_id"] for _ in range(8)]
    assert len({worker_index_for_session(sid, env.n) for sid in ids}) >= 2  # >=2 distinct shards exercised

    for sid in ids:
        chat = await env.client.post(
            f"/sessions/{sid}/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]}
        )
        assert chat.status_code == 200, chat.text

        got = await env.client.get(f"/sessions/{sid}")
        assert got.status_code == 200  # routed to the owning shard; 404 here would mean mis-routing
        assert len(got.json()["records"]) == 1

        assert (await env.client.delete(f"/sessions/{sid}")).status_code == 204
        assert (await env.client.get(f"/sessions/{sid}")).status_code == 404


async def test_get_missing_session_404(env):
    r = await env.client.get(f"/sessions/{'0' * 32}")
    assert r.status_code == 404
    assert "not found" in r.json()["error"]


async def test_equivalence_with_single_process(env):
    """Router (workers=N) and the single-process server produce identical chat/get/delete
    results (status + raw body bytes + content-type) for the same request."""
    messages = {"messages": [{"role": "user", "content": "What is 1+2?"}]}

    rid = (await env.client.post("/sessions")).json()["session_id"]
    sid = (await env.sp_client.post("/sessions")).json()["session_id"]

    rc = await env.client.post(f"/sessions/{rid}/v1/chat/completions", json=messages)
    sc = await env.sp_client.post(f"/sessions/{sid}/v1/chat/completions", json=messages)
    assert rc.status_code == sc.status_code == 200
    # Bodies match except the mock's per-call volatile fields (id/created), which differ
    # even between two calls to the same server. Equal-modulo-volatile proves the router
    # relays the worker's response the same way the single-process server renders it.
    rcb, scb = json.loads(rc.content), json.loads(sc.content)
    for volatile in ("id", "created"):
        rcb.pop(volatile, None)
        scb.pop(volatile, None)
    assert rcb == scb
    assert rc.headers.get("content-type") == sc.headers.get("content-type")
    assert rc.headers.get("content-length") == str(len(rc.content))  # router relayed framing intact

    # GET records: same shape (TITO metadata present and consistent across the spawn boundary)
    rg = await env.client.get(f"/sessions/{rid}")
    sg = await env.sp_client.get(f"/sessions/{sid}")
    assert rg.status_code == sg.status_code == 200
    rgj, sgj = rg.json(), sg.json()
    assert len(rgj["records"]) == len(sgj["records"]) == 1
    assert set(rgj["metadata"]) == set(sgj["metadata"])
    assert rgj["metadata"]["accumulated_token_ids"] == sgj["metadata"]["accumulated_token_ids"]

    # DELETE: identical 204 + empty body
    rd = await env.client.delete(f"/sessions/{rid}")
    sd = await env.sp_client.delete(f"/sessions/{sid}")
    assert rd.status_code == sd.status_code == 204
    assert rd.content == sd.content == b""


async def test_transport_error_502_equivalence():
    """A dead backend maps to the same 502 in workers=N (ProxyBackend) and workers=1
    (SessionServer.do_proxy) — pins the two do_proxy implementations against drift."""
    from miles.rollout.session.core import ProxyRequest
    from miles.utils.http_utils import find_available_port

    dead = f"http://127.0.0.1:{find_available_port(42000)}"  # nothing listening → connection refused
    args = _make_args()
    pb = ProxyBackend(dead, timeout=2)
    sp = SessionServer(args, backend_url=dead)
    try:
        req = ProxyRequest(method="POST", query="")
        r_pb = await pb.do_proxy(req, "v1/chat/completions", body=b"{}", headers={})
        r_sp = await sp.do_proxy(req, "v1/chat/completions", body=b"{}", headers={})
        assert r_pb["status_code"] == r_sp["status_code"] == 502
        assert r_pb["headers"] == r_sp["headers"]
        assert json.loads(r_pb["response_body"])["error"].startswith("backend transport error")
        assert json.loads(r_sp["response_body"])["error"].startswith("backend transport error")
    finally:
        await pb.aclose()
        await sp.client.aclose()
