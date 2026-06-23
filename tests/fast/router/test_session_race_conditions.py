"""E2E session stress tests.

Contract under test (per-session in-flight gate):
- A session admits at most one in-flight chat completion. A second concurrent
  same-session chat fast-fails 409 ("session already has an in-flight chat
  completion") at slot-claim time, before the body is read/parsed and before
  the backend is hit; it never reaches the backend.
- The in-flight slot is released on every exit path (success, malformed JSON,
  prepare error, upstream non-200, transport 502, response validation failure,
  state-update error, client cancel/disconnect), and only by the request that
  claimed it: a request that got 409/404 does not clear the owner's slot.
- Different sessions still run in parallel (no global lock); per-session clients
  can run turn-by-turn without idle gaps while global load stays parallel.
- Delete marks session.closing=True, acquires session.lock, then removes. The
  lock is not held during the proxy, so delete can proceed while a chat is
  mid-proxy; that chat's commit sees closing=True and skips the state update.
- Chat to a closing session gets 404 immediately, and closing (404) has
  priority over busy (409).
- Concurrent deletes on the same session: second delete gets 404.
- The upstream body is passed through faithfully; stale framing headers
  (content-length / transfer-encoding / content-encoding) are stripped.
"""

from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import requests
from starlette.responses import Response

from miles.rollout.session.linear_trajectory import LinearTrajectory
from miles.rollout.session.session_errors import (
    MessageValidationError,
    SessionInvariantError,
    TokenizationError,
    UpstreamResponseError,
)
from miles.rollout.session.session_server import SessionServer
from miles.rollout.session.sessions import _parse_and_validate_response
from miles.utils.http_utils import find_available_port
from miles.utils.test_utils.mock_sglang_server import MockSGLangServer, ProcessResult, with_mock_server
from miles.utils.test_utils.uvicorn_thread_server import UvicornThreadServer

HF_CHECKPOINT = "Qwen/Qwen3-0.6B"


def _patch_mock_chat_response():
    original_chat_response = MockSGLangServer._compute_chat_completions_response

    def patched_chat_response(self, payload: dict) -> dict:
        response = original_chat_response(self, payload)
        # Session server expects output_token_logprobs as (logprob, token_id).
        choice = response["choices"][0]
        logprobs_content = choice["logprobs"]["content"]
        output_token_logprobs = [
            (item["logprob"], self.tokenizer.convert_tokens_to_ids(item["token"])) for item in logprobs_content
        ]
        choice["meta_info"] = {
            "output_token_logprobs": output_token_logprobs,
            "completion_tokens": len(output_token_logprobs),
        }
        return response

    return patch.object(MockSGLangServer, "_compute_chat_completions_response", new=patched_chat_response)


def _patch_mock_chat_response_bad_first():
    """Like `_patch_mock_chat_response`, but the FIRST chat response is missing
    `meta_info` so the session server raises UpstreamResponseError (502). The
    second and later responses are valid, so a retry after the failure can 200.
    """
    original_chat_response = MockSGLangServer._compute_chat_completions_response
    state = {"calls": 0}

    def patched_chat_response(self, payload: dict) -> dict:
        response = original_chat_response(self, payload)
        choice = response["choices"][0]
        logprobs_content = choice["logprobs"]["content"]
        output_token_logprobs = [
            (item["logprob"], self.tokenizer.convert_tokens_to_ids(item["token"])) for item in logprobs_content
        ]
        choice["meta_info"] = {
            "output_token_logprobs": output_token_logprobs,
            "completion_tokens": len(output_token_logprobs),
        }
        state["calls"] += 1
        if state["calls"] == 1:
            # Strip meta_info from the first response only -> 502 on first chat.
            choice.pop("meta_info", None)
        return response

    return patch.object(MockSGLangServer, "_compute_chat_completions_response", new=patched_chat_response)


def _patch_mock_chat_response_bad_logprob_first(bad_logprob="bad-logprob"):
    """Like `_patch_mock_chat_response`, but the FIRST chat response carries a
    non-numeric logprob value (entry[0]) so the session server rejects it with
    UpstreamResponseError (502) instead of letting the bad value flow into
    Sample.rollout_log_probs downstream. Later responses are valid.
    """
    original_chat_response = MockSGLangServer._compute_chat_completions_response
    state = {"calls": 0}

    def patched_chat_response(self, payload: dict) -> dict:
        response = original_chat_response(self, payload)
        choice = response["choices"][0]
        logprobs_content = choice["logprobs"]["content"]
        output_token_logprobs = [
            (item["logprob"], self.tokenizer.convert_tokens_to_ids(item["token"])) for item in logprobs_content
        ]
        state["calls"] += 1
        if state["calls"] == 1 and output_token_logprobs:
            # Corrupt the first response's leading logprob value only.
            _logprob, token_id = output_token_logprobs[0][0], output_token_logprobs[0][1]
            output_token_logprobs[0] = (bad_logprob, token_id)
        choice["meta_info"] = {
            "output_token_logprobs": output_token_logprobs,
            "completion_tokens": len(output_token_logprobs),
        }
        return response

    return patch.object(MockSGLangServer, "_compute_chat_completions_response", new=patched_chat_response)


@contextmanager
def _router_env(process_fn, *, latency: float = 0.0, response_patch=None):
    with (response_patch or _patch_mock_chat_response)():
        with with_mock_server(model_name=HF_CHECKPOINT, process_fn=process_fn, latency=latency) as backend:
            args = SimpleNamespace(
                miles_router_timeout=30,
                hf_checkpoint=HF_CHECKPOINT,
                chat_template_path=None,
                trajectory_manager="linear_trajectory",
                tito_allowed_append_roles=["tool", "system"],
            )
            server_obj = SessionServer(args, backend_url=backend.url)

            port = find_available_port(31000)
            server = UvicornThreadServer(server_obj.app, host="127.0.0.1", port=port)
            server.start()
            url = f"http://127.0.0.1:{port}"

            try:
                yield SimpleNamespace(url=url, backend=backend, server=server)
            finally:
                server.stop()


def _create_session(url: str) -> str:
    response = requests.post(f"{url}/sessions", timeout=5.0)
    assert response.status_code == 200
    return response.json()["session_id"]


def _chat(url: str, session_id: str, payload: dict, timeout: float = 20.0) -> requests.Response:
    return requests.post(
        f"{url}/sessions/{session_id}/v1/chat/completions",
        json=payload,
        timeout=timeout,
    )


def _wait_for_backend_requests(backend, count: int, timeout: float = 5.0) -> None:
    """Block until the backend has logged exactly `count` requests.

    Using a backend-arrival barrier instead of sleeping makes the "request is
    parked in proxy" precondition deterministic: once the entry is logged the
    owner has claimed the in-flight slot and is sitting in the latency window.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        if len(backend.request_log) == count:
            return
        time.sleep(0.005)
    raise AssertionError(f"backend did not reach {count} requests in {timeout}s (saw {len(backend.request_log)})")


class TestSessionConcurrencyContracts:
    def test_same_session_second_chat_returns_409(self):
        """A session admits one in-flight chat; concurrents fast-fail 409.

        Park chat A in proxy (held by backend latency) and confirm via the
        arrival barrier that A holds the slot. Contenders B/C/D on the same
        session must each get 409 without ever reaching the backend, and they
        must not release A's slot. The 409 is returned at slot-claim time,
        before the contender's body is read and before the backend is hit, so
        each contender returns near-instantly rather than waiting out A's
        backend latency. After A finishes 200, the slot is free and a fresh
        same-session chat succeeds.
        """

        # Latency comfortably larger than the time to fire three contenders.
        latency = 0.5

        def process_fn(prompt: str) -> ProcessResult:
            return ProcessResult(text="concurrent-ok", finish_reason="stop")

        with _router_env(process_fn, latency=latency) as env:
            session_id = _create_session(env.url)
            payload = {"messages": [{"role": "user", "content": "park-in-proxy"}]}

            env.backend.reset_stats()
            with ThreadPoolExecutor(max_workers=1) as pool:
                # Fire A and wait until it is parked in proxy holding the slot.
                chat_a = pool.submit(_chat, env.url, session_id, payload, 30.0)
                _wait_for_backend_requests(env.backend, 1)

                # Contenders on the SAME session while A is parked -> 409 each.
                # Each contender's wall-clock is measured: the gate rejects at
                # slot-claim time (before body read, before backend), so a
                # contender must NOT block for the owner's backend latency.
                contender_codes = []
                contender_elapsed_s = []
                for _ in range(3):
                    t0 = time.perf_counter()
                    resp = _chat(env.url, session_id, payload, timeout=10.0)
                    contender_elapsed_s.append(time.perf_counter() - t0)
                    contender_codes.append(resp.status_code)
                    assert resp.status_code == 409, f"contender should be 409, got {resp.status_code}"
                    assert resp.json()["error"] == "session already has an in-flight chat completion"

                # Each 409 returned well under the owner's backend latency,
                # proving the contender did not block on A's parked proxy. Half
                # the latency is a generous ceiling vs. the near-instant gate.
                assert all(dt < latency / 2 for dt in contender_elapsed_s), (
                    f"a contender blocked on the owner's backend latency "
                    f"(latency={latency}s, contender elapsed={contender_elapsed_s})"
                )

                # Contenders never reached the backend: still exactly A's request.
                assert len(env.backend.request_log) == 1

                # A finishes 200; a 409 contender did NOT clear A's slot.
                chat_a_resp = chat_a.result(timeout=30.0)

            assert chat_a_resp.status_code == 200
            assert contender_codes == [409, 409, 409]
            assert len(env.backend.request_log) == 1

            # Slot was released on A's success: a fresh same-session chat works.
            # The follow-up must be an append-only continuation of A's committed
            # trajectory, so build it from A's assistant message.
            assistant = chat_a_resp.json()["choices"][0]["message"]
            follow_up = {
                "messages": [
                    {"role": "user", "content": "park-in-proxy"},
                    assistant,
                    {"role": "system", "content": "continue-after-A"},
                ]
            }
            after = _chat(env.url, session_id, follow_up, timeout=20.0)
            assert after.status_code == 200
            assert len(env.backend.request_log) == 2

    def test_different_sessions_can_run_in_parallel(self):
        def process_fn(prompt: str) -> ProcessResult:
            return ProcessResult(text="parallel-ok", finish_reason="stop")

        with _router_env(process_fn, latency=0.2) as env:
            session_ids = [_create_session(env.url) for _ in range(6)]

            env.backend.reset_stats()
            with ThreadPoolExecutor(max_workers=6) as pool:
                futures = [
                    pool.submit(
                        _chat,
                        env.url,
                        sid,
                        {"messages": [{"role": "user", "content": f"parallel-{i}"}]},
                    )
                    for i, sid in enumerate(session_ids)
                ]
                responses = [f.result(timeout=30.0) for f in futures]

            assert all(resp.status_code == 200 for resp in responses)
            assert len(env.backend.request_log) == 6
            assert env.backend.max_concurrent >= 3

    def test_e2e_pressure_serial_per_session_parallel_globally(self):
        num_sessions = 8
        turns_per_session = 3

        def process_fn(prompt: str) -> ProcessResult:
            return ProcessResult(text="turn-ok", finish_reason="stop")

        with _router_env(process_fn, latency=0.08) as env:
            session_ids = [_create_session(env.url) for _ in range(num_sessions)]

            def run_session_worker(session_id: str, idx: int) -> bool:
                messages: list[dict] = [{"role": "user", "content": f"session-{idx}-turn-0"}]
                for turn in range(turns_per_session):
                    resp = _chat(env.url, session_id, {"messages": messages}, timeout=30.0)
                    assert resp.status_code == 200
                    assistant = resp.json()["choices"][0]["message"]
                    if turn < turns_per_session - 1:
                        messages = [
                            *messages,
                            assistant,
                            {"role": "system", "content": f"session-{idx}-continue-{turn}"},
                        ]
                return True

            env.backend.reset_stats()
            with ThreadPoolExecutor(max_workers=num_sessions) as pool:
                futures = [pool.submit(run_session_worker, sid, idx) for idx, sid in enumerate(session_ids)]
                results = [f.result(timeout=120.0) for f in futures]

            assert all(results)
            assert len(env.backend.request_log) == num_sessions * turns_per_session
            assert env.backend.max_concurrent >= 4

    def test_delete_can_proceed_while_chat_is_mid_proxy(self):
        """Delete can acquire the lock while a chat is mid-proxy.

        The lock is not held during the proxy, so delete proceeds; the in-flight
        chat's commit step sees session.closing=True and skips the state update.
        Both chat and delete complete without error.
        """

        def process_fn(prompt: str) -> ProcessResult:
            return ProcessResult(text="slow-turn", finish_reason="stop")

        with _router_env(process_fn, latency=0.35) as env:
            session_id = _create_session(env.url)
            payload = {"messages": [{"role": "user", "content": "slow-turn-0"}]}

            with ThreadPoolExecutor(max_workers=2) as pool:
                inflight = pool.submit(_chat, env.url, session_id, payload, 30.0)

                # Wait until the first request has reached backend before deleting.
                deadline = time.time() + 5.0
                while time.time() < deadline:
                    if env.backend.request_log:
                        break
                    time.sleep(0.01)
                else:
                    raise AssertionError("in-flight request did not reach backend in time")

                delete_resp = requests.delete(f"{env.url}/sessions/{session_id}", timeout=30.0)
                inflight_resp = inflight.result(timeout=30.0)

            # Chat returns 200 (backend responded); delete returns 204.
            assert inflight_resp.status_code == 200
            assert delete_resp.status_code == 204
            # Session is gone after delete.
            post_delete = _chat(env.url, session_id, payload, timeout=10.0)
            assert post_delete.status_code == 404


class TestClosingRaceConditions:
    """Tests for race conditions around session.closing flag."""

    def test_chat_during_delete_returns_404(self):
        """Chat requests arriving after delete sets closing=True get 404.

        Timeline:
        1. Chat A starts, claims the in-flight slot, and is proxying to backend.
        2. Delete arrives, sets session.closing=True, acquires lock, removes session.
        3. Chat B arrives, sees session.closing=True, returns 404 immediately.
        4. Chat A's commit sees closing=True, skips the state update, returns 200.
        """

        def process_fn(prompt: str) -> ProcessResult:
            return ProcessResult(text="slow", finish_reason="stop")

        with _router_env(process_fn, latency=0.5) as env:
            session_id = _create_session(env.url)
            payload = {"messages": [{"role": "user", "content": "slow-chat"}]}

            with ThreadPoolExecutor(max_workers=3) as pool:
                # 1. Start slow chat A
                chat_a = pool.submit(_chat, env.url, session_id, payload, 30.0)

                # Wait for chat A to reach backend
                deadline = time.time() + 5.0
                while time.time() < deadline:
                    if env.backend.request_log:
                        break
                    time.sleep(0.01)

                # 2. Start delete (will block waiting for lock)
                delete_future = pool.submit(
                    requests.delete,
                    f"{env.url}/sessions/{session_id}",
                    timeout=30.0,
                )
                # Small delay to ensure delete has set closing=True
                time.sleep(0.05)

                # 3. Chat B should get 404 because session.closing=True
                chat_b = _chat(env.url, session_id, payload, timeout=10.0)
                assert chat_b.status_code == 404, f"Chat during closing should return 404, got {chat_b.status_code}"

                # Wait for remaining futures
                chat_a_resp = chat_a.result(timeout=30.0)
                delete_resp = delete_future.result(timeout=30.0)

            assert chat_a_resp.status_code == 200
            assert delete_resp.status_code == 204

    def test_double_delete_second_returns_404(self):
        """Concurrent delete on the same session: second delete gets 404.

        With session.closing flag, the first delete sets closing=True.
        The second delete sees closing=True and returns 404.
        """

        def process_fn(prompt: str) -> ProcessResult:
            return ProcessResult(text="ok", finish_reason="stop")

        with _router_env(process_fn, latency=0.3) as env:
            session_id = _create_session(env.url)

            # Start a slow chat to hold the lock
            payload = {"messages": [{"role": "user", "content": "hold-lock"}]}
            with ThreadPoolExecutor(max_workers=3) as pool:
                chat_future = pool.submit(_chat, env.url, session_id, payload, 30.0)

                # Wait for chat to reach backend
                deadline = time.time() + 5.0
                while time.time() < deadline:
                    if env.backend.request_log:
                        break
                    time.sleep(0.01)

                # Fire two deletes concurrently
                delete_1 = pool.submit(
                    requests.delete,
                    f"{env.url}/sessions/{session_id}",
                    timeout=30.0,
                )
                time.sleep(0.02)  # tiny delay to let first delete set closing
                delete_2 = pool.submit(
                    requests.delete,
                    f"{env.url}/sessions/{session_id}",
                    timeout=30.0,
                )

                chat_resp = chat_future.result(timeout=30.0)
                d1 = delete_1.result(timeout=30.0)
                d2 = delete_2.result(timeout=30.0)

            assert chat_resp.status_code == 200
            # One delete succeeds, the other gets 404
            codes = sorted([d1.status_code, d2.status_code])
            assert codes == [204, 404], f"Expected [204, 404], got {codes}"

    def test_chat_after_delete_returns_404(self):
        """Chat request after session is fully deleted returns 404."""

        def process_fn(prompt: str) -> ProcessResult:
            return ProcessResult(text="ok", finish_reason="stop")

        with _router_env(process_fn) as env:
            session_id = _create_session(env.url)

            # Delete the session
            delete_resp = requests.delete(f"{env.url}/sessions/{session_id}", timeout=5.0)
            assert delete_resp.status_code == 204

            # Chat should get 404
            payload = {"messages": [{"role": "user", "content": "hello"}]}
            chat_resp = _chat(env.url, session_id, payload, timeout=5.0)
            assert chat_resp.status_code == 404

            # GET should also get 404
            get_resp = requests.get(f"{env.url}/sessions/{session_id}", timeout=5.0)
            assert get_resp.status_code == 404

    def test_concurrent_chats_then_delete(self):
        """Concurrent same-session chats do not queue under the gate.

        One chat parks in proxy holding the slot; a couple more same-session
        chats fired concurrently get 409 (gate, before the backend). A delete
        then returns 204 and the parked chat returns 200 (commit skips the
        state update on closing). No 500s; codes are a subset of {200,409,404}.
        """

        def process_fn(prompt: str) -> ProcessResult:
            return ProcessResult(text="queued-ok", finish_reason="stop")

        with _router_env(process_fn, latency=0.3) as env:
            session_id = _create_session(env.url)
            payload = {"messages": [{"role": "user", "content": "queued"}]}

            with ThreadPoolExecutor(max_workers=2) as pool:
                # Park the owner chat in proxy holding the in-flight slot.
                owner = pool.submit(_chat, env.url, session_id, payload, 30.0)
                _wait_for_backend_requests(env.backend, 1)

                # Concurrent same-session chats hit the gate -> 409.
                contender_codes = [_chat(env.url, session_id, payload, timeout=10.0).status_code for _ in range(2)]

                # Delete the session while the owner is still parked.
                delete_future = pool.submit(
                    requests.delete,
                    f"{env.url}/sessions/{session_id}",
                    timeout=30.0,
                )

                owner_resp = owner.result(timeout=30.0)
                delete_resp = delete_future.result(timeout=30.0)

            assert delete_resp.status_code == 204
            assert owner_resp.status_code == 200
            assert contender_codes == [409, 409]

            # No 500s; every chat code is in the allowed set, and exactly one
            # chat (the owner) reached the backend.
            all_chat_codes = [owner_resp.status_code, *contender_codes]
            assert all(c in (200, 409, 404) for c in all_chat_codes), f"Unexpected status codes: {all_chat_codes}"
            assert len(env.backend.request_log) == 1

    def test_rapid_create_chat_delete_cycles(self):
        """Rapidly create, chat, and delete sessions to stress the lifecycle.

        Ensures no deadlocks or crashes from rapid session lifecycle operations.
        """

        def process_fn(prompt: str) -> ProcessResult:
            return ProcessResult(text="cycle-ok", finish_reason="stop")

        with _router_env(process_fn) as env:

            def lifecycle_cycle(idx: int) -> bool:
                session_id = _create_session(env.url)
                payload = {"messages": [{"role": "user", "content": f"cycle-{idx}"}]}
                chat_resp = _chat(env.url, session_id, payload, timeout=10.0)
                assert chat_resp.status_code == 200
                delete_resp = requests.delete(f"{env.url}/sessions/{session_id}", timeout=5.0)
                assert delete_resp.status_code == 204
                # Verify gone
                get_resp = requests.get(f"{env.url}/sessions/{session_id}", timeout=5.0)
                assert get_resp.status_code == 404
                return True

            with ThreadPoolExecutor(max_workers=8) as pool:
                futures = [pool.submit(lifecycle_cycle, i) for i in range(20)]
                results = [f.result(timeout=60.0) for f in futures]

            assert all(results)


class TestSlotReleaseAfterError:
    """The in-flight slot is freed on failing exit paths: a fresh legal chat on
    the same session after a failure must succeed (200, not 409).
    """

    def test_slot_released_after_malformed_request_json(self):
        """Malformed JSON body errors (500-class) before the backend; the slot
        is released so a subsequent normal chat on the same session returns 200.
        """

        def process_fn(prompt: str) -> ProcessResult:
            return ProcessResult(text="ok", finish_reason="stop")

        with _router_env(process_fn) as env:
            session_id = _create_session(env.url)

            bad = requests.post(
                f"{env.url}/sessions/{session_id}/v1/chat/completions",
                data="{not json",
                headers={"content-type": "application/json"},
                timeout=10.0,
            )
            assert bad.status_code >= 500
            # The malformed body never reached the backend.
            assert len(env.backend.request_log) == 0

            good = _chat(env.url, session_id, {"messages": [{"role": "user", "content": "hi"}]}, timeout=20.0)
            assert good.status_code == 200
            assert len(env.backend.request_log) == 1

    def test_slot_released_after_response_validation_failure(self):
        """An upstream response missing meta_info raises UpstreamResponseError
        (502); the slot is released so a subsequent normal chat returns 200.
        """

        def process_fn(prompt: str) -> ProcessResult:
            return ProcessResult(text="ok", finish_reason="stop")

        with _router_env(process_fn, response_patch=_patch_mock_chat_response_bad_first) as env:
            session_id = _create_session(env.url)

            bad = _chat(env.url, session_id, {"messages": [{"role": "user", "content": "hi"}]}, timeout=20.0)
            assert bad.status_code == 502
            assert len(env.backend.request_log) == 1

            good = _chat(env.url, session_id, {"messages": [{"role": "user", "content": "hi again"}]}, timeout=20.0)
            assert good.status_code == 200
            assert len(env.backend.request_log) == 2

    def test_bad_logprob_value_rejected_without_committing(self):
        """A successful (200) upstream response carrying a non-numeric logprob
        value must return 502, commit NOTHING to the session (no record, no
        accumulated token id), and release the slot so the next legal chat 200s.
        Guards against the bad logprob flowing into rollout_log_probs downstream.
        """

        def process_fn(prompt: str) -> ProcessResult:
            return ProcessResult(text="ok", finish_reason="stop")

        with _router_env(process_fn, response_patch=_patch_mock_chat_response_bad_logprob_first) as env:
            session_id = _create_session(env.url)

            bad = _chat(env.url, session_id, {"messages": [{"role": "user", "content": "hi"}]}, timeout=20.0)
            assert bad.status_code == 502
            assert len(env.backend.request_log) == 1

            # Nothing from the malformed response was committed.
            state = requests.get(f"{env.url}/sessions/{session_id}", timeout=5.0).json()
            assert state["records"] == []
            assert state["metadata"]["accumulated_token_ids"] == []

            good = _chat(env.url, session_id, {"messages": [{"role": "user", "content": "hi again"}]}, timeout=20.0)
            assert good.status_code == 200
            assert len(env.backend.request_log) == 2

            after = requests.get(f"{env.url}/sessions/{session_id}", timeout=5.0).json()
            assert len(after["records"]) == 1
            assert len(after["metadata"]["accumulated_token_ids"]) > 0

    def test_non_finite_logprob_rejected_without_committing(self):
        """Raw upstream NaN is accepted by Python's default json decoder; the
        session boundary must still reject it before recording training data.
        """

        def process_fn(prompt: str) -> ProcessResult:
            return ProcessResult(text="ok", finish_reason="stop")

        original_do_proxy = SessionServer.do_proxy
        state = {"calls": 0}

        async def one_shot_do_proxy(self, request, path, body=None, headers=None):
            state["calls"] += 1
            if state["calls"] == 1:
                return {
                    "request_body": body,
                    "response_body": (
                        b'{"choices":[{"message":{"role":"assistant","content":"ok"},'
                        b'"meta_info":{"output_token_logprobs":[[NaN,562]],"completion_tokens":1}}]}'
                    ),
                    "status_code": 200,
                    "headers": {"content-type": "application/json"},
                }
            return await original_do_proxy(self, request, path, body=body, headers=headers)

        with _router_env(process_fn) as env:
            session_id = _create_session(env.url)
            with patch.object(SessionServer, "do_proxy", one_shot_do_proxy):
                bad = _chat(env.url, session_id, {"messages": [{"role": "user", "content": "hi"}]}, timeout=20.0)
                assert bad.status_code == 502

                session_state = requests.get(f"{env.url}/sessions/{session_id}", timeout=5.0).json()
                assert session_state["records"] == []
                assert session_state["metadata"]["accumulated_token_ids"] == []

                good = _chat(
                    env.url,
                    session_id,
                    {"messages": [{"role": "user", "content": "hi again"}]},
                    timeout=20.0,
                )
                assert good.status_code == 200


def _normal_messages(content: str) -> dict:
    return {"messages": [{"role": "user", "content": content}]}


class TestSlotReleaseInjectedFailures:
    """Deterministic slot-release coverage for the remaining enumerated failure
    paths. Each test injects a one-shot failure on the first same-session chat,
    then proves the slot was released: the NEXT same-session chat returns 200
    (no residual 409). Failures are injected via class-level mock patches so the
    server thread's instance/session is affected; ordering is enforced by a
    per-test call counter, not by sleeps.
    """

    def test_slot_released_after_prepare_validation_error(self):
        """`prepare_pretokenized` raising MessageValidationError (400) before the
        backend releases the slot; the next normal chat returns 200.
        """

        def process_fn(prompt: str) -> ProcessResult:
            return ProcessResult(text="ok", finish_reason="stop")

        original_prepare = LinearTrajectory.prepare_pretokenized
        state = {"calls": 0}

        def one_shot_prepare(self, *args, **kwargs):
            state["calls"] += 1
            if state["calls"] == 1:
                raise MessageValidationError("injected prepare failure")
            return original_prepare(self, *args, **kwargs)

        with _router_env(process_fn) as env:
            session_id = _create_session(env.url)
            with patch.object(LinearTrajectory, "prepare_pretokenized", one_shot_prepare):
                bad = _chat(env.url, session_id, _normal_messages("hi"), timeout=20.0)
                assert bad.status_code == 400
                # prepare failed before the backend was hit.
                assert len(env.backend.request_log) == 0

                good = _chat(env.url, session_id, _normal_messages("hi again"), timeout=20.0)
                assert good.status_code == 200
                assert len(env.backend.request_log) == 1

    def test_slot_released_after_upstream_non_200(self):
        """An upstream non-200 passes through (400) without recording; the slot
        is released so the next normal chat returns 200.
        """

        def process_fn(prompt: str) -> ProcessResult:
            return ProcessResult(text="ok", finish_reason="stop")

        original_do_proxy = SessionServer.do_proxy
        state = {"calls": 0}

        async def one_shot_do_proxy(self, request, path, body=None, headers=None):
            state["calls"] += 1
            if state["calls"] == 1:
                return {
                    "request_body": body,
                    "response_body": b'{"error":"bad"}',
                    "status_code": 400,
                    "headers": {"content-type": "application/json"},
                }
            return await original_do_proxy(self, request, path, body=body, headers=headers)

        with _router_env(process_fn) as env:
            session_id = _create_session(env.url)
            with patch.object(SessionServer, "do_proxy", one_shot_do_proxy):
                bad = _chat(env.url, session_id, _normal_messages("hi"), timeout=20.0)
                assert bad.status_code == 400
                assert bad.json()["error"] == "bad"

                good = _chat(env.url, session_id, _normal_messages("hi again"), timeout=20.0)
                assert good.status_code == 200

    def test_slot_released_after_transport_502(self):
        """A transport-error 502 (do_proxy's real error shape) passes through; the
        slot is released so the next normal chat returns 200.
        """

        def process_fn(prompt: str) -> ProcessResult:
            return ProcessResult(text="ok", finish_reason="stop")

        original_do_proxy = SessionServer.do_proxy
        state = {"calls": 0}

        async def one_shot_do_proxy(self, request, path, body=None, headers=None):
            state["calls"] += 1
            if state["calls"] == 1:
                return {
                    "request_body": body,
                    "response_body": b'{"error":"backend transport error"}',
                    "status_code": 502,
                    "headers": {"content-type": "application/json"},
                }
            return await original_do_proxy(self, request, path, body=body, headers=headers)

        with _router_env(process_fn) as env:
            session_id = _create_session(env.url)
            with patch.object(SessionServer, "do_proxy", one_shot_do_proxy):
                bad = _chat(env.url, session_id, _normal_messages("hi"), timeout=20.0)
                assert bad.status_code == 502

                good = _chat(env.url, session_id, _normal_messages("hi again"), timeout=20.0)
                assert good.status_code == 200

    def test_slot_released_after_state_update_error(self):
        """`update_pretokenized_state` raising TokenizationError (500) after a 200
        proxy releases the slot; the next normal chat returns 200.
        """

        def process_fn(prompt: str) -> ProcessResult:
            return ProcessResult(text="ok", finish_reason="stop")

        original_update = LinearTrajectory.update_pretokenized_state
        state = {"calls": 0}

        def one_shot_update(self, *args, **kwargs):
            state["calls"] += 1
            if state["calls"] == 1:
                raise TokenizationError("injected state-update failure")
            return original_update(self, *args, **kwargs)

        with _router_env(process_fn) as env:
            session_id = _create_session(env.url)
            with patch.object(LinearTrajectory, "update_pretokenized_state", one_shot_update):
                bad = _chat(env.url, session_id, _normal_messages("hi"), timeout=20.0)
                assert bad.status_code == 500
                # The proxy did run; the failure is in the commit step.
                assert len(env.backend.request_log) == 1

                good = _chat(env.url, session_id, _normal_messages("hi again"), timeout=20.0)
                assert good.status_code == 200
                assert len(env.backend.request_log) == 2

    def test_invariant_mismatch_returns_500_and_releases_slot(self, caplog):
        """If session.num_assistant changes between the prepare-segment capture
        and the commit-segment check, the commit raises SessionInvariantError
        (500) instead of silently skipping the state update; the slot is then
        released so a fresh session's normal chat returns 200.

        The mismatch is forced deterministically (no sleeps): prepare_pretokenized
        stashes the live session object, then do_proxy mutates that session's
        num_assistant on its first call — an out-of-band change the in-flight
        gate is supposed to make impossible. The commit check then trips.

        The server runs in a uvicorn thread in the same process, so its `logging`
        records propagate to the root logger and `caplog` captures them. We assert
        the commit segment emitted an ERROR naming the invariant and the session
        id (matching the `logger.error(...)` in `chat_completions`), and that a
        normal fresh chat never emits that ERROR (no false trigger).
        """

        def process_fn(prompt: str) -> ProcessResult:
            return ProcessResult(text="ok", finish_reason="stop")

        holder: dict = {}
        original_prepare = LinearTrajectory.prepare_pretokenized
        original_do_proxy = SessionServer.do_proxy
        proxy_state = {"calls": 0}

        def stashing_prepare(self, *args, **kwargs):
            result = original_prepare(self, *args, **kwargs)
            # Stash the live session so the do_proxy wrapper can mutate it after
            # expected_num_assistant has already been captured.
            holder["session"] = self
            return result

        async def mutating_do_proxy(self, request, path, body=None, headers=None):
            proxy_state["calls"] += 1
            if proxy_state["calls"] == 1:
                # Out-of-band state change mid-proxy, after the prepare segment
                # captured expected_num_assistant -> commit-time check must trip.
                holder["session"].num_assistant += 1
            return await original_do_proxy(self, request, path, body=body, headers=headers)

        with _router_env(process_fn) as env:
            session_id = _create_session(env.url)
            with (
                patch.object(LinearTrajectory, "prepare_pretokenized", stashing_prepare),
                patch.object(SessionServer, "do_proxy", mutating_do_proxy),
                caplog.at_level(logging.ERROR),
            ):
                bad = _chat(env.url, session_id, _normal_messages("hi"), timeout=20.0)
                assert bad.status_code == 500, f"invariant mismatch must be 500, got {bad.status_code}"
                # Body carries SessionInvariantError's message (the gate did not
                # silently 200-skip the commit).
                error = bad.json()["error"]
                assert "session state changed under the in-flight gate" in error
                assert SessionInvariantError(error).status_code == 500
                # The proxy ran (the mismatch is raised in the commit step after it).
                assert len(env.backend.request_log) == 1

                # The commit segment logged an ERROR naming the invariant and the
                # session id (matching the `logger.error(...)` in chat_completions).
                invariant_errors = [
                    rec
                    for rec in caplog.records
                    if rec.levelno == logging.ERROR
                    and "invariant" in rec.getMessage()
                    and session_id in rec.getMessage()
                ]
                assert invariant_errors, (
                    "expected an ERROR log naming the invariant and the session id; "
                    f"saw records: {[(r.levelname, r.getMessage()) for r in caplog.records]}"
                )

            # Patches removed. The slot must have been released on the 500 exit
            # path: a follow-up on the SAME session is NOT a stuck 409. (Its
            # bumped num_assistant left the prefix state inconsistent, so the
            # legal-continuation code may be a non-200 error, but never 409.)
            same_session_followup = _chat(env.url, session_id, _normal_messages("hi same"), timeout=20.0)
            assert same_session_followup.status_code != 409, (
                f"slot was not released after the 500: same-session follow-up got "
                f"{same_session_followup.status_code} (a stuck 409)"
            )

            # A FRESH session's normal chat returns 200, confirming the event
            # loop stayed live and the gate path recovers cleanly. Capture ERROR
            # records during this clean chat to prove no false invariant trigger.
            fresh_session_id = _create_session(env.url)
            with caplog.at_level(logging.ERROR):
                caplog.clear()
                good = _chat(env.url, fresh_session_id, _normal_messages("hi again"), timeout=20.0)
                assert good.status_code == 200
                false_triggers = [
                    rec for rec in caplog.records if rec.levelno == logging.ERROR and "invariant" in rec.getMessage()
                ]
                assert not false_triggers, (
                    "a normal chat must not emit the invariant ERROR; "
                    f"saw: {[r.getMessage() for r in false_triggers]}"
                )

    def test_slot_released_after_client_cancel_mid_proxy(self):
        """If the handler is cancelled mid-proxy (client disconnect), the request
        errors at the client side but the `finally` releases the slot; the next
        normal same-session chat returns 200.
        """

        def process_fn(prompt: str) -> ProcessResult:
            return ProcessResult(text="ok", finish_reason="stop")

        original_do_proxy = SessionServer.do_proxy
        state = {"calls": 0}

        async def one_shot_do_proxy(self, request, path, body=None, headers=None):
            state["calls"] += 1
            if state["calls"] == 1:
                raise asyncio.CancelledError()
            return await original_do_proxy(self, request, path, body=body, headers=headers)

        with _router_env(process_fn) as env:
            session_id = _create_session(env.url)
            with patch.object(SessionServer, "do_proxy", one_shot_do_proxy):
                try:
                    _chat(env.url, session_id, _normal_messages("hi"), timeout=20.0)
                except requests.exceptions.RequestException:
                    pass

                good = _chat(env.url, session_id, _normal_messages("hi again"), timeout=20.0)
                assert good.status_code == 200

    def test_slot_released_after_real_client_disconnect_mid_proxy(self):
        """A REAL client disconnect mid-proxy releases the slot and leaves the
        session usable by its next LEGAL continuation (200), not a stuck 409.

        Harness note: a real client socket abort does NOT cancel the in-flight
        handler in this harness (uvicorn + httpx backend). The handler stays
        parked in the proxy until the backend responds, then runs to completion
        and commits the (abandoned) backend response; the slot is released on
        that normal-completion `finally` (the `claimed` guard), not via an early
        cancellation. So after a disconnect the session has advanced exactly one
        committed turn. Recovery is therefore via a LEGAL append-only
        continuation of that committed turn — a fresh first-turn message would
        instead fail the append-only prefix check (400), which is expected, not
        a slot leak.

        We (1) fire a chat with a tiny client timeout against a high backend
        `latency` so `requests` aborts mid-proxy; (2) bounded-poll until a probe
        chat is no longer 409 (slot released — no fixed ordering sleep); then
        (3) read the committed turn via GET /sessions/{id} and prove a legal
        append-only continuation (committed user + assistant, then an allowed
        appended `system` message) returns 200.
        """

        # High latency: the client times out long before the backend responds,
        # so the abort lands while the handler is parked mid-proxy.
        latency = 1.0

        def process_fn(prompt: str) -> ProcessResult:
            return ProcessResult(text="ok", finish_reason="stop")

        with _router_env(process_fn, latency=latency) as env:
            session_id = _create_session(env.url)

            env.backend.reset_stats()
            # Fire the owner with a tiny timeout so `requests` aborts the
            # connection while the handler is parked in the latency window.
            try:
                _chat(env.url, session_id, _normal_messages("disconnect-me"), timeout=0.1)
            except requests.exceptions.RequestException:
                pass
            # Confirm the request actually reached the backend (handler was
            # genuinely parked mid-proxy when the client aborted).
            _wait_for_backend_requests(env.backend, 1)

            # The same-session slot must free up. Bounded-poll a probe chat until
            # it is NOT a 409: the abort did not cancel the parked handler, so the
            # gate stays busy until the owner completes, then releases. We use an
            # invalid (empty) probe payload so the probe itself never commits a
            # turn — it only reads the gate state (409 while busy, non-409 once
            # released) and leaves the committed trajectory untouched.
            deadline = time.time() + 10.0
            released = False
            last_status = None
            while time.time() < deadline:
                last_status = _chat(env.url, session_id, {"messages": []}, timeout=20.0).status_code
                if last_status != 409:
                    released = True
                    break
                time.sleep(0.05)
            assert released, f"slot not released after real disconnect (still {last_status} after retries)"

            # The disconnected owner's backend response committed one turn. Read
            # it back and build a LEGAL append-only continuation: the committed
            # user + assistant messages, then an allowed appended `system`
            # message (mirrors the retry payload in
            # test_same_session_second_chat_returns_409).
            get_resp = requests.get(f"{env.url}/sessions/{session_id}", timeout=5.0)
            assert get_resp.status_code == 200
            records = get_resp.json()["records"]
            assert records, "the abandoned backend response should have committed one turn"
            committed = records[-1]
            committed_user = committed["request"]["messages"]
            committed_assistant = committed["response"]["choices"][0]["message"]
            continuation = {
                "messages": [
                    *committed_user,
                    committed_assistant,
                    {"role": "system", "content": "continue-after-disconnect"},
                ]
            }
            after = _chat(env.url, session_id, continuation, timeout=20.0)
            assert after.status_code == 200, (
                f"legal same-session continuation after a real disconnect must 200, got {after.status_code}: "
                f"{after.text}"
            )


class TestBuildProxyResponse:
    """Unit tests for SessionServer.build_proxy_response passthrough fidelity.
    No running server needed: with no hf_checkpoint, setup_session_routes
    returns early so construction is light.
    """

    def _server(self) -> SessionServer:
        return SessionServer(SimpleNamespace(miles_router_timeout=30), backend_url="http://unused")

    def test_json_200_body_and_headers_passthrough(self):
        server = self._server()
        body = b'{"object":"chat.completion","choices":[]}'
        result = {
            "response_body": body,
            "status_code": 200,
            "headers": {
                "content-type": "application/json",
                "content-length": "999",
                "transfer-encoding": "chunked",
                "content-encoding": "gzip",
            },
        }
        resp = server.build_proxy_response(result)
        assert isinstance(resp, Response)
        assert resp.status_code == 200
        assert resp.body == body
        lowered = {k.lower() for k in resp.headers.keys()}
        assert resp.headers["content-type"].startswith("application/json")
        # The stale upstream content-length is dropped; transfer/content-encoding
        # are absent. Starlette recomputes content-length from the actual body, so
        # if present it must equal the body length (never the stale upstream "999").
        assert "transfer-encoding" not in lowered
        assert "content-encoding" not in lowered
        assert resp.headers.get("content-length", str(len(body))) == str(len(body))

    def test_non_json_200_body_and_media_type_preserved(self):
        server = self._server()
        body = b"plain text"
        result = {
            "response_body": body,
            "status_code": 200,
            "headers": {"content-type": "text/plain"},
        }
        resp = server.build_proxy_response(result)
        assert isinstance(resp, Response)
        assert resp.status_code == 200
        assert resp.body == body
        assert resp.media_type == "text/plain"
        assert resp.headers["content-type"].startswith("text/plain")

    def test_synthetic_502_status_and_body_passthrough(self):
        server = self._server()
        body = b'{"error":"backend transport error"}'
        result = {
            "response_body": body,
            "status_code": 502,
            "headers": {"content-type": "application/json"},
        }
        resp = server.build_proxy_response(result)
        assert isinstance(resp, Response)
        assert resp.status_code == 502
        assert resp.body == body

    def test_compressed_upstream_strips_stale_framing(self):
        server = self._server()
        body = b'{"ok":true}'
        result = {
            "response_body": body,
            "status_code": 200,
            "headers": {
                "content-type": "application/json",
                "content-encoding": "gzip",
                "content-length": "5",
            },
        }
        resp = server.build_proxy_response(result)
        assert isinstance(resp, Response)
        assert resp.body == body
        lowered = {k.lower() for k in resp.headers.keys()}
        # content-encoding stripped; the stale upstream content-length ("5") is
        # dropped — Starlette recomputes from the body, never carries the stale one.
        assert "content-encoding" not in lowered
        assert resp.headers.get("content-length", str(len(body))) == str(len(body))

    def test_generic_session_proxy_route_passes_through(self):
        """The generic /sessions/{id}/{path} route proxies and passes the body
        through: /abort_request on the mock backend returns {"status":"ok"} 200.
        """

        def process_fn(prompt: str) -> ProcessResult:
            return ProcessResult(text="ok", finish_reason="stop")

        with _router_env(process_fn) as env:
            session_id = _create_session(env.url)
            resp = requests.post(f"{env.url}/sessions/{session_id}/abort_request", timeout=10.0)
            assert resp.status_code == 200
            assert resp.json() == {"status": "ok"}


class TestPassthroughFidelity:
    """A successful chat response is passed through faithfully and stale
    framing headers from upstream are not copied to the client.
    """

    def test_successful_response_body_and_headers_passthrough(self):
        def process_fn(prompt: str) -> ProcessResult:
            return ProcessResult(text="passthrough-ok", finish_reason="stop")

        with _router_env(process_fn) as env:
            session_id = _create_session(env.url)
            resp = _chat(env.url, session_id, {"messages": [{"role": "user", "content": "hi"}]}, timeout=20.0)
            assert resp.status_code == 200

            # Body matches the upstream mock shape (id/object/choices content)
            # passed through unchanged, including the meta_info the mock adds.
            client_body = resp.json()
            assert client_body["object"] == "chat.completion"
            assert client_body["id"].startswith("chatcmpl-")
            assert client_body["choices"][0]["message"]["content"] == "passthrough-ok"
            assert "meta_info" in client_body["choices"][0]

            # Framing headers copied from upstream must have been stripped; the
            # body is intact (decodable JSON), so requests itself did not need
            # transfer/content-encoding framing from upstream.
            lowered = {k.lower() for k in resp.headers.keys()}
            assert "transfer-encoding" not in lowered
            assert "content-encoding" not in lowered
            assert resp.headers.get("content-type", "").startswith("application/json")


class TestResponseTokenIdValidation:
    """A malformed-but-200 upstream response must be rejected (502) rather than
    yielding non-integer token ids that would corrupt the stored trajectory."""

    @staticmethod
    def _payload(output_token_logprobs, completion_tokens=1) -> bytes:
        import json

        return json.dumps(
            {
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "ok"},
                        "meta_info": {
                            "output_token_logprobs": output_token_logprobs,
                            "completion_tokens": completion_tokens,
                        },
                    }
                ]
            }
        ).encode()

    def test_valid_integer_token_ids_accepted(self):
        _resp, _msg, ids = _parse_and_validate_response(self._payload([[-0.1, 123], [-0.2, 456]], completion_tokens=2))
        assert ids == [123, 456]

    def test_non_integer_token_ids_rejected(self):
        # str / None / float / bool token ids, a non-pair string entry, a short
        # entry, and a bool completion_tokens must all raise UpstreamResponseError.
        bad_cases = [
            ([[-0.1, "x"]], 1),
            ([[-0.1, None]], 1),
            ([[-0.1, 1.2]], 1),
            ([[-0.1, True]], 1),
            (["ab"], 1),
            ([[123]], 1),
            ([[-0.1, 123]], True),
        ]
        for logprobs, completion_tokens in bad_cases:
            try:
                _parse_and_validate_response(self._payload(logprobs, completion_tokens))
            except UpstreamResponseError:
                continue
            raise AssertionError(
                f"expected UpstreamResponseError for {logprobs!r}, completion_tokens={completion_tokens!r}"
            )

    def test_non_numeric_logprob_values_rejected(self):
        # entry[0] (the logprob) flows into Sample.rollout_log_probs downstream
        # (openai_endpoint_utils), so a str / None / bool / non-finite logprob with an
        # otherwise-valid token id must still raise rather than being accepted.
        bad_cases = [
            [["bad-logprob", 562]],
            [[None, 562]],
            [[True, 562]],
            [[float("nan"), 562]],
            [[float("inf"), 562]],
            [[float("-inf"), 562]],
        ]
        for logprobs in bad_cases:
            try:
                _parse_and_validate_response(self._payload(logprobs, completion_tokens=1))
            except UpstreamResponseError:
                continue
            raise AssertionError(f"expected UpstreamResponseError for logprob in {logprobs!r}")

    def test_non_standard_json_constants_rejected(self):
        for constant in ("NaN", "Infinity", "-Infinity"):
            raw = (
                b'{"choices":[{"message":{"role":"assistant","content":"ok"},'
                b'"meta_info":{"output_token_logprobs":[[' + constant.encode() + b',562]],"completion_tokens":1}}]}'
            )
            try:
                _parse_and_validate_response(raw)
            except UpstreamResponseError:
                continue
            raise AssertionError(f"expected UpstreamResponseError for JSON constant {constant}")

    def test_sglang_triples_with_token_text_accepted(self):
        # SGLang emits [logprob, token_id, token_text] triples; len > 2 is normal
        # and must NOT be rejected. Integer logprobs (e.g. 0) are also valid numbers.
        _resp, _msg, ids = _parse_and_validate_response(
            self._payload([[-0.1, 123, "he"], [0, 456, "llo"]], completion_tokens=2)
        )
        assert ids == [123, 456]
