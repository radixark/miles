"""E2E session stress tests.

Contract under test (with session.lock / session.closing):
- Requests within the same session are serialized via session.lock.
- Different sessions can run in parallel (no global lock).
- Per-session clients can run turn-by-turn without idle gaps while global load stays parallel.
- Delete marks session.closing=True, waits for session.lock, then removes.
- Chat requests to a closing session get 404 immediately (pre-lock check).
- Chat requests arriving while delete waits for lock get 404 (double-check after lock).
- Concurrent deletes on the same session: second delete gets 404.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import requests

from miles.rollout.session.session_server import SessionServer
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


@contextmanager
def _router_env(process_fn, *, latency: float = 0.0):
    with _patch_mock_chat_response():
        with with_mock_server(model_name=HF_CHECKPOINT, process_fn=process_fn, latency=latency) as backend:
            args = SimpleNamespace(
                miles_router_timeout=30,
                hf_checkpoint=HF_CHECKPOINT,
                chat_template_path=None,
                trajectory_manager="single_user_turn_trajectory",
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


class TestSessionConcurrencyContracts:
    def test_same_session_requests_are_serialized(self):
        def process_fn(prompt: str) -> ProcessResult:
            return ProcessResult(text="serialized-ok", finish_reason="stop")

        with _router_env(process_fn, latency=0.2) as env:
            session_id = _create_session(env.url)

            # Warm up one assistant checkpoint so repeated identical retry payloads are valid.
            warmup_payload = {"messages": [{"role": "user", "content": "warmup"}]}
            warmup_resp = _chat(env.url, session_id, warmup_payload)
            assert warmup_resp.status_code == 200
            assistant = warmup_resp.json()["choices"][0]["message"]

            retry_payload = {
                "messages": [
                    {"role": "user", "content": "warmup"},
                    assistant,
                    {"role": "system", "content": "retry-from-assistant-checkpoint"},
                ]
            }

            env.backend.reset_stats()
            with ThreadPoolExecutor(max_workers=4) as pool:
                futures = [pool.submit(_chat, env.url, session_id, retry_payload) for _ in range(4)]
                responses = [f.result(timeout=30.0) for f in futures]

            assert all(resp.status_code == 200 for resp in responses)
            assert len(env.backend.request_log) == 4
            assert env.backend.max_concurrent == 1

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

    def test_delete_waits_for_inflight_then_removes_session(self):
        def process_fn(prompt: str) -> ProcessResult:
            return ProcessResult(text="slow-turn", finish_reason="stop")

        with _router_env(process_fn, latency=0.35) as env:
            session_id = _create_session(env.url)
            payload = {"messages": [{"role": "user", "content": "slow-turn-0"}]}

            with ThreadPoolExecutor(max_workers=1) as pool:
                inflight = pool.submit(_chat, env.url, session_id, payload, 30.0)

                # Wait until the first request has reached backend before deleting.
                deadline = time.time() + 5.0
                while time.time() < deadline:
                    if env.backend.request_log:
                        break
                    time.sleep(0.01)
                else:
                    raise AssertionError("in-flight request did not reach backend in time")

                t0 = time.perf_counter()
                delete_resp = requests.delete(f"{env.url}/sessions/{session_id}", timeout=30.0)
                delete_elapsed = time.perf_counter() - t0

                inflight_resp = inflight.result(timeout=30.0)

            assert inflight_resp.status_code == 200
            assert delete_resp.status_code == 204
            # Delete should wait for in-flight request to drain.
            assert delete_elapsed >= 0.2
            # Session is gone after delete.
            post_delete = _chat(env.url, session_id, payload, timeout=10.0)
            assert post_delete.status_code == 404


class TestClosingRaceConditions:
    """Tests for race conditions around session.closing flag."""

    def test_chat_during_delete_wait_returns_404(self):
        """Chat requests arriving while delete is waiting for the lock get 404.

        Timeline:
        1. Chat A starts (holds session.lock, slow backend)
        2. Delete arrives, sets session.closing=True, waits for session.lock
        3. Chat B arrives, sees session.closing=True, returns 404 immediately
        4. Chat A finishes, delete acquires lock, removes session
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

    def test_multiple_chats_queued_then_delete(self):
        """Multiple chat requests queued behind session.lock, then delete.

        After delete marks closing=True, queued chats that acquire the lock
        should check closing and return 404.
        """

        def process_fn(prompt: str) -> ProcessResult:
            return ProcessResult(text="queued-ok", finish_reason="stop")

        with _router_env(process_fn, latency=0.3) as env:
            session_id = _create_session(env.url)
            payload = {"messages": [{"role": "user", "content": "queued"}]}

            with ThreadPoolExecutor(max_workers=6) as pool:
                # Fire 3 chats (first holds lock, others queue)
                chat_futures = [pool.submit(_chat, env.url, session_id, payload, 30.0) for _ in range(3)]

                # Wait for first to reach backend
                deadline = time.time() + 5.0
                while time.time() < deadline:
                    if env.backend.request_log:
                        break
                    time.sleep(0.01)

                # Now delete - sets closing, waits for first chat to finish
                delete_future = pool.submit(
                    requests.delete,
                    f"{env.url}/sessions/{session_id}",
                    timeout=30.0,
                )

                results = [f.result(timeout=30.0) for f in chat_futures]
                delete_resp = delete_future.result(timeout=30.0)

            assert delete_resp.status_code == 204

            # At least one chat must succeed (the one holding the lock when
            # delete arrived).  Others may get 200 (acquired lock before
            # closing) or 404 (saw closing=True).  No 500s allowed.
            status_codes = [r.status_code for r in results]
            assert all(c in (200, 404) for c in status_codes), f"Unexpected status codes: {status_codes}"
            assert 200 in status_codes, f"Expected at least one 200, got {status_codes}"

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
