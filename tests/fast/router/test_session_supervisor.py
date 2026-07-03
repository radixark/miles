"""Real-process supervisor: spawns worker+router processes, serves end-to-end over IPC,
and fail-fasts when a worker dies (so the rollout path raises instead of hanging).

Unlike test_session_dataplane (in-loop channels), this spawns actual OS processes via the
supervisor — the control-plane path that PR-C adds.
"""

import time
import uuid
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pytest

from miles.rollout.session.supervisor import SessionServerSupervisor
from miles.utils.http_utils import find_available_port
from miles.utils.test_utils.mock_sglang_server import MockSGLangServer, ProcessResult, with_mock_server


def _process_fn(prompt: str) -> ProcessResult:
    return ProcessResult(text=f"echo: {prompt}", finish_reason="stop")


def _patched_chat_response():
    original = MockSGLangServer._compute_chat_completions_response

    def patched(self, payload: dict) -> dict:
        response = original(self, payload)
        choice = response["choices"][0]
        otl = [
            (it["logprob"], self.tokenizer.convert_tokens_to_ids(it["token"])) for it in choice["logprobs"]["content"]
        ]
        choice["meta_info"] = {"output_token_logprobs": otl, "completion_tokens": len(otl)}
        return response

    return patched


def _args():
    return SimpleNamespace(
        miles_router_timeout=30,
        hf_checkpoint="Qwen/Qwen3-0.6B",
        chat_template_path=None,
        apply_chat_template_kwargs={"enable_thinking": False},
        tito_model="default",
        tito_allowed_append_roles=["tool"],
        session_server_instance_id=uuid.uuid4().hex,
        session_server_workers=2,
    )


def test_supervisor_serves_then_fails_fast_on_worker_death():
    with patch.object(MockSGLangServer, "_compute_chat_completions_response", new=_patched_chat_response()):
        with with_mock_server(process_fn=_process_fn) as backend:
            ip, port = "127.0.0.1", find_available_port(45000)
            sup = SessionServerSupervisor(_args(), backend.url, ip, port)
            sup.start()
            base = f"http://{ip}:{port}"
            try:
                # End-to-end through real router + worker processes.
                assert httpx.get(f"{base}/health", timeout=10).status_code == 200
                sup.check()  # healthy → no raise

                sid = httpx.post(f"{base}/sessions", timeout=10).json()["session_id"]
                chat = httpx.post(
                    f"{base}/sessions/{sid}/v1/chat/completions",
                    json={"messages": [{"role": "user", "content": "hi"}]},
                    timeout=30,
                )
                assert chat.status_code == 200, chat.text
                assert len(httpx.get(f"{base}/sessions/{sid}", timeout=10).json()["records"]) == 1

                # Kill a worker → the monitor must record it and check() must raise.
                sup._procs[0].kill()
                deadline = time.monotonic() + 10
                while time.monotonic() < deadline:
                    try:
                        sup.check()
                    except RuntimeError:
                        break
                    time.sleep(0.2)
                with pytest.raises(RuntimeError, match="session server failed"):
                    sup.check()
            finally:
                sup.shutdown()
            # No orphans. Teardown may be running in the monitor thread (SIGTERM + grace +
            # SIGKILL), so wait for the children to be reaped before asserting.
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline and any(p.is_alive() for p in sup._procs):
                time.sleep(0.2)
            assert all(not p.is_alive() for p in sup._procs)
