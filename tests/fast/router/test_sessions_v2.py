"""HTTP tests for session server v2 (tree serving, --use-session-server v2).

The v1 HTTP surface keeps its own modules (test_sessions.py untouched at
base + test_sessions_v1_pins.py); everything here runs against a v2-flagged
server. Helpers and the class matrix are carried verbatim from the v5
development line so the behavior pins stay byte-comparable.
"""

import json
import uuid
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import requests
import safetensors.numpy
from fastapi.responses import JSONResponse
from tests.fast.router.test_sessions import _create_session, _post_chat

from miles.rollout.session.server import SessionServer
from miles.utils.http_utils import find_available_port
from miles.utils.test_utils.mock_sglang_server import MockSGLangServer, ProcessResult, with_mock_server
from miles.utils.test_utils.uvicorn_thread_server import UvicornThreadServer


@contextmanager
def _serve_router(extra_args: dict | None = None):
    """A standalone v2 SessionServer with arg overrides."""

    def process_fn(prompt: str) -> ProcessResult:
        return ProcessResult(text=f"echo: {prompt}", finish_reason="stop")

    with with_mock_server(process_fn=process_fn) as backend:
        args = SimpleNamespace(
            miles_router_timeout=30,
            hf_checkpoint="Qwen/Qwen3-0.6B",
            chat_template_path=None,
            apply_chat_template_kwargs={"enable_thinking": False},
            tito_model="default",
            tito_allowed_append_roles=["tool"],
            use_session_server="v2",
            session_server_instance_id=uuid.uuid4().hex,
            **(extra_args or {}),
        )
        server_obj = SessionServer(args, backend_url=backend.url)
        port = find_available_port(31000)
        server = UvicornThreadServer(server_obj.app, host="127.0.0.1", port=port)
        server.start()
        try:
            yield SimpleNamespace(url=f"http://127.0.0.1:{port}", backend=backend)
        finally:
            server.stop()


@pytest.fixture(scope="class")
def router_env():
    """v2 twin of test_sessions.router_env: same mock backend + R3 payloads,
    tree serving enabled."""

    def process_fn(prompt: str) -> ProcessResult:
        return ProcessResult(text=f"echo: {prompt}", finish_reason="stop")

    original_chat_response = MockSGLangServer._compute_chat_completions_response

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
            # R3 replay payloads: must reach the session record but never the
            # client-facing chat response (see _strip_replay_payloads).
            "routed_experts": [[0, 1], [2, 3]],
            "indexer_topk": [[4], [5]],
        }
        return response

    with patch.object(MockSGLangServer, "_compute_chat_completions_response", new=patched_chat_response):
        with _serve_router() as env:
            yield env


class TestRollbackPins:
    """HTTP-level pins for the rollback dispatch surface (retry semantics).

    These lock today's behavior before the classify/apply split rewires the
    internals (MULTI_LINEAGE_DESIGN.md, milestone M2): the unit-level
    TestRollback suite is legitimately rewritten by that split, so byte-level
    fidelity must live at this layer. Every 400 text is asserted exactly,
    including interpolated numbers.
    """

    U1 = {"role": "user", "content": "What is 1+2?"}
    T1 = {"role": "tool", "content": "tool-result-1", "tool_call_id": "t0"}
    T1_DIFF = {"role": "tool", "content": "tool-result-DIFFERENT", "tool_call_id": "t0"}

    def _turn(self, url: str, session_id: str, messages: list) -> dict:
        resp = _post_chat(url, session_id, {"messages": messages})
        assert resp.status_code == 200
        return resp.json()["choices"][0]["message"]

    def _get(self, url: str, session_id: str) -> dict:
        return requests.get(f"{url}/sessions/{session_id}", timeout=5.0).json()

    def _two_turn_session(self, env) -> tuple[str, dict, dict]:
        """Stored history after this: [U1, a1, T1, a2] with 2 records."""
        session_id = _create_session(env.url)
        a1 = self._turn(env.url, session_id, [self.U1])
        a2 = self._turn(env.url, session_id, [self.U1, a1, self.T1])
        assert len(self._get(env.url, session_id)["records"]) == 2
        return session_id, a1, a2

    def test_pure_drop_retry_rolls_back_and_regenerates(self, router_env):
        session_id, a1, _ = self._two_turn_session(router_env)

        retry = _post_chat(router_env.url, session_id, {"messages": [self.U1, a1, self.T1]})

        assert retry.status_code == 200
        records = self._get(router_env.url, session_id)["records"]
        assert len(records) == 2
        assert records[-1]["request"]["messages"][-1] == self.T1

    def test_divergent_retry_rolls_back_and_continues(self, router_env):
        session_id, a1, _ = self._two_turn_session(router_env)

        retry = _post_chat(router_env.url, session_id, {"messages": [self.U1, a1, self.T1_DIFF]})

        assert retry.status_code == 200
        records = self._get(router_env.url, session_id)["records"]
        assert len(records) == 2
        assert records[-1]["request"]["messages"][-1] == self.T1_DIFF

    def test_deep_divergence_branches_and_keeps_both_lines(self, router_env):
        """Was the deep-rollback 400 pin: a divergence beyond one generation now
        branches at the deep anchor (200), the abandoned deep line stays in the
        tree, and the samples op emits BOTH lines (deep abandons are data)."""
        with _clean_r3_meta():
            session_id, a1, a2 = self._two_turn_session(router_env)
            t2 = {"role": "tool", "content": "tool-result-2", "tool_call_id": "t1"}
            self._turn(router_env.url, session_id, [self.U1, a1, self.T1, a2, t2])
            assert len(self._get(router_env.url, session_id)["records"]) == 3

            resp = _post_chat(router_env.url, session_id, {"messages": [self.U1, a1, self.T1_DIFF]})

            assert resp.status_code == 200
            after = self._get(router_env.url, session_id)
            # The view follows the new branch: anchor turn + the fresh generation.
            assert len(after["records"]) == 2
            tree = after["metadata"]["tree"]
            assert len(tree["nodes"]) == 4
            assert sorted(len(leaf["path_node_ids"]) for leaf in tree["leaves"]) == [2, 3]

            samples = requests.post(f"{router_env.url}/sessions/{session_id}/samples", json={}, timeout=10.0)
            assert samples.status_code == 200
            meta = _decode_samples_meta(samples.content)
            assert len(meta["samples"]) == 2  # the deep abandoned line still trains

    def test_root_divergence_opens_new_root(self, router_env):
        """Was the no-anchor 400 pin: divergence inside the root delta now opens
        a second root (200); both roots produce samples."""
        with _clean_r3_meta():
            session_id = _create_session(router_env.url)
            self._turn(router_env.url, session_id, [self.U1])

            resp = _post_chat(
                router_env.url, session_id, {"messages": [{"role": "user", "content": "a different opening"}]}
            )

            assert resp.status_code == 200
            after = self._get(router_env.url, session_id)
            assert len(after["records"]) == 1  # view follows the new root
            tree = after["metadata"]["tree"]
            assert [n["parent"] for n in tree["nodes"]] == [None, None]

            samples = requests.post(f"{router_env.url}/sessions/{session_id}/samples", json={}, timeout=10.0)
            assert samples.status_code == 200
            assert len(_decode_samples_meta(samples.content)["samples"]) == 2

    def test_failed_first_turn_then_different_first_request_accepted(self, router_env):
        """A failed first turn records nothing, so a completely different first
        request is a fresh first turn and must be accepted — the empty-session
        accept-anything semantics that later dispatch rewrites must preserve."""
        session_id = _create_session(router_env.url)

        async def reject(self, request, compute_fn):
            return JSONResponse(content={"error": "context too long"}, status_code=400)

        with patch.object(MockSGLangServer, "_handle_generate_like_request", new=reject):
            failed = _post_chat(router_env.url, session_id, {"messages": [self.U1]})
        assert failed.status_code == 400
        assert self._get(router_env.url, session_id)["records"] == []

        other_first = {"role": "user", "content": "a completely different opening"}
        resp = _post_chat(router_env.url, session_id, {"messages": [other_first]})
        assert resp.status_code == 200
        records = self._get(router_env.url, session_id)["records"]
        assert len(records) == 1
        assert records[0]["request"]["messages"] == [other_first]

    def test_degenerate_extension_resend_exact_history(self, router_env):
        session_id = _create_session(router_env.url)
        a1 = self._turn(router_env.url, session_id, [self.U1])

        resend = _post_chat(router_env.url, session_id, {"messages": [self.U1, a1]})

        assert resend.status_code == 200
        assert len(self._get(router_env.url, session_id)["records"]) == 2

    def test_disallowed_append_role_400_with_rollback_side_effect(self, router_env):
        session_id, a1, _ = self._two_turn_session(router_env)

        resp = _post_chat(
            router_env.url, session_id, {"messages": [self.U1, a1, {"role": "user", "content": "another"}]}
        )

        assert resp.status_code == 400
        error = resp.json()["error"]
        assert error.endswith("; to allow more roles use --tito-allowed-append-roles")
        # Characterization: today the rollback mutates BEFORE the append-only
        # check rejects, and the 400 leaves the rolled-back state behind. The
        # classify/apply split must keep this order.
        assert len(self._get(router_env.url, session_id)["records"]) == 1

    def test_collect_samples_after_rollback_single_sample(self, router_env):
        from miles.rollout.session.samples.codec import decode_samples_reply
        from miles.utils.types import Sample

        # The class fixture plants fake R3 replay payloads that only the
        # records path tolerates; assembly would try to decode them. This pin
        # is about rollback x assembly, so run it with clean meta_info.
        fixture_response = MockSGLangServer._compute_chat_completions_response

        def clean_meta_response(mock_self, payload: dict) -> dict:
            response = fixture_response(mock_self, payload)
            meta = response["choices"][0]["meta_info"]
            meta.pop("routed_experts", None)
            meta.pop("indexer_topk", None)
            return response

        with patch.object(MockSGLangServer, "_compute_chat_completions_response", new=clean_meta_response):
            session_id, a1, _ = self._two_turn_session(router_env)
            retry = _post_chat(router_env.url, session_id, {"messages": [self.U1, a1, self.T1]})
            assert retry.status_code == 200

            resp = requests.post(f"{router_env.url}/sessions/{session_id}/samples", json={}, timeout=10.0)

        assert resp.status_code == 200
        reply = decode_samples_reply(resp.content, Sample())
        assert reply.empty_reason is None
        assert len(reply.samples) == 1
        [sample] = reply.samples
        assert sample.response_length > 0
        assert len(sample.loss_mask) == sample.response_length

    def test_few_shot_first_request_divergent_retry_rolls_back_cleanly(self, router_env):
        """Assistants carried by the first request are prompt, not checkpoints.

        Historically the rollback anchor math counted the few-shot assistant
        as a checkpoint, so this divergent retry computed discard_count=0,
        kept the stale second checkpoint, and answered 200 with a corrupted
        token stream (records grew to 3). With ``prompt_assistant_count`` the
        anchor is the generated assistant: one-step rollback + regenerate,
        records land at 2."""
        few_shot = [
            {"role": "user", "content": "Q-few-shot"},
            {"role": "assistant", "content": "A-few-shot"},
            {"role": "user", "content": "Q-real"},
        ]
        session_id = _create_session(router_env.url)
        a1 = self._turn(router_env.url, session_id, few_shot)
        self._turn(router_env.url, session_id, [*few_shot, a1, self.T1])
        assert len(self._get(router_env.url, session_id)["records"]) == 2

        retry = _post_chat(router_env.url, session_id, {"messages": [*few_shot, a1, self.T1_DIFF]})

        assert retry.status_code == 200
        assert len(self._get(router_env.url, session_id)["records"]) == 2


@contextmanager
def _clean_r3_meta():
    """The class fixture plants fake R3 replay payloads that only the records
    path tolerates; assembly would try to decode them. Samples-op tests run
    with clean meta_info."""
    fixture_response = MockSGLangServer._compute_chat_completions_response

    def clean_meta_response(mock_self, payload: dict) -> dict:
        response = fixture_response(mock_self, payload)
        meta = response["choices"][0]["meta_info"]
        meta.pop("routed_experts", None)
        meta.pop("indexer_topk", None)
        return response

    with patch.object(MockSGLangServer, "_compute_chat_completions_response", new=clean_meta_response):
        yield


def _decode_samples_meta(payload: bytes) -> dict:
    tensors = safetensors.numpy.load(payload)
    return json.loads(tensors["_samples_meta"].tobytes().decode("utf-8"))


class TestStrictAppendOnly:
    """--session-strict-append-only: the single-chain invariant guard."""

    U1 = {"role": "user", "content": "What is 1+2?"}
    T1 = {"role": "tool", "content": "tool-result-1", "tool_call_id": "t0"}

    def _turn(self, url, session_id, messages):
        resp = _post_chat(url, session_id, {"messages": messages})
        assert resp.status_code == 200
        return resp.json()["choices"][0]["message"]

    def test_extension_ok_but_branch_shapes_fail_loud(self):
        with _serve_router({"session_strict_append_only": True}) as env:
            session_id = _create_session(env.url)
            a1 = self._turn(env.url, session_id, [self.U1])

            # Strict extension still works.
            a2 = self._turn(env.url, session_id, [self.U1, a1, self.T1])
            assert a2["content"]

            # A retry shape would create a sibling: loud 400 with diagnostics.
            t1_diff = {"role": "tool", "content": "DIFFERENT", "tool_call_id": "t0"}
            retry = _post_chat(env.url, session_id, {"messages": [self.U1, a1, t1_diff]})
            assert retry.status_code == 400
            assert "append-only" in retry.json()["error"]

            # A second root: loud 400 too.
            reroot = _post_chat(env.url, session_id, {"messages": [{"role": "user", "content": "other"}]})
            assert reroot.status_code == 400
            assert "append-only" in reroot.json()["error"]

            # State untouched by the rejections.
            records = requests.get(f"{env.url}/sessions/{session_id}", timeout=5.0).json()["records"]
            assert len(records) == 2

    def test_failed_first_turn_still_retryable(self):
        with _serve_router({"session_strict_append_only": True}) as env:
            session_id = _create_session(env.url)

            async def reject(self, request, compute_fn):
                return JSONResponse(content={"error": "context too long"}, status_code=400)

            with patch.object(MockSGLangServer, "_handle_generate_like_request", new=reject):
                failed = _post_chat(env.url, session_id, {"messages": [self.U1]})
            assert failed.status_code == 400

            other_first = {"role": "user", "content": "a completely different opening"}
            assert _post_chat(env.url, session_id, {"messages": [other_first]}).status_code == 200


class TestTruncationAndCompaction:
    U1 = {"role": "user", "content": "What is 1+2?"}
    T1 = {"role": "tool", "content": "tool-result-1", "tool_call_id": "t0"}

    def test_extending_truncated_generation_is_409(self, router_env):
        session_id = _create_session(router_env.url)
        fixture_response = MockSGLangServer._compute_chat_completions_response

        def length_finish(self, payload: dict) -> dict:
            response = fixture_response(self, payload)
            response["choices"][0]["finish_reason"] = "length"
            return response

        with patch.object(MockSGLangServer, "_compute_chat_completions_response", new=length_finish):
            first = _post_chat(router_env.url, session_id, {"messages": [self.U1]})
        assert first.status_code == 200
        a1 = first.json()["choices"][0]["message"]

        extend = _post_chat(router_env.url, session_id, {"messages": [self.U1, a1, self.T1]})
        assert extend.status_code == 409
        assert "truncated generation cannot be extended" in extend.json()["error"]

        # Branching BEFORE the cut still works: a different opening opens a new root.
        reroot = _post_chat(router_env.url, session_id, {"messages": [{"role": "user", "content": "fresh"}]})
        assert reroot.status_code == 200

    def test_compaction_branch_with_carried_assistant(self, router_env):
        """A branch delta carrying a client assistant (compaction) is accepted:
        the carried assistant is prompt (loss 0 comes from assembly), and the
        inherited snapshot splices with the canonical suffix render."""
        session_id = _create_session(router_env.url)
        first = _post_chat(router_env.url, session_id, {"messages": [self.U1]})
        a1 = first.json()["choices"][0]["message"]

        carried = {"role": "assistant", "content": "compacted summary of earlier work"}
        resp = _post_chat(
            router_env.url,
            session_id,
            {"messages": [self.U1, a1, self.T1, carried, {"role": "tool", "content": "go", "tool_call_id": "t1"}]},
        )
        assert resp.status_code == 200
        meta = requests.get(f"{router_env.url}/sessions/{session_id}", timeout=5.0).json()["metadata"]
        # Two generations in the tree; whether the branch inherited the parent
        # snapshot or degraded to a zero-inheritance root depends on the
        # family's canonical-render fidelity (Qwen3's empty-think prompt block
        # forces the root fallback here) — both are valid serving outcomes.
        assert len(meta["tree"]["nodes"]) == 2
