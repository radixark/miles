"""HTTP-level pins for the v1 (linear trajectory) retry/rollback surface.

v1 is the default session server; these pins lock its production behavior
byte-exactly — every 400 text asserted including interpolated numbers — so
the opt-in v2 tree server (--use-session-server v2) can never silently
change the default path. Additive file: the v1 implementation and its
original test modules stay untouched.
"""

from unittest.mock import patch

import requests
from fastapi.responses import JSONResponse
from tests.fast.router.test_sessions import _create_session, _post_chat

from miles.utils.test_utils.mock_sglang_server import MockSGLangServer


class TestRollbackPins:
    """Byte-exact pins for the rollback dispatch surface (retry semantics)."""

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

    def test_deep_rollback_400_byte_exact_and_state_unchanged(self, router_env):
        session_id, a1, a2 = self._two_turn_session(router_env)
        t2 = {"role": "tool", "content": "tool-result-2", "tool_call_id": "t1"}
        a3 = self._turn(router_env.url, session_id, [self.U1, a1, self.T1, a2, t2])
        before = self._get(router_env.url, session_id)
        assert len(before["records"]) == 3

        resp = _post_chat(router_env.url, session_id, {"messages": [self.U1, a1, self.T1_DIFF]})

        assert resp.status_code == 400
        assert resp.json()["error"] == (
            "rollback failed: discard_count=2 exceeds max_assistant_rollback_steps=1 "
            "(stored has 6 messages, request has 3 messages)"
        )
        after = self._get(router_env.url, session_id)
        assert after["records"] == before["records"]
        assert after["metadata"]["accumulated_token_ids"] == before["metadata"]["accumulated_token_ids"]

        t3 = {"role": "tool", "content": "tool-result-3", "tool_call_id": "t2"}
        extend = _post_chat(router_env.url, session_id, {"messages": [self.U1, a1, self.T1, a2, t2, a3, t3]})
        assert extend.status_code == 200

    def test_no_anchor_400_byte_exact_and_state_unchanged(self, router_env):
        session_id = _create_session(router_env.url)
        a1 = self._turn(router_env.url, session_id, [self.U1])
        before = self._get(router_env.url, session_id)

        resp = _post_chat(router_env.url, session_id, {"messages": [self.U1]})

        assert resp.status_code == 400
        assert resp.json()["error"] == (
            "rollback failed: no assistant message found in the first 1 matched messages "
            "(stored has 2 messages, request has 1 messages)"
        )
        after = self._get(router_env.url, session_id)
        assert after["records"] == before["records"]
        assert after["metadata"]["accumulated_token_ids"] == before["metadata"]["accumulated_token_ids"]

        extend = _post_chat(router_env.url, session_id, {"messages": [self.U1, a1, self.T1]})
        assert extend.status_code == 200

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
        # check rejects, and the 400 leaves the rolled-back state behind. Any
        # future rework of v1 must keep this order.
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

    def test_few_shot_first_request_divergent_retry_characterization(self, router_env):
        """Characterization of v1's prompt-assistant anchoring quirk.

        The first request carries a few-shot assistant; the rollback anchor
        math counts it as a checkpoint, so a divergent retry computes
        discard_count=0, keeps the stale second checkpoint, and answers 200
        with a corrupted token stream: records grow to 3 instead of rolling
        back to 1 + regenerating (=2). Few-shot / carried-assistant shapes
        are outside v1's supported envelope (append-only + one-step retry);
        flows that need them should run session server v2."""
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
        assert len(self._get(router_env.url, session_id)["records"]) == 3
