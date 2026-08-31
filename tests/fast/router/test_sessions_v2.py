"""HTTP tests for session server v2 (tree serving, --use-session-server v2).

The v1 HTTP surface keeps its own modules (test_sessions.py untouched at
base + test_sessions_v1_pins.py); everything here runs against a v2-flagged
server. The rollback pin classes mirror test_sessions_v1_pins.py so v1/v2
behavior stays byte-comparable.
"""

import asyncio
import json
import uuid
from concurrent.futures import ThreadPoolExecutor
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
from miles.utils.lora import LORA_ADAPTER_NAME
from miles.utils.misc import function_registry
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
            sglang_speculative_algorithm=None,
            use_session_server="v2",
            session_server_instance_id=uuid.uuid4().hex,
            save_debug_trajectory_data=None,
            session_sample_picker_path="miles.rollout.session.v2.picker_hub.drop_retries",
            session_sample_postprocessor_path="miles.rollout.session.v2.postprocessor_hub.default_postprocess",
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


class TestRequestChatTemplateKwargs:
    def test_override_reaches_render_and_backend(self, router_env):
        default_session = _create_session(router_env.url)
        resp = _post_chat(router_env.url, default_session, {"messages": [{"role": "user", "content": "hi"}]})
        assert resp.status_code == 200
        default_payload = router_env.backend.request_log[-1]
        assert default_payload["chat_template_kwargs"] == {"enable_thinking": False}

        override_session = _create_session(router_env.url)
        resp = _post_chat(
            router_env.url,
            override_session,
            {
                "messages": [{"role": "user", "content": "hi"}],
                "chat_template_kwargs": {"enable_thinking": True},
            },
        )
        assert resp.status_code == 200
        override_payload = router_env.backend.request_log[-1]
        assert override_payload["chat_template_kwargs"] == {"enable_thinking": True}
        assert override_payload["input_ids"] != default_payload["input_ids"]


def test_lora_adapter_reaches_backend():
    with _serve_router({"lora_rank": 8}) as env:
        session_id = _create_session(env.url)
        response = _post_chat(env.url, session_id, {"messages": [{"role": "user", "content": "hi"}]})

        assert response.status_code == 200
        assert env.backend.request_log[-1]["lora_path"] == LORA_ADAPTER_NAME


def _keep_all_picker(leaf_samples, _session_metadata):
    return list(leaf_samples)


def test_concurrent_requests_from_same_parent_commit_siblings():
    """Successful concurrent generations from one parent are both collected."""
    picker_path = "miles.rollout.session.v2.picker_hub.drop_retries"
    with function_registry.temporary(picker_path, _keep_all_picker):
        with _serve_router() as env:
            session_id = _create_session(env.url)
            first = _post_chat(env.url, session_id, {"messages": [{"role": "user", "content": "start"}]})
            assert first.status_code == 200
            assistant = first.json()["choices"][0]["message"]
            payload = {
                "messages": [
                    {"role": "user", "content": "start"},
                    assistant,
                    {"role": "user", "content": "branch"},
                ]
            }

            arrivals = 0
            release = None

            async def wait_for_pair(self, request, compute_fn):
                nonlocal arrivals, release
                payload = await request.json()
                self.request_log.append(payload)
                if release is None:
                    release = asyncio.Event()
                arrivals += 1
                if arrivals == 2:
                    release.set()
                await asyncio.wait_for(release.wait(), timeout=5.0)
                return JSONResponse(content=compute_fn(payload))

            with patch.object(MockSGLangServer, "_handle_generate_like_request", new=wait_for_pair):
                with ThreadPoolExecutor(max_workers=2) as pool:
                    futures = [pool.submit(_post_chat, env.url, session_id, payload) for _ in range(2)]
                    responses = [future.result(timeout=10.0) for future in futures]

            assert all(response.status_code == 200 for response in responses)
            response_ids = {response.json()["id"] for response in responses}
            tree = requests.get(f"{env.url}/sessions/{session_id}", timeout=5.0).json()["metadata"]["tree"]
            parent_id = tree["nodes"][0]["id"]
            children = tree["nodes"][1:]
            assert len(children) == 2
            assert {child["parent"] for child in children} == {parent_id}
            assert {child["response_id"] for child in children} == response_ids

            samples = requests.post(f"{env.url}/sessions/{session_id}/samples", json={}, timeout=10.0)
            assert samples.status_code == 200
            sample_meta = _decode_samples_meta(samples.content)
            assert {sample["metadata"]["leaf"]["response_id"] for sample in sample_meta["samples"]} == response_ids


class TestRollbackPins:
    """HTTP-level pins for the rollback dispatch surface (retry semantics).

    These lock today's behavior before the classify/apply split rewires the
    internals: the unit-level TestRollback suite is legitimately rewritten by
    that split, so byte-level fidelity must live at this layer. Every 400 text
    is asserted exactly, including interpolated numbers.
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
            for wire_sample in meta["samples"]:
                # v2 serving fills SessionRecord.request_timestamp like v1 does
                # (folding collects per-turn segments into a list).
                lifecycle = wire_sample["metadata"]["lifecycle"]
                segments = lifecycle if isinstance(lifecycle, list) else [lifecycle]
                assert segments and all(seg["req_ts"] is not None for seg in segments)

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
            router_env.url, session_id, {"messages": [self.U1, a1, {"role": "developer", "content": "another"}]}
        )

        assert resp.status_code == 400
        error = resp.json()["error"]
        assert error.endswith("; the selected TITO fixed template does not support appending this role")
        # Characterization: today the rollback mutates BEFORE the append-only
        # check rejects, and the 400 leaves the rolled-back state behind. The
        # classify/apply split must keep this order.
        assert len(self._get(router_env.url, session_id)["records"]) == 1

    def test_collect_samples_after_rollback_single_sample(self, router_env):
        from miles.rollout.session.samples.codec import decode_samples_and_merge_input_sample
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
        reply = decode_samples_and_merge_input_sample(resp.content, Sample())
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
        suffix render appends onto the inherited snapshot."""
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
        # Snapshot inheritance is unconditional: the carried-assistant branch
        # commits as a child of the first generation, never as a second root.
        nodes = meta["tree"]["nodes"]
        assert len(nodes) == 2
        assert nodes[1]["parent"] == nodes[0]["id"]


# ── additional R3 (colocated abort and in-place weight updates): request offsets on the tree ──


class TestAdditionR3RequestOffsetV2:
    MESSAGES = [{"role": "user", "content": "hi"}]

    def _accumulated(self, url: str, session_id: str) -> list[int]:
        data = requests.get(f"{url}/sessions/{session_id}", timeout=5.0).json()
        return data["metadata"]["accumulated_token_ids"]

    @pytest.mark.parametrize("mode", ["abort", "in_place"])
    def test_incremental_offsets_across_turns_and_branches(self, mode):
        with _serve_router({"use_rollout_routing_replay": True, "pause_generation_mode": mode}) as env:
            session_id = _create_session(env.url)

            first = _post_chat(env.url, session_id, {"messages": self.MESSAGES})
            assert first.status_code == 200
            turn1_body = env.backend.request_log[-1]
            assert turn1_body["return_routed_experts"] is True
            assert turn1_body["routed_experts_start_len"] == 0
            checkpoint1 = self._accumulated(env.url, session_id)

            assistant = first.json()["choices"][0]["message"]
            second = _post_chat(
                env.url,
                session_id,
                {"messages": [*self.MESSAGES, assistant, {"role": "tool", "content": "ok", "tool_call_id": "t0"}]},
            )
            assert second.status_code == 200
            assert env.backend.request_log[-1]["routed_experts_start_len"] == len(checkpoint1) - 1
            checkpoint2 = self._accumulated(env.url, session_id)
            assert len(checkpoint2) > len(checkpoint1)

            # A divergent replay attaches at the turn-1 node (a sibling branch,
            # never a rollback): the offset returns to that ancestor snapshot.
            branch = _post_chat(
                env.url,
                session_id,
                {
                    "messages": [
                        *self.MESSAGES,
                        assistant,
                        {"role": "tool", "content": "different", "tool_call_id": "t0"},
                    ]
                },
            )
            assert branch.status_code == 200
            branch_body = env.backend.request_log[-1]
            assert branch_body["routed_experts_start_len"] == len(checkpoint1) - 1
            assert len(checkpoint1) - 1 < len(checkpoint2) - 1

    def test_retract_request_has_no_start_len(self):
        with _serve_router({"use_rollout_routing_replay": True, "pause_generation_mode": "retract"}) as env:
            session_id = _create_session(env.url)
            assert _post_chat(env.url, session_id, {"messages": self.MESSAGES}).status_code == 200
            body = env.backend.request_log[-1]
            assert body["return_routed_experts"] is True
            assert "routed_experts_start_len" not in body
