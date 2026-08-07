"""Integration contracts for configurable session message matching."""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from tests.fast.router.test_linear_trajectory import _make_registry

from miles.rollout.session.core import SessionCore
from miles.rollout.session.errors import SessionMessageMatcherError
from miles.rollout.session.linear_trajectory import LinearTrajectory, SessionRegistry
from miles.rollout.session.message_matching import MessageMatchCache
from miles.rollout.session.types import SessionRecord
from miles.rollout.session.v2.core import SessionCoreV2
from miles.rollout.session.v2.session_state import (
    SessionRegistryV2,
    SessionStateV2,
    commit_generation,
    plan_pretokenized_request,
)
from miles.utils.chat_template_utils.message_matcher_hub import role_content_only_message_matches

_USER = {"role": "user", "content": "start"}
_NEXT_USER = {"role": "user", "content": "continue"}


class _Backend:
    def __init__(self) -> None:
        self.requests: list[dict] = []

    async def do_proxy(self, request, path, *, body, headers):
        self.requests.append(json.loads(body))
        sequence = len(self.requests)
        response = {
            "id": f"response-{sequence}",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": f"answer-{sequence}",
                    },
                    "finish_reason": "stop",
                    "meta_info": {
                        "output_token_logprobs": [[-0.1, 100 + sequence]],
                        "completion_tokens": 1,
                    },
                }
            ],
        }
        return {
            "status_code": 200,
            "headers": {},
            "response_body": json.dumps(response).encode(),
        }


def _args() -> SimpleNamespace:
    return SimpleNamespace(
        session_sample_picker_path="miles.rollout.session.v2.picker_hub.drop_retries",
        session_sample_postprocessor_path=("miles.rollout.session.v2.postprocessor_hub.default_postprocess"),
    )


def _build_core(version: str):
    base_registry = _make_registry()
    backend = _Backend()
    args = _args()
    registry_type = SessionRegistryV2 if version == "v2" else SessionRegistry
    registry = registry_type(
        args,
        tokenizer=None,
        tito_tokenizer=base_registry.tito_tokenizer,
        message_matcher=role_content_only_message_matches,
        message_matcher_selector="role_content_only",
    )
    core_type = SessionCoreV2 if version == "v2" else SessionCore
    return core_type(backend, registry, args), backend, registry


def _active_messages(version: str, session) -> list[dict]:
    return session.active_messages() if version == "v2" else session.messages


@pytest.mark.parametrize("version", ["v1", "v2"])
async def test_core_uses_stored_prefix_and_audits_raw_replay(version: str) -> None:
    core, backend, registry = _build_core(version)
    session_id = registry.create_session()

    first = await core.chat_completions(
        session_id,
        method="POST",
        query="",
        headers={},
        body=json.dumps({"messages": [_USER]}).encode(),
    )
    assert first.status_code == 200

    session = registry.get_session(session_id)
    stored_assistant = copy.deepcopy(_active_messages(version, session)[1])
    replayed_assistant = {
        **stored_assistant,
        "reasoning_content": "rewritten by harness",
        "tool_calls": [{"malformed": True}],
    }
    raw_replay = [_USER, replayed_assistant, _NEXT_USER]

    second = await core.chat_completions(
        session_id,
        method="POST",
        query="",
        headers={},
        body=json.dumps({"messages": raw_replay}).encode(),
    )
    assert second.status_code == 200

    effective = backend.requests[-1]["messages"]
    assert effective[1] == stored_assistant
    assert effective[2] == _NEXT_USER
    assert _active_messages(version, session)[1] == stored_assistant

    get_response = await core.get_session(session_id)
    records = json.loads(get_response.body)["records"]
    assert "replayed_messages" not in records[0]
    assert records[1]["request"]["messages"] == effective
    assert records[1]["replayed_messages"] == raw_replay


def _assistant_tool_call(call_id: str) -> dict:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": "lookup", "arguments": "{}"},
            }
        ],
    }


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_role_content_only_does_not_reconcile_cross_boundary_call_ids(
    version: str,
) -> None:
    registry = _make_registry()
    stored_assistant = _assistant_tool_call("A")
    replayed_assistant = _assistant_tool_call("B")
    tool_result = {
        "role": "tool",
        "content": "result",
        "tool_call_id": "B",
    }
    replayed = [_USER, replayed_assistant, tool_result]

    if version == "v1":
        state = LinearTrajectory()
        state.update_pretokenized_state(
            [_USER],
            stored_assistant,
            prompt_token_ids=[0],
            completion_token_ids=[1],
            max_trim_tokens=0,
        )
        prepared = state.plan_pretokenized(
            replayed,
            tito_tokenizer=registry.tito_tokenizer,
            message_matcher=role_content_only_message_matches,
        )
    else:
        state = SessionStateV2()
        record = SessionRecord(
            timestamp=0.0,
            method="POST",
            path="/v1/chat/completions",
            request={"messages": [_USER], "input_ids": [0]},
            response={},
            status_code=200,
        )
        commit_generation(
            state,
            parent=None,
            request_messages=[_USER],
            assistant_message=stored_assistant,
            prompt_token_ids=[0],
            completion_token_ids=[1],
            max_trim_tokens=0,
            record=record,
            response_id="response-A",
            finish_reason="stop",
        )
        prepared = plan_pretokenized_request(
            state,
            replayed,
            tools=None,
            tito_tokenizer=registry.tito_tokenizer,
            message_matcher=role_content_only_message_matches,
        )

    assert prepared.effective_messages[1]["tool_calls"][0]["id"] == "A"
    assert prepared.effective_messages[2]["tool_call_id"] == "B"
    assert prepared.replayed_messages == replayed


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_tito_failure_does_not_apply_proposed_view_change(version: str) -> None:
    registry = _make_registry()
    stored_assistant = {"role": "assistant", "content": "answer"}
    failing_tito = registry.tito_tokenizer

    if version == "v1":
        state = LinearTrajectory()
        state.update_pretokenized_state(
            [_USER],
            stored_assistant,
            prompt_token_ids=[0],
            completion_token_ids=[1],
            max_trim_tokens=0,
        )
        before = copy.deepcopy(
            (
                state.messages,
                state.trajectory_token_ids,
                state.generated_checkpoint_message_ends,
                state.num_assistant,
            )
        )
        failing_tito.apply_chat_template = MagicMock(side_effect=RuntimeError("boom"))
        with pytest.raises(RuntimeError, match="boom"):
            state.plan_pretokenized(
                [{"role": "user", "content": "different"}],
                tito_tokenizer=failing_tito,
            )
        after = (
            state.messages,
            state.trajectory_token_ids,
            state.generated_checkpoint_message_ends,
            state.num_assistant,
        )
    else:
        state = SessionStateV2()
        record = SessionRecord(
            timestamp=0.0,
            method="POST",
            path="/v1/chat/completions",
            request={},
            response={},
            status_code=200,
        )
        original_leaf = commit_generation(
            state,
            parent=None,
            request_messages=[_USER],
            assistant_message=stored_assistant,
            prompt_token_ids=[0],
            completion_token_ids=[1],
            max_trim_tokens=0,
            record=record,
            response_id="response-1",
            finish_reason="stop",
        )
        before = (state.active_leaf, list(state.tree.nodes))
        failing_tito.apply_chat_template = MagicMock(side_effect=RuntimeError("boom"))
        with pytest.raises(RuntimeError, match="boom"):
            plan_pretokenized_request(
                state,
                [{"role": "user", "content": "different"}],
                tools=None,
                tito_tokenizer=failing_tito,
            )
        after = (state.active_leaf, list(state.tree.nodes))
        assert state.active_leaf is original_leaf

    assert after == before


def test_matcher_failure_is_a_checked_server_error() -> None:
    stored = {"role": "user", "content": "same"}
    replayed = dict(stored)

    def non_bool_matcher(stored_message, replayed_message):
        return 1

    with pytest.raises(SessionMessageMatcherError, match="must return bool"):
        MessageMatchCache(non_bool_matcher).matches(stored, replayed)

    def exploding_matcher(stored_message, replayed_message):
        raise RuntimeError("secret message contents")

    with pytest.raises(
        SessionMessageMatcherError,
        match="raised an exception",
    ) as exc_info:
        MessageMatchCache(exploding_matcher).matches(stored, replayed)
    assert "secret message contents" not in str(exc_info.value)
