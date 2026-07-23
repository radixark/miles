"""Unit tests for SessionRegistry and LinearTrajectory.

Tests the session registry CRUD and the trajectory pretokenized state management
logic in isolation (no HTTP server, no real tokenizer).
"""

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from miles.rollout.session.errors import MessageValidationError, SessionNotFoundError, TokenizationError
from miles.rollout.session.types import SessionRecord
from miles.rollout.session.v2.session_state import (
    SessionRegistry,
    SessionState,
    commit_generation,
    position_for_request,
    prepare_input,
)
from miles.utils.chat_template_utils.tito_tokenizer import TITOTokenizer

_MOCK_FIRST_TURN_TOKENS = [0]


class _MockTITOTokenizer(TITOTokenizer):
    """Stub for unit tests: returns pretokenized_token_ids unchanged (no
    incremental tokens), renders first-turn prompts as a fixed sentinel, and
    skips real tokenizer operations.
    """

    def create_comparator(self):
        return None

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        add_generation_prompt: bool,
        tools: list[dict[str, Any]] | None = None,
        tokenize: bool = False,
    ) -> list[int]:
        return list(_MOCK_FIRST_TURN_TOKENS)

    def tokenize_additional_non_assistant(
        self,
        old_messages: list[dict[str, Any]],
        new_messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> list[int]:
        return []

    def merge_tokens(
        self,
        old_messages: list[dict[str, Any]],
        new_messages: list[dict[str, Any]],
        pretokenized_token_ids: list[int],
        tools: list[dict[str, Any]] | None = None,
    ) -> list[int]:
        return list(pretokenized_token_ids)


def _make_registry(allowed_append_roles: list[str] | None = None) -> SessionRegistry:
    args = SimpleNamespace()
    mock_tito = _MockTITOTokenizer(
        tokenizer=None, assistant_start_str="<|im_start|>assistant", allowed_append_roles=allowed_append_roles
    )
    return SessionRegistry(args, tokenizer=None, tito_tokenizer=mock_tito)


def _commit(
    state: SessionState, request_messages, assistant_message, prompt_ids, completion_ids, *, max_trim_tokens=0
):
    """Drive one committed generation through the serving surface (record stub)."""
    record = SessionRecord(
        timestamp=float(len(state.tree.nodes)),
        method="POST",
        path="/v1/chat/completions",
        status_code=200,
        request={"messages": list(request_messages)},
        response={},
    )
    return commit_generation(
        state,
        parent=state.active_leaf,
        request_messages=request_messages,
        assistant_message=assistant_message,
        prompt_token_ids=prompt_ids,
        completion_token_ids=completion_ids,
        max_trim_tokens=max_trim_tokens,
        record=record,
        response_id=f"resp-{len(state.tree.nodes)}",
        finish_reason="stop",
    )


@pytest.fixture
def registry():
    """Default registry: only tool messages allowed after assistant."""
    return _make_registry()


@pytest.fixture
def registry_with_system():
    """Registry that allows both tool and system in appended messages."""
    return _make_registry(allowed_append_roles=["tool", "system"])


@pytest.fixture
def registry_with_user():
    """Registry that allows tool and user in appended messages."""
    return _make_registry(allowed_append_roles=["tool", "user"])


class TestSessionCRUD:
    def test_create_session(self, registry: SessionRegistry):
        session_id = registry.create_session()
        assert session_id is not None
        assert len(session_id) == 32
        assert session_id in registry.sessions

    def test_get_session(self, registry: SessionRegistry):
        session_id = registry.create_session()
        session = registry.get_session(session_id)
        assert session.active_records() == []

    def test_get_session_not_found(self, registry: SessionRegistry):
        with pytest.raises(SessionNotFoundError):
            registry.get_session("nonexistent")

    def test_remove_session(self, registry: SessionRegistry):
        session_id = registry.create_session()
        registry.remove_session(session_id)  # no raise = success
        assert session_id not in registry.sessions
        with pytest.raises(SessionNotFoundError):
            registry.remove_session(session_id)

    def test_committed_generation_carries_its_record(self, registry: SessionRegistry):
        session_id = registry.create_session()
        session = registry.get_session(session_id)
        node = _commit(session, [{"role": "user", "content": "hello"}], ASSISTANT_MSG_1, [1, 2], [10])

        assert len(session.active_records()) == 1
        assert session.active_records()[0] is node.record
        assert session.active_records()[0].path == "/v1/chat/completions"

    def test_append_record_missing_session(self, registry: SessionRegistry):
        with pytest.raises(SessionNotFoundError):
            registry.get_session("missing")


# ---------------------------------------------------------------------------
# Messages for multi-turn pretokenized tests
# ---------------------------------------------------------------------------

SYS_MSG = {"role": "system", "content": "You are a helpful assistant."}
USER_MSG = {"role": "user", "content": "What's the weather in Beijing?"}
ASSISTANT_MSG_1 = {
    "role": "assistant",
    "content": None,
    "tool_calls": [
        {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Beijing"}'}}
    ],
}
TOOL_MSG_1 = {"role": "tool", "content": '{"temperature": 25}', "tool_call_id": "call_1"}
ASSISTANT_MSG_2 = {
    "role": "assistant",
    "content": "It's 25\u00b0C in Beijing. Let me also check Shanghai.",
    "tool_calls": [
        {"id": "call_2", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Shanghai"}'}}
    ],
}
TOOL_MSG_2 = {"role": "tool", "content": '{"temperature": 30}', "tool_call_id": "call_2"}
ASSISTANT_MSG_FINAL = {"role": "assistant", "content": "Beijing is 25\u00b0C and Shanghai is 30\u00b0C."}
RETRY_SYS_MSG = {"role": "system", "content": "Please try using the tools to answer."}


class TestSingleUserTurnPretokenized:
    """Test prepare_pretokenized and update_pretokenized_state across turns."""

    def test_first_turn_renders_from_scratch(self, registry: SessionRegistry):
        """First turn has no prior token_ids, so prepare renders from scratch."""
        sid = registry.create_session()
        session = registry.get_session(sid)
        messages = [SYS_MSG, USER_MSG]
        result = prepare_input(session, messages, tools=None, tito_tokenizer=registry.tito_tokenizer)
        assert result == _MOCK_FIRST_TURN_TOKENS

    def test_two_turn_trajectory(self, registry: SessionRegistry):
        """Full 2-turn: user -> assistant(tool_call) -> tool -> final answer."""
        sid = registry.create_session()
        session = registry.get_session(sid)

        # --- Turn 1: [sys, user] -> assistant with tool_call ---
        turn1_messages = [SYS_MSG, USER_MSG]
        assert (
            prepare_input(session, turn1_messages, tools=None, tito_tokenizer=registry.tito_tokenizer)
            == _MOCK_FIRST_TURN_TOKENS
        )

        turn1_prompt_ids = [1, 2, 3, 4, 5]
        turn1_completion_ids = [10, 11, 12]
        _commit(session, turn1_messages, ASSISTANT_MSG_1, turn1_prompt_ids, turn1_completion_ids, max_trim_tokens=0)

        assert session.active_messages() == [SYS_MSG, USER_MSG, ASSISTANT_MSG_1]
        assert session.active_token_ids() == [1, 2, 3, 4, 5, 10, 11, 12]

        # --- Turn 2: [sys, user, assistant, tool] -> final answer ---
        turn2_messages = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1]
        result = prepare_input(session, turn2_messages, tools=None, tito_tokenizer=registry.tito_tokenizer)
        assert result == [1, 2, 3, 4, 5, 10, 11, 12]

        turn2_prompt_ids = [1, 2, 3, 4, 5, 10, 11, 12, 20, 21]
        turn2_completion_ids = [30, 31, 32]
        _commit(
            session, turn2_messages, ASSISTANT_MSG_FINAL, turn2_prompt_ids, turn2_completion_ids, max_trim_tokens=0
        )

        assert session.active_messages() == [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1, ASSISTANT_MSG_FINAL]
        assert session.active_token_ids() == [1, 2, 3, 4, 5, 10, 11, 12, 20, 21, 30, 31, 32]

    def test_three_turn_trajectory(self, registry: SessionRegistry):
        """Full 3-turn: user -> ass(tool) -> tool -> ass(tool) -> tool -> final."""
        sid = registry.create_session()
        session = registry.get_session(sid)

        # Turn 1
        t1_msgs = [SYS_MSG, USER_MSG]
        _commit(session, t1_msgs, ASSISTANT_MSG_1, [1, 2, 3], [10, 11], max_trim_tokens=0)

        # Turn 2
        t2_msgs = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1]
        result = prepare_input(session, t2_msgs, tools=None, tito_tokenizer=registry.tito_tokenizer)
        assert result == [1, 2, 3, 10, 11]

        _commit(session, t2_msgs, ASSISTANT_MSG_2, [1, 2, 3, 10, 11, 20, 21], [30, 31], max_trim_tokens=0)

        # Turn 3
        t3_msgs = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1, ASSISTANT_MSG_2, TOOL_MSG_2]
        result = prepare_input(session, t3_msgs, tools=None, tito_tokenizer=registry.tito_tokenizer)
        assert result == [1, 2, 3, 10, 11, 20, 21, 30, 31]

        _commit(
            session, t3_msgs, ASSISTANT_MSG_FINAL, [1, 2, 3, 10, 11, 20, 21, 30, 31, 40], [50, 51], max_trim_tokens=0
        )

        assert len(session.active_messages()) == 7  # sys, user, ass1, tool1, ass2, tool2, final
        assert session.active_token_ids() == [1, 2, 3, 10, 11, 20, 21, 30, 31, 40, 50, 51]

    def test_prefix_mismatch_raises(self, registry: SessionRegistry):
        """update_pretokenized_state asserts stored token_ids is prefix of new."""
        sid = registry.create_session()
        session = registry.get_session(sid)
        _commit(session, [SYS_MSG, USER_MSG], ASSISTANT_MSG_1, [1, 2, 3], [10, 11], max_trim_tokens=0)

        with pytest.raises(TokenizationError, match="pretokenized prefix mismatch"):
            _commit(
                session,
                [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1],
                ASSISTANT_MSG_FINAL,
                [9, 9, 9, 20, 21],  # does NOT start with [1,2,3,10,11]
                [30],
                max_trim_tokens=0,
            )

    def test_disallowed_env_role_in_suffix_raises(self, registry: SessionRegistry):
        """The append-role gate still guards env roles (assistants are exempt:
        they are prompt material for compaction branches)."""
        sid = registry.create_session()
        session = registry.get_session(sid)
        _commit(session, [SYS_MSG, USER_MSG], ASSISTANT_MSG_1, [1, 2, 3], [10], max_trim_tokens=0)

        bad_messages = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, {"role": "user", "content": "not allowed here"}]
        position_for_request(session, bad_messages, strict=False)
        with pytest.raises(MessageValidationError, match="role=.user.*allowed="):
            prepare_input(session, bad_messages, tools=None, tito_tokenizer=registry.tito_tokenizer)

    def test_session_not_found_raises(self, registry: SessionRegistry):
        with pytest.raises(SessionNotFoundError, match="session not found"):
            registry.get_session("nonexistent")

    def test_no_system_message(self, registry: SessionRegistry):
        """Works without system message (system is optional)."""
        sid = registry.create_session()
        session = registry.get_session(sid)
        msgs = [USER_MSG]
        _commit(session, msgs, ASSISTANT_MSG_1, [1, 2], [10], max_trim_tokens=0)

        t2_msgs = [USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1]
        result = prepare_input(session, t2_msgs, tools=None, tito_tokenizer=registry.tito_tokenizer)
        assert result == [1, 2, 10]

    def test_multiple_system_messages_at_start(self, registry: SessionRegistry):
        """Multiple system messages before the user message are allowed (part of stored prefix)."""
        sid = registry.create_session()
        session = registry.get_session(sid)
        extra_sys = {"role": "system", "content": "Extra instructions."}
        msgs = [SYS_MSG, extra_sys, USER_MSG]
        result = prepare_input(session, msgs, tools=None, tito_tokenizer=registry.tito_tokenizer)
        assert result == _MOCK_FIRST_TURN_TOKENS  # first turn, no prior tokens

        _commit(session, msgs, ASSISTANT_MSG_1, [1, 2, 3, 4], [10, 11], max_trim_tokens=0)
        assert session.active_messages() == [SYS_MSG, extra_sys, USER_MSG, ASSISTANT_MSG_1]

        t2_msgs = [SYS_MSG, extra_sys, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1]
        result = prepare_input(session, t2_msgs, tools=None, tito_tokenizer=registry.tito_tokenizer)
        assert result == [1, 2, 3, 4, 10, 11]


# ---------------------------------------------------------------------------
# TestAppendRole* — allowed_append_roles policy tests
#
# Each class tests one configuration: tool-only (default), tool+system,
# tool+user.  Tests verify which appended roles are accepted or rejected
# under each allowed_append_roles setting.
# ---------------------------------------------------------------------------


class TestAppendRoleToolOnly:
    """Default config: allowed_append_roles=['tool']."""

    def test_tool_append_allowed(self, registry: SessionRegistry):
        sid = registry.create_session()
        session = registry.get_session(sid)
        _commit(session, [SYS_MSG, USER_MSG], ASSISTANT_MSG_1, [1, 2, 3], [10], max_trim_tokens=0)

        messages = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1]
        result = prepare_input(session, messages, tools=None, tito_tokenizer=registry.tito_tokenizer)
        assert isinstance(result, list)

    def test_system_append_rejected(self, registry: SessionRegistry):
        sid = registry.create_session()
        session = registry.get_session(sid)
        _commit(session, [SYS_MSG, USER_MSG], ASSISTANT_MSG_1, [1, 2, 3], [10, 11], max_trim_tokens=0)

        messages = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1, RETRY_SYS_MSG]
        with pytest.raises(MessageValidationError, match="role='system'.*allowed="):
            prepare_input(session, messages, tools=None, tito_tokenizer=registry.tito_tokenizer)

    def test_user_append_rejected(self, registry: SessionRegistry):
        sid = registry.create_session()
        session = registry.get_session(sid)
        _commit(session, [SYS_MSG, USER_MSG], ASSISTANT_MSG_1, [1, 2, 3], [10], max_trim_tokens=0)

        messages = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1, {"role": "user", "content": "extra"}]
        with pytest.raises(MessageValidationError, match="role='user'.*allowed="):
            prepare_input(session, messages, tools=None, tito_tokenizer=registry.tito_tokenizer)


class TestAppendRoleToolSystem:
    """Config: allowed_append_roles=['tool', 'system']."""

    def test_tool_append_allowed(self, registry_with_system: SessionRegistry):
        sid = registry_with_system.create_session()
        session = registry_with_system.get_session(sid)
        _commit(session, [SYS_MSG, USER_MSG], ASSISTANT_MSG_1, [1, 2, 3], [10], max_trim_tokens=0)

        messages = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1]
        result = prepare_input(session, messages, tools=None, tito_tokenizer=registry_with_system.tito_tokenizer)
        assert isinstance(result, list)

    def test_system_append_allowed(self, registry_with_system: SessionRegistry):
        sid = registry_with_system.create_session()
        session = registry_with_system.get_session(sid)
        _commit(session, [SYS_MSG, USER_MSG], ASSISTANT_MSG_1, [1, 2, 3], [10, 11], max_trim_tokens=0)

        messages = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1, RETRY_SYS_MSG]
        result = prepare_input(session, messages, tools=None, tito_tokenizer=registry_with_system.tito_tokenizer)
        assert result == [1, 2, 3, 10, 11]

    def test_system_then_assistant_trajectory(self, registry_with_system: SessionRegistry):
        """Full trajectory with a retry system message between tool-call turns."""
        sid = registry_with_system.create_session()
        session = registry_with_system.get_session(sid)

        t1_msgs = [SYS_MSG, USER_MSG]
        _commit(session, t1_msgs, ASSISTANT_MSG_1, [1, 2, 3], [10, 11], max_trim_tokens=0)

        t2_msgs = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1, RETRY_SYS_MSG]
        result = prepare_input(session, t2_msgs, tools=None, tito_tokenizer=registry_with_system.tito_tokenizer)
        assert isinstance(result, list)

        _commit(session, t2_msgs, ASSISTANT_MSG_2, [1, 2, 3, 10, 11, 20, 21, 22], [30, 31], max_trim_tokens=0)
        assert session.active_messages() == [
            SYS_MSG,
            USER_MSG,
            ASSISTANT_MSG_1,
            TOOL_MSG_1,
            RETRY_SYS_MSG,
            ASSISTANT_MSG_2,
        ]

        t3_msgs = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1, RETRY_SYS_MSG, ASSISTANT_MSG_2, TOOL_MSG_2]
        result = prepare_input(session, t3_msgs, tools=None, tito_tokenizer=registry_with_system.tito_tokenizer)
        assert isinstance(result, list)

    def test_user_append_rejected(self, registry_with_system: SessionRegistry):
        sid = registry_with_system.create_session()
        session = registry_with_system.get_session(sid)
        _commit(session, [SYS_MSG, USER_MSG], ASSISTANT_MSG_1, [1, 2, 3], [10], max_trim_tokens=0)

        messages = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1, {"role": "user", "content": "extra"}]
        with pytest.raises(MessageValidationError, match="role='user'.*allowed="):
            prepare_input(session, messages, tools=None, tito_tokenizer=registry_with_system.tito_tokenizer)


class TestAppendRoleToolUser:
    """Config: allowed_append_roles=['tool', 'user']; user follow-ups are allowed here."""

    def test_tool_append_allowed(self, registry_with_user: SessionRegistry):
        sid = registry_with_user.create_session()
        session = registry_with_user.get_session(sid)
        _commit(session, [SYS_MSG, USER_MSG], ASSISTANT_MSG_1, [1, 2, 3], [10], max_trim_tokens=0)

        messages = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1]
        result = prepare_input(session, messages, tools=None, tito_tokenizer=registry_with_user.tito_tokenizer)
        assert isinstance(result, list)

    def test_user_append_allowed(self, registry_with_user: SessionRegistry):
        sid = registry_with_user.create_session()
        session = registry_with_user.get_session(sid)
        _commit(session, [SYS_MSG, USER_MSG], ASSISTANT_MSG_1, [1, 2, 3], [10], max_trim_tokens=0)

        messages = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1, {"role": "user", "content": "follow-up"}]
        result = prepare_input(session, messages, tools=None, tito_tokenizer=registry_with_user.tito_tokenizer)
        assert isinstance(result, list)

    def test_user_then_assistant_trajectory(self, registry_with_user: SessionRegistry):
        """Full trajectory: tool → user follow-up → assistant → tool → final."""
        sid = registry_with_user.create_session()
        session = registry_with_user.get_session(sid)

        # Turn 1: [sys, user] -> assistant(tool_call)
        t1_msgs = [SYS_MSG, USER_MSG]
        _commit(session, t1_msgs, ASSISTANT_MSG_1, [1, 2, 3], [10, 11], max_trim_tokens=0)

        # Turn 2: append tool + user follow-up -> assistant(tool_call)
        follow_up = {"role": "user", "content": "Also check Shanghai."}
        t2_msgs = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1, follow_up]
        result = prepare_input(session, t2_msgs, tools=None, tito_tokenizer=registry_with_user.tito_tokenizer)
        assert isinstance(result, list)

        _commit(session, t2_msgs, ASSISTANT_MSG_2, [1, 2, 3, 10, 11, 20, 21, 22], [30, 31], max_trim_tokens=0)
        assert session.active_messages() == [
            SYS_MSG,
            USER_MSG,
            ASSISTANT_MSG_1,
            TOOL_MSG_1,
            follow_up,
            ASSISTANT_MSG_2,
        ]

        # Turn 3: append tool after the second assistant
        t3_msgs = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1, follow_up, ASSISTANT_MSG_2, TOOL_MSG_2]
        result = prepare_input(session, t3_msgs, tools=None, tito_tokenizer=registry_with_user.tito_tokenizer)
        assert isinstance(result, list)

    def test_system_append_rejected(self, registry_with_user: SessionRegistry):
        sid = registry_with_user.create_session()
        session = registry_with_user.get_session(sid)
        _commit(session, [SYS_MSG, USER_MSG], ASSISTANT_MSG_1, [1, 2, 3], [10, 11], max_trim_tokens=0)

        messages = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1, RETRY_SYS_MSG]
        with pytest.raises(MessageValidationError, match="role='system'.*allowed="):
            prepare_input(session, messages, tools=None, tito_tokenizer=registry_with_user.tito_tokenizer)


class TestRollback:
    """Retry handling through the single-chain view over the tree.

    ``judge_and_position`` never destroys anything: a legal one-step retry
    moves ``active_leaf`` back to the anchor node (abandoned generations stay
    in the tree, invisible to the view). Byte-level HTTP fidelity is pinned
    by ``TestRollbackPins`` in ``test_sessions.py``; this suite covers the
    mechanism."""

    def _dispatch_and_apply(self, state, messages):
        position_for_request(state, messages, strict=False)

    def test_rollback_to_first_assistant(self, registry: SessionRegistry):
        """After 2 completions, a divergent retry rolls back to the first checkpoint."""
        sid = registry.create_session()
        state = registry.get_session(sid)
        session = state

        # Turn 1: [sys, user] -> assistant1
        t1_msgs = [SYS_MSG, USER_MSG]
        _commit(session, t1_msgs, ASSISTANT_MSG_1, [1, 2, 3], [10, 11], max_trim_tokens=0)

        # Turn 2: [sys, user, asst1, tool1] -> assistant2
        t2_msgs = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1]
        prepare_input(session, t2_msgs, tools=None, tito_tokenizer=registry.tito_tokenizer)
        _commit(session, t2_msgs, ASSISTANT_MSG_2, [1, 2, 3, 10, 11, 20, 21], [30, 31], max_trim_tokens=0)

        assert len(session.active_path()) == 2
        assert len([n.token_ids for n in session.active_path()]) == 2

        # Retry: send [sys, user, asst1, NEW_tool] - diverges after asst1
        new_tool = {"role": "tool", "content": '{"temperature": 99}', "tool_call_id": "call_1"}
        rollback_msgs = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, new_tool]
        self._dispatch_and_apply(state, rollback_msgs)
        assert state.active_leaf is state.tree.nodes[0]
        result = prepare_input(session, rollback_msgs, tools=None, tito_tokenizer=registry.tito_tokenizer)
        assert isinstance(result, list)

        # View rolled back to the first generation; the abandoned node stays in the tree
        assert len(state.tree.nodes) == 2
        assert len(session.active_path()) == 1
        assert len([n.token_ids for n in session.active_path()]) == 1
        assert session.active_token_ids() == [1, 2, 3, 10, 11]
        assert session.active_messages() == [SYS_MSG, USER_MSG, ASSISTANT_MSG_1]

    def test_multi_step_rollback_raises(self, registry: SessionRegistry):
        """Rollback that discards >1 assistant raises MessageValidationError and leaves state unchanged."""
        sid = registry.create_session()
        state = registry.get_session(sid)
        session = state

        _commit(session, [SYS_MSG, USER_MSG], ASSISTANT_MSG_1, [1, 2, 3], [10, 11], max_trim_tokens=0)

        t2_msgs = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1]
        prepare_input(session, t2_msgs, tools=None, tito_tokenizer=registry.tito_tokenizer)
        _commit(session, t2_msgs, ASSISTANT_MSG_2, [1, 2, 3, 10, 11, 20, 21], [30, 31], max_trim_tokens=0)

        t3_msgs = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1, ASSISTANT_MSG_2, TOOL_MSG_2]
        prepare_input(session, t3_msgs, tools=None, tito_tokenizer=registry.tito_tokenizer)
        _commit(
            session, t3_msgs, ASSISTANT_MSG_FINAL, [1, 2, 3, 10, 11, 20, 21, 30, 31, 40], [50, 51], max_trim_tokens=0
        )

        assert len(session.active_path()) == 3

        # Snapshot state before attempted rollback
        prev_messages = list(session.active_messages())
        prev_token_ids = list([n.token_ids for n in session.active_path()])
        prev_records = list(session.active_records())
        prev_num_assistant = len(session.active_path())

        # Deep divergence: the view positions at the deep anchor (node 0) and
        # NOTHING is destroyed — the abandoned generations stay in the tree.
        new_tool = {"role": "tool", "content": '{"alt": true}', "tool_call_id": "call_1"}
        position_for_request(state, [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, new_tool], strict=False)
        assert state.active_leaf is state.tree.nodes[0]
        assert len(state.tree.nodes) == prev_num_assistant  # nothing destroyed
        assert [n.token_ids for n in state.tree.nodes] == prev_token_ids
        assert [n.record for n in state.tree.nodes] == prev_records
        assert state.tree.nodes[2].path_messages() == prev_messages

    def test_rollback_then_continue_full_trajectory(self, registry: SessionRegistry):
        """Rollback and then complete a full new trajectory from the checkpoint."""
        sid = registry.create_session()
        state = registry.get_session(sid)
        session = state

        # Turn 1
        t1_msgs = [SYS_MSG, USER_MSG]
        _commit(session, t1_msgs, ASSISTANT_MSG_1, [1, 2, 3], [10, 11], max_trim_tokens=0)

        # Turn 2
        t2_msgs = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1]
        prepare_input(session, t2_msgs, tools=None, tito_tokenizer=registry.tito_tokenizer)
        _commit(session, t2_msgs, ASSISTANT_MSG_2, [1, 2, 3, 10, 11, 20], [30], max_trim_tokens=0)

        # Rollback to asst1, send different tool
        new_tool = {"role": "tool", "content": '{"retry": true}', "tool_call_id": "call_1"}
        rollback_msgs = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, new_tool]
        self._dispatch_and_apply(state, rollback_msgs)
        result = prepare_input(session, rollback_msgs, tools=None, tito_tokenizer=registry.tito_tokenizer)
        assert isinstance(result, list)

        # Continue: complete a new turn from the rolled-back state
        _commit(session, rollback_msgs, ASSISTANT_MSG_FINAL, [1, 2, 3, 10, 11, 40, 41], [50, 51], max_trim_tokens=0)

        assert len(session.active_path()) == 2
        assert len([n.token_ids for n in session.active_path()]) == 2
        assert session.active_token_ids() == [1, 2, 3, 10, 11, 40, 41, 50, 51]
        assert session.active_messages() == [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, new_tool, ASSISTANT_MSG_FINAL]

    def test_rollback_fewer_messages_than_stored(self, registry_with_system: SessionRegistry):
        """Rollback triggered when request has strictly fewer messages than stored."""
        sid = registry_with_system.create_session()
        state = registry_with_system.get_session(sid)
        session = state

        # Turn 1: [sys, user] -> asst1
        _commit(session, [SYS_MSG, USER_MSG], ASSISTANT_MSG_1, [1, 2], [10], max_trim_tokens=0)

        # Turn 2: [sys, user, asst1, tool1] -> asst2
        t2_msgs = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1]
        prepare_input(session, t2_msgs, tools=None, tito_tokenizer=registry_with_system.tito_tokenizer)
        _commit(session, t2_msgs, ASSISTANT_MSG_2, [1, 2, 10, 20], [30], max_trim_tokens=0)
        # stored messages: [sys, user, asst1, tool1, asst2] (5 messages)

        # Agent retries with only [sys, user, asst1, sys_retry] (4 messages)
        retry_msgs = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, RETRY_SYS_MSG]
        self._dispatch_and_apply(state, retry_msgs)
        result = prepare_input(session, retry_msgs, tools=None, tito_tokenizer=registry_with_system.tito_tokenizer)
        assert isinstance(result, list)

        assert len(session.active_path()) == 1
        assert session.active_messages() == [SYS_MSG, USER_MSG, ASSISTANT_MSG_1]

    def test_rollback_to_second_assistant(self, registry: SessionRegistry):
        """Rollback to the second checkpoint (skipping the third)."""
        sid = registry.create_session()
        state = registry.get_session(sid)
        session = state

        # 3 completions
        _commit(session, [SYS_MSG, USER_MSG], ASSISTANT_MSG_1, [1, 2], [10], max_trim_tokens=0)

        t2 = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1]
        prepare_input(session, t2, tools=None, tito_tokenizer=registry.tito_tokenizer)
        _commit(session, t2, ASSISTANT_MSG_2, [1, 2, 10, 20], [30], max_trim_tokens=0)

        t3 = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1, ASSISTANT_MSG_2, TOOL_MSG_2]
        prepare_input(session, t3, tools=None, tito_tokenizer=registry.tito_tokenizer)
        _commit(session, t3, ASSISTANT_MSG_FINAL, [1, 2, 10, 20, 30, 40], [50], max_trim_tokens=0)

        assert len(session.active_path()) == 3

        # Rollback: keep up to asst2, diverge at tool2
        new_tool = {"role": "tool", "content": '{"alt": 1}', "tool_call_id": "call_2"}
        rollback_msgs = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1, ASSISTANT_MSG_2, new_tool]
        self._dispatch_and_apply(state, rollback_msgs)
        assert state.active_leaf is state.tree.nodes[1]
        result = prepare_input(session, rollback_msgs, tools=None, tito_tokenizer=registry.tito_tokenizer)
        assert isinstance(result, list)

        assert len(session.active_path()) == 2
        assert len([n.token_ids for n in session.active_path()]) == 2
        assert session.active_token_ids() == [1, 2, 10, 20, 30]
        assert session.active_messages() == [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1, ASSISTANT_MSG_2]

    def test_no_rollback_when_append_only(self, registry: SessionRegistry):
        """Normal append-only flow classifies as EXTEND and mutates nothing."""
        sid = registry.create_session()
        state = registry.get_session(sid)
        session = state

        _commit(session, [SYS_MSG, USER_MSG], ASSISTANT_MSG_1, [1, 2], [10], max_trim_tokens=0)

        # Append tool - not a rollback
        t2_msgs = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1]
        position_for_request(state, t2_msgs, strict=False)
        assert state.active_leaf is state.tree.nodes[0]
        result = prepare_input(session, t2_msgs, tools=None, tito_tokenizer=registry.tito_tokenizer)
        assert isinstance(result, list)

        # State should NOT have been rolled back
        assert len(session.active_path()) == 1
        assert len([n.token_ids for n in session.active_path()]) == 1
        assert session.active_token_ids() == [1, 2, 10]

    def test_rollback_no_assistant_in_prefix_raises(self, registry: SessionRegistry):
        """Dispatch rejects when no assistant anchor exists in the matched prefix."""
        sid = registry.create_session()
        state = registry.get_session(sid)
        session = state
        _commit(session, [SYS_MSG, USER_MSG], ASSISTANT_MSG_1, [1, 2], [10], max_trim_tokens=0)

        # Diverge inside the root delta: nothing fully matches, so the request
        # opens a new root (the old chain stays intact in the tree).
        bad_msgs = [SYS_MSG, {"role": "user", "content": "different question"}]
        position_for_request(state, bad_msgs, strict=False)
        assert state.active_leaf is None
        assert len(state.tree.nodes) == 1

    def test_rollback_shrinks_record_view_in_sync(self, registry: SessionRegistry):
        """The record view shrinks in sync with the token view on rollback."""
        sid = registry.create_session()
        state = registry.get_session(sid)
        session = state

        _commit(session, [SYS_MSG, USER_MSG], ASSISTANT_MSG_1, [1, 2], [10], max_trim_tokens=0)
        t2 = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1]
        prepare_input(session, t2, tools=None, tito_tokenizer=registry.tito_tokenizer)
        _commit(session, t2, ASSISTANT_MSG_2, [1, 2, 10, 20], [30], max_trim_tokens=0)

        assert len(session.active_records()) == 2

        new_tool = {"role": "tool", "content": '{"alt": 1}', "tool_call_id": "call_1"}
        self._dispatch_and_apply(state, [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, new_tool])

        assert len(session.active_records()) == 1
        assert session.active_records()[0] is state.tree.nodes[0].record


class TestJudgmentCountingMatrix:
    """Counting matrix for the single-chain judgment: the step unit is the
    GENERATION (tree node), not the message — client-carried foreign
    assistants live inside a node's delta and are never anchors (the shape
    the prompt-assistant fix pinned)."""

    FS_USER_2 = {"role": "user", "content": "the real question"}

    def _fresh(self, registry):
        sid = registry.create_session()
        state = registry.get_session(sid)
        return state, state

    def _two_turns(self, registry):
        """stored: [sys, user, asst1, tool1, asst2]; 2 generations."""
        state, session = self._fresh(registry)
        _commit(session, [SYS_MSG, USER_MSG], ASSISTANT_MSG_1, [1, 2], [10], max_trim_tokens=0)
        t2 = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1]
        prepare_input(session, t2, tools=None, tito_tokenizer=registry.tito_tokenizer)
        _commit(session, t2, ASSISTANT_MSG_2, [1, 2, 10, 20], [30], max_trim_tokens=0)
        return state, session

    def test_empty_view_extends_anything(self, registry: SessionRegistry):
        state, _ = self._fresh(registry)
        position_for_request(state, [USER_MSG], strict=False)
        assert state.active_leaf is None

    def test_strict_extension_leaves_view_alone(self, registry: SessionRegistry):
        state, _ = self._two_turns(registry)
        request = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1, ASSISTANT_MSG_2, TOOL_MSG_2]
        position_for_request(state, request, strict=False)
        assert state.active_leaf is state.tree.nodes[1]

    def test_degenerate_equal_history_is_extend(self, registry: SessionRegistry):
        state, _ = self._two_turns(registry)
        request = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1, ASSISTANT_MSG_2]
        position_for_request(state, request, strict=False)
        assert state.active_leaf is state.tree.nodes[1]

    def test_pure_drop_one_generation(self, registry: SessionRegistry):
        """Dropping asst2 (and nothing else) is one step even though the
        request also re-sends tool1: 2 messages beyond the anchor, 1 generation."""
        state, _ = self._two_turns(registry)
        position_for_request(state, [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1], strict=False)
        assert state.active_leaf is state.tree.nodes[0]

    def test_divergent_one_generation(self, registry: SessionRegistry):
        state, _ = self._two_turns(registry)
        new_tool = {"role": "tool", "content": '{"alt": 1}', "tool_call_id": "call_1"}
        position_for_request(state, [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, new_tool], strict=False)
        assert state.active_leaf is state.tree.nodes[0]
        assert state.active_messages() == [SYS_MSG, USER_MSG, ASSISTANT_MSG_1]

    def test_deep_divergence_two_generations_rejected(self, registry: SessionRegistry):
        state, session = self._two_turns(registry)
        t3 = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1, ASSISTANT_MSG_2, TOOL_MSG_2]
        prepare_input(session, t3, tools=None, tito_tokenizer=registry.tito_tokenizer)
        _commit(session, t3, ASSISTANT_MSG_FINAL, [1, 2, 10, 20, 30, 40], [50], max_trim_tokens=0)

        new_tool = {"role": "tool", "content": '{"alt": 1}', "tool_call_id": "call_1"}
        position_for_request(state, [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, new_tool], strict=False)
        assert state.active_leaf is state.tree.nodes[0]  # deep anchor, no destruction

    def test_no_anchor(self, registry: SessionRegistry):
        state, session = self._fresh(registry)
        _commit(session, [SYS_MSG, USER_MSG], ASSISTANT_MSG_1, [1, 2], [10], max_trim_tokens=0)
        position_for_request(state, [SYS_MSG, {"role": "user", "content": "different"}], strict=False)
        assert state.active_leaf is None  # new root; old chain intact
        assert len(state.tree.nodes) == 1

    def test_prompt_assistant_is_not_an_anchor(self, registry: SessionRegistry):
        """Few-shot first request: the anchor is the generation boundary, and
        a divergent retry drops exactly one generation (historically this
        computed discard_count=0 and kept a stale checkpoint)."""
        state, session = self._fresh(registry)
        few_shot = [USER_MSG, ASSISTANT_MSG_1, self.FS_USER_2]
        _commit(session, few_shot, ASSISTANT_MSG_2, [1, 2], [10], max_trim_tokens=0)
        t2 = [*few_shot, ASSISTANT_MSG_2, TOOL_MSG_2]
        prepare_input(session, t2, tools=None, tito_tokenizer=registry.tito_tokenizer)
        _commit(session, t2, ASSISTANT_MSG_FINAL, [1, 2, 10, 20], [30], max_trim_tokens=0)
        # stored: [user, fs_asst, user2, asst2, tool2, final]; generations: asst2, final

        new_tool = {"role": "tool", "content": '{"alt": 1}', "tool_call_id": "call_2"}
        position_for_request(state, [*few_shot, ASSISTANT_MSG_2, new_tool], strict=False)
        assert state.active_leaf is state.tree.nodes[0]
        assert state.active_messages() == [*few_shot, ASSISTANT_MSG_2]

    def test_prefix_with_only_prompt_assistants_has_no_anchor(self, registry: SessionRegistry):
        state, session = self._fresh(registry)
        few_shot = [USER_MSG, ASSISTANT_MSG_1, self.FS_USER_2]
        _commit(session, few_shot, ASSISTANT_MSG_2, [1, 2], [10], max_trim_tokens=0)

        position_for_request(state, [USER_MSG, ASSISTANT_MSG_1, {"role": "user", "content": "changed"}], strict=False)
        assert state.active_leaf is None  # divergence inside the root delta opens a new root


class TestUpdatePretokenizedStateMissingSession:
    """update_pretokenized_state raises SessionNotFoundError for unknown session."""

    def test_raises_on_missing_session(self, registry: SessionRegistry):
        with pytest.raises(SessionNotFoundError, match="session not found"):
            registry.get_session("nonexistent")


class TestComputeSessionMismatch:
    """Tests for compute_session_mismatch."""

    def test_raises_for_missing_session(self, registry: SessionRegistry):
        with pytest.raises(SessionNotFoundError):
            registry.get_session("nonexistent")

    def test_returns_none_for_empty_token_ids(self, registry: SessionRegistry):
        sid = registry.create_session()
        session = registry.get_session(sid)
        assert registry.compute_session_mismatch(session) is None

    def test_returns_empty_list_when_no_mismatch(self, registry: SessionRegistry):
        sid = registry.create_session()
        session = registry.get_session(sid)
        _commit(session, [SYS_MSG, USER_MSG], ASSISTANT_MSG_1, [1, 2, 3], [10, 11], max_trim_tokens=0)

        # Simulate: template returns same IDs as stored
        registry.tito_tokenizer.apply_chat_template = MagicMock(return_value=[1, 2, 3, 10, 11])

        # Need a real comparator; replace the None one
        mock_comparator = MagicMock()
        mock_comparator.compare_sequences.return_value = []
        registry.comparator = mock_comparator

        result = registry.compute_session_mismatch(session)
        assert result == []
        mock_comparator.compare_sequences.assert_called_once_with([1, 2, 3, 10, 11], [1, 2, 3, 10, 11])
        registry.tito_tokenizer.apply_chat_template.assert_called_once_with(
            session.active_messages(),
            tools=None,
            add_generation_prompt=False,
            tokenize=True,
        )

    def test_returns_mismatch_dicts(self, registry: SessionRegistry):
        sid = registry.create_session()
        session = registry.get_session(sid)
        _commit(session, [SYS_MSG, USER_MSG], ASSISTANT_MSG_1, [1, 2, 3], [10, 11], max_trim_tokens=0)

        registry.tito_tokenizer.apply_chat_template = MagicMock(return_value=[1, 2, 99, 10, 11])

        @dataclass
        class FakeMismatch:
            position: int

            def to_dict(self):
                return {"position": self.position, "detail": "mismatch"}

        mock_comparator = MagicMock()
        mock_comparator.compare_sequences.return_value = [FakeMismatch(position=2)]
        registry.comparator = mock_comparator

        result = registry.compute_session_mismatch(session)
        assert result == [{"position": 2, "detail": "mismatch"}]

    def test_raises_tokenization_error_on_exception(self, registry: SessionRegistry):
        sid = registry.create_session()
        session = registry.get_session(sid)
        _commit(session, [SYS_MSG, USER_MSG], ASSISTANT_MSG_1, [1, 2, 3], [10, 11], max_trim_tokens=0)

        registry.tito_tokenizer.apply_chat_template = MagicMock(side_effect=RuntimeError("tokenizer failed"))

        with pytest.raises(TokenizationError, match="tokenizer failed"):
            registry.compute_session_mismatch(session)

    def test_uses_tools_from_last_record(self, registry: SessionRegistry):
        sid = registry.create_session()
        session = registry.get_session(sid)
        _commit(session, [SYS_MSG, USER_MSG], ASSISTANT_MSG_1, [1, 2], [10], max_trim_tokens=0)

        tools = [{"type": "function", "function": {"name": "get_weather"}}]
        record = SessionRecord(
            timestamp=1.0,
            method="POST",
            path="/v1/chat/completions",
            status_code=200,
            request={"tools": tools},
            response={},
        )
        t2 = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1]
        commit_generation(
            session,
            parent=session.active_leaf,
            request_messages=t2,
            assistant_message=ASSISTANT_MSG_2,
            prompt_token_ids=[1, 2, 10, 20],
            completion_token_ids=[30],
            max_trim_tokens=0,
            record=record,
            response_id="resp-tools",
            finish_reason="stop",
        )

        mock_tokenize = MagicMock(return_value=[1, 2, 10])
        registry.tito_tokenizer.apply_chat_template = mock_tokenize
        mock_comparator = MagicMock()
        mock_comparator.compare_sequences.return_value = []
        registry.comparator = mock_comparator

        registry.compute_session_mismatch(session)

        # Verify tools were passed to the TITO renderer.
        _, kwargs = mock_tokenize.call_args
        assert kwargs["tools"] == tools
        assert kwargs["add_generation_prompt"] is False
