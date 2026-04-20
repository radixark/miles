"""Unit tests for SessionRegistry and LinearTrajectory.

Tests the session registry CRUD and the trajectory pretokenized state management
logic in isolation (no HTTP server, no real tokenizer).
"""

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from miles.rollout.session.linear_trajectory import SessionRegistry
from miles.rollout.session.session_errors import MessageValidationError, SessionNotFoundError, TokenizationError
from miles.rollout.session.session_types import SessionRecord
from miles.utils.chat_template_utils.tito_tokenizer import TITOTokenizer


class _MockTITOTokenizer(TITOTokenizer):
    """Stub for unit tests: returns pretokenized_token_ids unchanged (no
    incremental tokens) and skips real tokenizer operations.
    """

    def create_comparator(self):
        return None

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
        assert session.records == []

    def test_get_session_not_found(self, registry: SessionRegistry):
        with pytest.raises(SessionNotFoundError):
            registry.get_session("nonexistent")

    def test_remove_session(self, registry: SessionRegistry):
        session_id = registry.create_session()
        registry.remove_session(session_id)  # no raise = success
        assert session_id not in registry.sessions
        with pytest.raises(SessionNotFoundError):
            registry.remove_session(session_id)

    def test_append_record(self, registry: SessionRegistry):
        session_id = registry.create_session()
        record = SessionRecord(
            timestamp=0.0,
            method="POST",
            path="/v1/chat/completions",
            status_code=200,
            request={"messages": [{"role": "user", "content": "hello"}]},
            response={"choices": []},
        )

        session = registry.get_session(session_id)
        session.append_record(record)

        assert len(session.records) == 1
        assert session.records[0].path == record.path

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

    def test_first_turn_returns_none(self, registry: SessionRegistry):
        """First turn has no prior token_ids, so prepare returns None."""
        sid = registry.create_session()
        session = registry.get_session(sid)
        messages = [SYS_MSG, USER_MSG]
        result = session.prepare_pretokenized(messages, tito_tokenizer=registry.tito_tokenizer)
        assert result is None

    def test_two_turn_trajectory(self, registry: SessionRegistry):
        """Full 2-turn: user -> assistant(tool_call) -> tool -> final answer."""
        sid = registry.create_session()
        session = registry.get_session(sid)

        # --- Turn 1: [sys, user] -> assistant with tool_call ---
        turn1_messages = [SYS_MSG, USER_MSG]
        assert session.prepare_pretokenized(turn1_messages, tito_tokenizer=registry.tito_tokenizer) is None

        turn1_prompt_ids = [1, 2, 3, 4, 5]
        turn1_completion_ids = [10, 11, 12]
        session.update_pretokenized_state(
            turn1_messages, ASSISTANT_MSG_1, turn1_prompt_ids, turn1_completion_ids, max_trim_tokens=0
        )

        assert session.messages == [SYS_MSG, USER_MSG, ASSISTANT_MSG_1]
        assert session.token_ids == [1, 2, 3, 4, 5, 10, 11, 12]

        # --- Turn 2: [sys, user, assistant, tool] -> final answer ---
        turn2_messages = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1]
        result = session.prepare_pretokenized(turn2_messages, tito_tokenizer=registry.tito_tokenizer)
        assert result is not None
        assert result["input_ids"] == [1, 2, 3, 4, 5, 10, 11, 12]

        turn2_prompt_ids = [1, 2, 3, 4, 5, 10, 11, 12, 20, 21]
        turn2_completion_ids = [30, 31, 32]
        session.update_pretokenized_state(
            turn2_messages, ASSISTANT_MSG_FINAL, turn2_prompt_ids, turn2_completion_ids, max_trim_tokens=0
        )

        assert session.messages == [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1, ASSISTANT_MSG_FINAL]
        assert session.token_ids == [1, 2, 3, 4, 5, 10, 11, 12, 20, 21, 30, 31, 32]

    def test_three_turn_trajectory(self, registry: SessionRegistry):
        """Full 3-turn: user -> ass(tool) -> tool -> ass(tool) -> tool -> final."""
        sid = registry.create_session()
        session = registry.get_session(sid)

        # Turn 1
        t1_msgs = [SYS_MSG, USER_MSG]
        session.update_pretokenized_state(t1_msgs, ASSISTANT_MSG_1, [1, 2, 3], [10, 11], max_trim_tokens=0)

        # Turn 2
        t2_msgs = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1]
        result = session.prepare_pretokenized(t2_msgs, tito_tokenizer=registry.tito_tokenizer)
        assert result == {"input_ids": [1, 2, 3, 10, 11]}

        session.update_pretokenized_state(
            t2_msgs, ASSISTANT_MSG_2, [1, 2, 3, 10, 11, 20, 21], [30, 31], max_trim_tokens=0
        )

        # Turn 3
        t3_msgs = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1, ASSISTANT_MSG_2, TOOL_MSG_2]
        result = session.prepare_pretokenized(t3_msgs, tito_tokenizer=registry.tito_tokenizer)
        assert result == {"input_ids": [1, 2, 3, 10, 11, 20, 21, 30, 31]}

        session.update_pretokenized_state(
            t3_msgs, ASSISTANT_MSG_FINAL, [1, 2, 3, 10, 11, 20, 21, 30, 31, 40], [50, 51], max_trim_tokens=0
        )

        assert len(session.messages) == 7  # sys, user, ass1, tool1, ass2, tool2, final
        assert session.token_ids == [1, 2, 3, 10, 11, 20, 21, 30, 31, 40, 50, 51]

    def test_prefix_mismatch_raises(self, registry: SessionRegistry):
        """update_pretokenized_state asserts stored token_ids is prefix of new."""
        sid = registry.create_session()
        session = registry.get_session(sid)
        session.update_pretokenized_state([SYS_MSG, USER_MSG], ASSISTANT_MSG_1, [1, 2, 3], [10, 11], max_trim_tokens=0)

        with pytest.raises(TokenizationError, match="pretokenized prefix mismatch"):
            session.update_pretokenized_state(
                [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1],
                ASSISTANT_MSG_FINAL,
                [9, 9, 9, 20, 21],  # does NOT start with [1,2,3,10,11]
                [30],
                max_trim_tokens=0,
            )

    def test_not_append_only_raises(self, registry: SessionRegistry):
        """prepare raises when new messages modify stored prefix."""
        sid = registry.create_session()
        session = registry.get_session(sid)
        session.update_pretokenized_state([SYS_MSG, USER_MSG], ASSISTANT_MSG_1, [1, 2, 3], [10], max_trim_tokens=0)

        bad_messages = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, {"role": "assistant", "content": "oops"}]
        with pytest.raises(MessageValidationError, match="role=.assistant.*allowed="):
            session.prepare_pretokenized(bad_messages, tito_tokenizer=registry.tito_tokenizer)

    def test_session_not_found_raises(self, registry: SessionRegistry):
        with pytest.raises(SessionNotFoundError, match="session not found"):
            registry.get_session("nonexistent")

    def test_no_system_message(self, registry: SessionRegistry):
        """Works without system message (system is optional)."""
        sid = registry.create_session()
        session = registry.get_session(sid)
        msgs = [USER_MSG]
        session.update_pretokenized_state(msgs, ASSISTANT_MSG_1, [1, 2], [10], max_trim_tokens=0)

        t2_msgs = [USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1]
        result = session.prepare_pretokenized(t2_msgs, tito_tokenizer=registry.tito_tokenizer)
        assert result == {"input_ids": [1, 2, 10]}

    def test_multiple_system_messages_at_start(self, registry: SessionRegistry):
        """Multiple system messages before the user message are allowed (part of stored prefix)."""
        sid = registry.create_session()
        session = registry.get_session(sid)
        extra_sys = {"role": "system", "content": "Extra instructions."}
        msgs = [SYS_MSG, extra_sys, USER_MSG]
        result = session.prepare_pretokenized(msgs, tito_tokenizer=registry.tito_tokenizer)
        assert result is None  # first turn, no prior tokens

        session.update_pretokenized_state(msgs, ASSISTANT_MSG_1, [1, 2, 3, 4], [10, 11], max_trim_tokens=0)
        assert session.messages == [SYS_MSG, extra_sys, USER_MSG, ASSISTANT_MSG_1]

        t2_msgs = [SYS_MSG, extra_sys, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1]
        result = session.prepare_pretokenized(t2_msgs, tito_tokenizer=registry.tito_tokenizer)
        assert result is not None
        assert result["input_ids"] == [1, 2, 3, 4, 10, 11]


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
        session.update_pretokenized_state([SYS_MSG, USER_MSG], ASSISTANT_MSG_1, [1, 2, 3], [10], max_trim_tokens=0)

        messages = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1]
        result = session.prepare_pretokenized(messages, tito_tokenizer=registry.tito_tokenizer)
        assert result is not None

    def test_system_append_rejected(self, registry: SessionRegistry):
        sid = registry.create_session()
        session = registry.get_session(sid)
        session.update_pretokenized_state([SYS_MSG, USER_MSG], ASSISTANT_MSG_1, [1, 2, 3], [10, 11], max_trim_tokens=0)

        messages = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1, RETRY_SYS_MSG]
        with pytest.raises(MessageValidationError, match="role='system'.*allowed="):
            session.prepare_pretokenized(messages, tito_tokenizer=registry.tito_tokenizer)

    def test_user_append_rejected(self, registry: SessionRegistry):
        sid = registry.create_session()
        session = registry.get_session(sid)
        session.update_pretokenized_state([SYS_MSG, USER_MSG], ASSISTANT_MSG_1, [1, 2, 3], [10], max_trim_tokens=0)

        messages = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1, {"role": "user", "content": "extra"}]
        with pytest.raises(MessageValidationError, match="role='user'.*allowed="):
            session.prepare_pretokenized(messages, tito_tokenizer=registry.tito_tokenizer)


class TestAppendRoleToolSystem:
    """Config: allowed_append_roles=['tool', 'system']."""

    def test_tool_append_allowed(self, registry_with_system: SessionRegistry):
        sid = registry_with_system.create_session()
        session = registry_with_system.get_session(sid)
        session.update_pretokenized_state([SYS_MSG, USER_MSG], ASSISTANT_MSG_1, [1, 2, 3], [10], max_trim_tokens=0)

        messages = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1]
        result = session.prepare_pretokenized(messages, tito_tokenizer=registry_with_system.tito_tokenizer)
        assert result is not None

    def test_system_append_allowed(self, registry_with_system: SessionRegistry):
        sid = registry_with_system.create_session()
        session = registry_with_system.get_session(sid)
        session.update_pretokenized_state([SYS_MSG, USER_MSG], ASSISTANT_MSG_1, [1, 2, 3], [10, 11], max_trim_tokens=0)

        messages = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1, RETRY_SYS_MSG]
        result = session.prepare_pretokenized(messages, tito_tokenizer=registry_with_system.tito_tokenizer)
        assert result is not None
        assert result["input_ids"] == [1, 2, 3, 10, 11]

    def test_system_then_assistant_trajectory(self, registry_with_system: SessionRegistry):
        """Full trajectory with a retry system message between tool-call turns."""
        sid = registry_with_system.create_session()
        session = registry_with_system.get_session(sid)

        t1_msgs = [SYS_MSG, USER_MSG]
        session.update_pretokenized_state(t1_msgs, ASSISTANT_MSG_1, [1, 2, 3], [10, 11], max_trim_tokens=0)

        t2_msgs = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1, RETRY_SYS_MSG]
        result = session.prepare_pretokenized(t2_msgs, tito_tokenizer=registry_with_system.tito_tokenizer)
        assert result is not None

        session.update_pretokenized_state(
            t2_msgs, ASSISTANT_MSG_2, [1, 2, 3, 10, 11, 20, 21, 22], [30, 31], max_trim_tokens=0
        )
        assert session.messages == [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1, RETRY_SYS_MSG, ASSISTANT_MSG_2]

        t3_msgs = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1, RETRY_SYS_MSG, ASSISTANT_MSG_2, TOOL_MSG_2]
        result = session.prepare_pretokenized(t3_msgs, tito_tokenizer=registry_with_system.tito_tokenizer)
        assert result is not None

    def test_user_append_rejected(self, registry_with_system: SessionRegistry):
        sid = registry_with_system.create_session()
        session = registry_with_system.get_session(sid)
        session.update_pretokenized_state([SYS_MSG, USER_MSG], ASSISTANT_MSG_1, [1, 2, 3], [10], max_trim_tokens=0)

        messages = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1, {"role": "user", "content": "extra"}]
        with pytest.raises(MessageValidationError, match="role='user'.*allowed="):
            session.prepare_pretokenized(messages, tito_tokenizer=registry_with_system.tito_tokenizer)


class TestAppendRoleToolUser:
    """Config: allowed_append_roles=['tool', 'user']; user follow-ups are allowed here."""

    def test_tool_append_allowed(self, registry_with_user: SessionRegistry):
        sid = registry_with_user.create_session()
        session = registry_with_user.get_session(sid)
        session.update_pretokenized_state([SYS_MSG, USER_MSG], ASSISTANT_MSG_1, [1, 2, 3], [10], max_trim_tokens=0)

        messages = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1]
        result = session.prepare_pretokenized(messages, tito_tokenizer=registry_with_user.tito_tokenizer)
        assert result is not None

    def test_user_append_allowed(self, registry_with_user: SessionRegistry):
        sid = registry_with_user.create_session()
        session = registry_with_user.get_session(sid)
        session.update_pretokenized_state([SYS_MSG, USER_MSG], ASSISTANT_MSG_1, [1, 2, 3], [10], max_trim_tokens=0)

        messages = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1, {"role": "user", "content": "follow-up"}]
        result = session.prepare_pretokenized(messages, tito_tokenizer=registry_with_user.tito_tokenizer)
        assert result is not None

    def test_user_then_assistant_trajectory(self, registry_with_user: SessionRegistry):
        """Full trajectory: tool → user follow-up → assistant → tool → final."""
        sid = registry_with_user.create_session()
        session = registry_with_user.get_session(sid)

        # Turn 1: [sys, user] -> assistant(tool_call)
        t1_msgs = [SYS_MSG, USER_MSG]
        session.update_pretokenized_state(t1_msgs, ASSISTANT_MSG_1, [1, 2, 3], [10, 11], max_trim_tokens=0)

        # Turn 2: append tool + user follow-up -> assistant(tool_call)
        follow_up = {"role": "user", "content": "Also check Shanghai."}
        t2_msgs = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1, follow_up]
        result = session.prepare_pretokenized(t2_msgs, tito_tokenizer=registry_with_user.tito_tokenizer)
        assert result is not None

        session.update_pretokenized_state(
            t2_msgs, ASSISTANT_MSG_2, [1, 2, 3, 10, 11, 20, 21, 22], [30, 31], max_trim_tokens=0
        )
        assert session.messages == [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1, follow_up, ASSISTANT_MSG_2]

        # Turn 3: append tool after the second assistant
        t3_msgs = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1, follow_up, ASSISTANT_MSG_2, TOOL_MSG_2]
        result = session.prepare_pretokenized(t3_msgs, tito_tokenizer=registry_with_user.tito_tokenizer)
        assert result is not None

    def test_system_append_rejected(self, registry_with_user: SessionRegistry):
        sid = registry_with_user.create_session()
        session = registry_with_user.get_session(sid)
        session.update_pretokenized_state([SYS_MSG, USER_MSG], ASSISTANT_MSG_1, [1, 2, 3], [10, 11], max_trim_tokens=0)

        messages = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1, RETRY_SYS_MSG]
        with pytest.raises(MessageValidationError, match="role='system'.*allowed="):
            session.prepare_pretokenized(messages, tito_tokenizer=registry_with_user.tito_tokenizer)


class TestRollback:
    """Tests for session rollback to a previous assistant checkpoint."""

    def test_rollback_to_first_assistant(self, registry: SessionRegistry):
        """After 2 completions, rolling back to the first assistant checkpoint works."""
        sid = registry.create_session()
        session = registry.get_session(sid)

        # Turn 1: [sys, user] -> assistant1
        t1_msgs = [SYS_MSG, USER_MSG]
        session.update_pretokenized_state(t1_msgs, ASSISTANT_MSG_1, [1, 2, 3], [10, 11], max_trim_tokens=0)

        # Turn 2: [sys, user, asst1, tool1] -> assistant2
        t2_msgs = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1]
        session.prepare_pretokenized(t2_msgs, tito_tokenizer=registry.tito_tokenizer)
        session.update_pretokenized_state(
            t2_msgs, ASSISTANT_MSG_2, [1, 2, 3, 10, 11, 20, 21], [30, 31], max_trim_tokens=0
        )

        assert session.num_assistant == 2
        assert len(session.trajectory_token_ids) == 2

        # Rollback: send [sys, user, asst1, NEW_tool] - diverges after asst1
        new_tool = {"role": "tool", "content": '{"temperature": 99}', "tool_call_id": "call_1"}
        rollback_msgs = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, new_tool]
        result = session.prepare_pretokenized(rollback_msgs, tito_tokenizer=registry.tito_tokenizer)
        assert result is not None

        # State should be rolled back to checkpoint 0
        assert session.num_assistant == 1
        assert len(session.trajectory_token_ids) == 1
        assert session.token_ids == [1, 2, 3, 10, 11]
        assert session.messages == [SYS_MSG, USER_MSG, ASSISTANT_MSG_1]

    def test_multi_step_rollback_raises(self, registry: SessionRegistry):
        """Rollback that discards >1 assistant raises MessageValidationError and leaves state unchanged."""
        sid = registry.create_session()
        session = registry.get_session(sid)

        session.update_pretokenized_state([SYS_MSG, USER_MSG], ASSISTANT_MSG_1, [1, 2, 3], [10, 11], max_trim_tokens=0)

        t2_msgs = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1]
        session.prepare_pretokenized(t2_msgs, tito_tokenizer=registry.tito_tokenizer)
        session.update_pretokenized_state(
            t2_msgs, ASSISTANT_MSG_2, [1, 2, 3, 10, 11, 20, 21], [30, 31], max_trim_tokens=0
        )

        t3_msgs = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1, ASSISTANT_MSG_2, TOOL_MSG_2]
        session.prepare_pretokenized(t3_msgs, tito_tokenizer=registry.tito_tokenizer)
        session.update_pretokenized_state(
            t3_msgs, ASSISTANT_MSG_FINAL, [1, 2, 3, 10, 11, 20, 21, 30, 31, 40], [50, 51], max_trim_tokens=0
        )

        assert session.num_assistant == 3

        # Snapshot state before attempted rollback
        prev_messages = list(session.messages)
        prev_token_ids = list(session.trajectory_token_ids)
        prev_records = list(session.records)
        prev_num_assistant = session.num_assistant

        # Attempt rollback to checkpoint 0 (discard 2 assistants) — should fail
        new_tool = {"role": "tool", "content": '{"alt": true}', "tool_call_id": "call_1"}
        with pytest.raises(MessageValidationError, match="exceeds max_assistant_rollback_steps"):
            session.prepare_pretokenized(
                [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, new_tool], tito_tokenizer=registry.tito_tokenizer
            )

        # State must be unchanged
        assert session.messages == prev_messages
        assert session.trajectory_token_ids == prev_token_ids
        assert session.records == prev_records
        assert session.num_assistant == prev_num_assistant

    def test_rollback_then_continue_full_trajectory(self, registry: SessionRegistry):
        """Rollback and then complete a full new trajectory from the checkpoint."""
        sid = registry.create_session()
        session = registry.get_session(sid)

        # Turn 1
        t1_msgs = [SYS_MSG, USER_MSG]
        session.update_pretokenized_state(t1_msgs, ASSISTANT_MSG_1, [1, 2, 3], [10, 11], max_trim_tokens=0)

        # Turn 2
        t2_msgs = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1]
        session.prepare_pretokenized(t2_msgs, tito_tokenizer=registry.tito_tokenizer)
        session.update_pretokenized_state(t2_msgs, ASSISTANT_MSG_2, [1, 2, 3, 10, 11, 20], [30], max_trim_tokens=0)

        # Rollback to asst1, send different tool
        new_tool = {"role": "tool", "content": '{"retry": true}', "tool_call_id": "call_1"}
        rollback_msgs = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, new_tool]
        result = session.prepare_pretokenized(rollback_msgs, tito_tokenizer=registry.tito_tokenizer)
        assert result is not None

        # Continue: complete a new turn from the rolled-back state
        session.update_pretokenized_state(
            rollback_msgs, ASSISTANT_MSG_FINAL, [1, 2, 3, 10, 11, 40, 41], [50, 51], max_trim_tokens=0
        )

        assert session.num_assistant == 2
        assert len(session.trajectory_token_ids) == 2
        assert session.token_ids == [1, 2, 3, 10, 11, 40, 41, 50, 51]
        assert session.messages == [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, new_tool, ASSISTANT_MSG_FINAL]

    def test_rollback_fewer_messages_than_stored(self, registry_with_system: SessionRegistry):
        """Rollback triggered when request has strictly fewer messages than stored."""
        sid = registry_with_system.create_session()
        session = registry_with_system.get_session(sid)

        # Turn 1: [sys, user] -> asst1
        session.update_pretokenized_state([SYS_MSG, USER_MSG], ASSISTANT_MSG_1, [1, 2], [10], max_trim_tokens=0)

        # Turn 2: [sys, user, asst1, tool1] -> asst2
        t2_msgs = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1]
        session.prepare_pretokenized(t2_msgs, tito_tokenizer=registry_with_system.tito_tokenizer)
        session.update_pretokenized_state(t2_msgs, ASSISTANT_MSG_2, [1, 2, 10, 20], [30], max_trim_tokens=0)
        # stored messages: [sys, user, asst1, tool1, asst2] (5 messages)

        # Agent retries with only [sys, user, asst1, sys_retry] (4 messages)
        retry_msgs = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, RETRY_SYS_MSG]
        result = session.prepare_pretokenized(retry_msgs, tito_tokenizer=registry_with_system.tito_tokenizer)
        assert result is not None

        assert session.num_assistant == 1
        assert session.messages == [SYS_MSG, USER_MSG, ASSISTANT_MSG_1]

    def test_rollback_to_second_assistant(self, registry: SessionRegistry):
        """Rollback to the second checkpoint (skipping the third)."""
        sid = registry.create_session()
        session = registry.get_session(sid)

        # 3 completions
        session.update_pretokenized_state([SYS_MSG, USER_MSG], ASSISTANT_MSG_1, [1, 2], [10], max_trim_tokens=0)

        t2 = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1]
        session.prepare_pretokenized(t2, tito_tokenizer=registry.tito_tokenizer)
        session.update_pretokenized_state(t2, ASSISTANT_MSG_2, [1, 2, 10, 20], [30], max_trim_tokens=0)

        t3 = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1, ASSISTANT_MSG_2, TOOL_MSG_2]
        session.prepare_pretokenized(t3, tito_tokenizer=registry.tito_tokenizer)
        session.update_pretokenized_state(t3, ASSISTANT_MSG_FINAL, [1, 2, 10, 20, 30, 40], [50], max_trim_tokens=0)

        assert session.num_assistant == 3

        # Rollback: keep up to asst2, diverge at tool2
        new_tool = {"role": "tool", "content": '{"alt": 1}', "tool_call_id": "call_2"}
        rollback_msgs = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1, ASSISTANT_MSG_2, new_tool]
        result = session.prepare_pretokenized(rollback_msgs, tito_tokenizer=registry.tito_tokenizer)
        assert result is not None

        assert session.num_assistant == 2
        assert len(session.trajectory_token_ids) == 2
        assert session.token_ids == [1, 2, 10, 20, 30]
        assert session.messages == [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1, ASSISTANT_MSG_2]

    def test_no_rollback_when_append_only(self, registry: SessionRegistry):
        """Normal append-only flow does not trigger rollback."""
        sid = registry.create_session()
        session = registry.get_session(sid)

        session.update_pretokenized_state([SYS_MSG, USER_MSG], ASSISTANT_MSG_1, [1, 2], [10], max_trim_tokens=0)

        # Append tool - not a rollback
        t2_msgs = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1]
        result = session.prepare_pretokenized(t2_msgs, tito_tokenizer=registry.tito_tokenizer)
        assert result is not None

        # State should NOT have been rolled back
        assert session.num_assistant == 1
        assert len(session.trajectory_token_ids) == 1
        assert session.token_ids == [1, 2, 10]

    def test_rollback_no_assistant_in_prefix_raises(self, registry: SessionRegistry):
        """Rollback raises if no assistant message exists in the matched prefix."""
        sid = registry.create_session()
        session = registry.get_session(sid)
        session.update_pretokenized_state([SYS_MSG, USER_MSG], ASSISTANT_MSG_1, [1, 2], [10], max_trim_tokens=0)

        # Diverge at user message (index 1) - only sys matched, no assistant
        bad_msgs = [SYS_MSG, {"role": "user", "content": "different question"}]
        with pytest.raises(MessageValidationError, match="rollback failed.*no assistant"):
            session.prepare_pretokenized(bad_msgs, tito_tokenizer=registry.tito_tokenizer)

    def test_rollback_records_truncated(self, registry: SessionRegistry):
        """Records are truncated in sync with trajectory_token_ids on rollback."""
        sid = registry.create_session()
        session = registry.get_session(sid)

        # Turn 1
        session.update_pretokenized_state([SYS_MSG, USER_MSG], ASSISTANT_MSG_1, [1, 2], [10], max_trim_tokens=0)
        r1 = SessionRecord(
            timestamp=1.0, method="POST", path="/v1/chat/completions", status_code=200, request={}, response={}
        )
        session.append_record(r1)

        # Turn 2
        t2 = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1]
        session.prepare_pretokenized(t2, tito_tokenizer=registry.tito_tokenizer)
        session.update_pretokenized_state(t2, ASSISTANT_MSG_2, [1, 2, 10, 20], [30], max_trim_tokens=0)
        r2 = SessionRecord(
            timestamp=2.0, method="POST", path="/v1/chat/completions", status_code=200, request={}, response={}
        )
        session.append_record(r2)

        assert len(session.records) == 2

        # Rollback to checkpoint 0
        new_tool = {"role": "tool", "content": '{"alt": 1}', "tool_call_id": "call_1"}
        session.prepare_pretokenized(
            [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, new_tool], tito_tokenizer=registry.tito_tokenizer
        )

        assert len(session.records) == 1
        assert session.records[0].timestamp == 1.0


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

    @patch("miles.rollout.session.linear_trajectory.apply_chat_template")
    def test_returns_empty_list_when_no_mismatch(self, mock_template, registry: SessionRegistry):
        sid = registry.create_session()
        session = registry.get_session(sid)
        session.update_pretokenized_state([SYS_MSG, USER_MSG], ASSISTANT_MSG_1, [1, 2, 3], [10, 11], max_trim_tokens=0)

        # Simulate: template returns same IDs as stored
        mock_template.return_value = [1, 2, 3, 10, 11]

        # Need a real comparator; replace the None one
        mock_comparator = MagicMock()
        mock_comparator.compare_sequences.return_value = []
        registry.comparator = mock_comparator

        result = registry.compute_session_mismatch(session)
        assert result == []
        mock_comparator.compare_sequences.assert_called_once_with([1, 2, 3, 10, 11], [1, 2, 3, 10, 11])

    @patch("miles.rollout.session.linear_trajectory.apply_chat_template")
    def test_returns_mismatch_dicts(self, mock_template, registry: SessionRegistry):
        sid = registry.create_session()
        session = registry.get_session(sid)
        session.update_pretokenized_state([SYS_MSG, USER_MSG], ASSISTANT_MSG_1, [1, 2, 3], [10, 11], max_trim_tokens=0)

        mock_template.return_value = [1, 2, 99, 10, 11]

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

    @patch("miles.rollout.session.linear_trajectory.apply_chat_template")
    def test_raises_tokenization_error_on_exception(self, mock_template, registry: SessionRegistry):
        sid = registry.create_session()
        session = registry.get_session(sid)
        session.update_pretokenized_state([SYS_MSG, USER_MSG], ASSISTANT_MSG_1, [1, 2, 3], [10, 11], max_trim_tokens=0)

        mock_template.side_effect = RuntimeError("tokenizer failed")

        with pytest.raises(TokenizationError, match="tokenizer failed"):
            registry.compute_session_mismatch(session)

    @patch("miles.rollout.session.linear_trajectory.apply_chat_template")
    def test_uses_tools_from_last_record(self, mock_template, registry: SessionRegistry):
        sid = registry.create_session()
        session = registry.get_session(sid)
        session.update_pretokenized_state([SYS_MSG, USER_MSG], ASSISTANT_MSG_1, [1, 2], [10], max_trim_tokens=0)

        tools = [{"type": "function", "function": {"name": "get_weather"}}]
        record = SessionRecord(
            timestamp=1.0,
            method="POST",
            path="/v1/chat/completions",
            status_code=200,
            request={"tools": tools},
            response={},
        )
        session.append_record(record)

        mock_template.return_value = [1, 2, 10]
        mock_comparator = MagicMock()
        mock_comparator.compare_sequences.return_value = []
        registry.comparator = mock_comparator

        registry.compute_session_mismatch(session)

        # Verify tools were passed to apply_chat_template
        _, kwargs = mock_template.call_args
        assert kwargs["tools"] == tools
