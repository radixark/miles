"""Unit tests for SingleUserTurnTrajectoryManager.

Tests the trajectory manager's session CRUD and pretokenized state management
logic in isolation (no HTTP server, no real tokenizer).
"""

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from miles.rollout.session.session_errors import MessageValidationError, SessionNotFoundError, TokenizationError
from miles.rollout.session.session_types import SessionRecord
from miles.rollout.session.single_user_turn_trajectory import SingleUserTurnTrajectoryManager
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


@pytest.fixture
def manager():
    args = SimpleNamespace()
    mock_tito = _MockTITOTokenizer(tokenizer=None, assistant_start_str="<|im_start|>assistant")
    return SingleUserTurnTrajectoryManager(args, tokenizer=None, tito_tokenizer=mock_tito)


class TestSessionCRUD:
    def test_create_session(self, manager: SingleUserTurnTrajectoryManager):
        session_id = manager.create_session()
        assert session_id is not None
        assert len(session_id) == 32
        assert session_id in manager.sessions

    def test_get_session_records_by_id(self, manager: SingleUserTurnTrajectoryManager):
        session_id = manager.create_session()
        records = manager.get_session_records_by_id(session_id)
        assert records == []

    def test_get_session_records_by_id_not_found(self, manager: SingleUserTurnTrajectoryManager):
        with pytest.raises(SessionNotFoundError):
            manager.get_session_records_by_id("nonexistent")

    def test_delete_session_by_id(self, manager: SingleUserTurnTrajectoryManager):
        session_id = manager.create_session()
        assert manager.delete_session_by_id(session_id) is True
        assert session_id not in manager.sessions
        with pytest.raises(SessionNotFoundError):
            manager.delete_session_by_id(session_id)

    def test_append_session_record(self, manager: SingleUserTurnTrajectoryManager):
        session_id = manager.create_session()
        record = SessionRecord(
            timestamp=0.0,
            method="POST",
            path="/v1/chat/completions",
            status_code=200,
            request={"messages": [{"role": "user", "content": "hello"}]},
            response={"choices": []},
        )

        appended = manager.append_session_record(session_id, record)

        assert appended is True
        records = manager.get_session_records_by_id(session_id)
        assert records is not None
        assert len(records) == 1
        assert records[0].path == record.path

    def test_append_session_record_missing_session(self, manager: SingleUserTurnTrajectoryManager):
        record = SessionRecord(
            timestamp=0.0,
            method="POST",
            path="/v1/chat/completions",
            status_code=200,
            request={},
            response={},
        )
        with pytest.raises(SessionNotFoundError):
            manager.append_session_record("missing", record)


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
    "content": "It's 25°C in Beijing. Let me also check Shanghai.",
    "tool_calls": [
        {"id": "call_2", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Shanghai"}'}}
    ],
}
TOOL_MSG_2 = {"role": "tool", "content": '{"temperature": 30}', "tool_call_id": "call_2"}
ASSISTANT_MSG_FINAL = {"role": "assistant", "content": "Beijing is 25°C and Shanghai is 30°C."}
RETRY_SYS_MSG = {"role": "system", "content": "Please try using the tools to answer."}


class TestSingleUserTurnPretokenized:
    """Test prepare_pretokenized and update_pretokenized_state across turns."""

    def test_first_turn_returns_none(self, manager: SingleUserTurnTrajectoryManager):
        """First turn has no prior token_ids, so try_prepare returns None."""
        sid = manager.create_session()
        messages = [SYS_MSG, USER_MSG]
        result = manager.prepare_pretokenized(sid, messages)
        assert result is None

    def test_two_turn_trajectory(self, manager: SingleUserTurnTrajectoryManager):
        """Full 2-turn: user -> assistant(tool_call) -> tool -> final answer."""
        sid = manager.create_session()

        # --- Turn 1: [sys, user] -> assistant with tool_call ---
        turn1_messages = [SYS_MSG, USER_MSG]
        assert manager.prepare_pretokenized(sid, turn1_messages) is None

        turn1_prompt_ids = [1, 2, 3, 4, 5]
        turn1_completion_ids = [10, 11, 12]
        manager.update_pretokenized_state(sid, turn1_messages, ASSISTANT_MSG_1, turn1_prompt_ids, turn1_completion_ids)

        session = manager.sessions[sid]
        assert session.messages == [SYS_MSG, USER_MSG, ASSISTANT_MSG_1]
        assert session.token_ids == [1, 2, 3, 4, 5, 10, 11, 12]

        # --- Turn 2: [sys, user, assistant, tool] -> final answer ---
        turn2_messages = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1]
        result = manager.prepare_pretokenized(sid, turn2_messages)
        assert result is not None
        assert result["input_ids"] == [1, 2, 3, 4, 5, 10, 11, 12]

        turn2_prompt_ids = [1, 2, 3, 4, 5, 10, 11, 12, 20, 21]
        turn2_completion_ids = [30, 31, 32]
        manager.update_pretokenized_state(
            sid, turn2_messages, ASSISTANT_MSG_FINAL, turn2_prompt_ids, turn2_completion_ids
        )

        assert session.messages == [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1, ASSISTANT_MSG_FINAL]
        assert session.token_ids == [1, 2, 3, 4, 5, 10, 11, 12, 20, 21, 30, 31, 32]

    def test_three_turn_trajectory(self, manager: SingleUserTurnTrajectoryManager):
        """Full 3-turn: user -> ass(tool) -> tool -> ass(tool) -> tool -> final."""
        sid = manager.create_session()

        # Turn 1
        t1_msgs = [SYS_MSG, USER_MSG]
        manager.update_pretokenized_state(sid, t1_msgs, ASSISTANT_MSG_1, [1, 2, 3], [10, 11])

        # Turn 2
        t2_msgs = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1]
        result = manager.prepare_pretokenized(sid, t2_msgs)
        assert result == {"input_ids": [1, 2, 3, 10, 11]}

        manager.update_pretokenized_state(sid, t2_msgs, ASSISTANT_MSG_2, [1, 2, 3, 10, 11, 20, 21], [30, 31])

        # Turn 3
        t3_msgs = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1, ASSISTANT_MSG_2, TOOL_MSG_2]
        result = manager.prepare_pretokenized(sid, t3_msgs)
        assert result == {"input_ids": [1, 2, 3, 10, 11, 20, 21, 30, 31]}

        manager.update_pretokenized_state(
            sid, t3_msgs, ASSISTANT_MSG_FINAL, [1, 2, 3, 10, 11, 20, 21, 30, 31, 40], [50, 51]
        )

        session = manager.sessions[sid]
        assert len(session.messages) == 7  # sys, user, ass1, tool1, ass2, tool2, final
        assert session.token_ids == [1, 2, 3, 10, 11, 20, 21, 30, 31, 40, 50, 51]

    def test_prefix_mismatch_raises(self, manager: SingleUserTurnTrajectoryManager):
        """update_pretokenized_state asserts stored token_ids is prefix of new."""
        sid = manager.create_session()
        manager.update_pretokenized_state(sid, [SYS_MSG, USER_MSG], ASSISTANT_MSG_1, [1, 2, 3], [10, 11])

        with pytest.raises(TokenizationError, match="pretokenized prefix mismatch"):
            manager.update_pretokenized_state(
                sid,
                [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1],
                ASSISTANT_MSG_FINAL,
                [9, 9, 9, 20, 21],  # does NOT start with [1,2,3,10,11]
                [30],
            )

    def test_not_append_only_raises(self, manager: SingleUserTurnTrajectoryManager):
        """try_prepare raises when new messages modify stored prefix."""
        sid = manager.create_session()
        manager.update_pretokenized_state(sid, [SYS_MSG, USER_MSG], ASSISTANT_MSG_1, [1, 2, 3], [10])

        bad_messages = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, {"role": "assistant", "content": "oops"}]
        with pytest.raises(MessageValidationError, match="role=.assistant.*allowed="):
            manager.prepare_pretokenized(sid, bad_messages)

    def test_user_after_assistant_raises(self, manager: SingleUserTurnTrajectoryManager):
        """try_prepare raises when user message appears after assistant."""
        sid = manager.create_session()
        manager.update_pretokenized_state(sid, [SYS_MSG, USER_MSG], ASSISTANT_MSG_1, [1, 2, 3], [10])

        bad_messages = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1, {"role": "user", "content": "second"}]
        with pytest.raises(MessageValidationError, match="user message at index 4.*after the first assistant"):
            manager.prepare_pretokenized(sid, bad_messages)

    def test_session_not_found_raises(self, manager: SingleUserTurnTrajectoryManager):
        with pytest.raises(SessionNotFoundError, match="session not found"):
            manager.prepare_pretokenized("nonexistent", [SYS_MSG, USER_MSG])

    def test_no_system_message(self, manager: SingleUserTurnTrajectoryManager):
        """Works without system message (system is optional)."""
        sid = manager.create_session()
        msgs = [USER_MSG]
        manager.update_pretokenized_state(sid, msgs, ASSISTANT_MSG_1, [1, 2], [10])

        t2_msgs = [USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1]
        result = manager.prepare_pretokenized(sid, t2_msgs)
        assert result == {"input_ids": [1, 2, 10]}

    def test_append_system_message_allowed(self, manager: SingleUserTurnTrajectoryManager):
        """Appending a system message after tool messages is allowed (e.g. retry prompt)."""
        sid = manager.create_session()
        manager.update_pretokenized_state(sid, [SYS_MSG, USER_MSG], ASSISTANT_MSG_1, [1, 2, 3], [10, 11])

        messages = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1, RETRY_SYS_MSG]
        result = manager.prepare_pretokenized(sid, messages)
        assert result is not None
        assert result["input_ids"] == [1, 2, 3, 10, 11]

    def test_append_system_then_assistant_trajectory(self, manager: SingleUserTurnTrajectoryManager):
        """Full trajectory with a retry system message between tool-call turns."""
        sid = manager.create_session()

        # Turn 1: [sys, user] -> assistant(tool_call)
        t1_msgs = [SYS_MSG, USER_MSG]
        manager.update_pretokenized_state(sid, t1_msgs, ASSISTANT_MSG_1, [1, 2, 3], [10, 11])

        # Turn 2: append tool + system_retry -> assistant(tool_call)
        t2_msgs = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1, RETRY_SYS_MSG]
        result = manager.prepare_pretokenized(sid, t2_msgs)
        assert result is not None

        manager.update_pretokenized_state(
            sid,
            t2_msgs,
            ASSISTANT_MSG_2,
            [1, 2, 3, 10, 11, 20, 21, 22],
            [30, 31],
        )

        session = manager.sessions[sid]
        assert session.messages == [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1, RETRY_SYS_MSG, ASSISTANT_MSG_2]

        # Turn 3: append tool after the second assistant
        t3_msgs = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1, RETRY_SYS_MSG, ASSISTANT_MSG_2, TOOL_MSG_2]
        result = manager.prepare_pretokenized(sid, t3_msgs)
        assert result is not None

    def test_multiple_system_messages_at_start(self, manager: SingleUserTurnTrajectoryManager):
        """Multiple system messages before the user message are allowed."""
        sid = manager.create_session()
        extra_sys = {"role": "system", "content": "Extra instructions."}
        msgs = [SYS_MSG, extra_sys, USER_MSG]
        result = manager.prepare_pretokenized(sid, msgs)
        assert result is None  # first turn, no prior tokens

        manager.update_pretokenized_state(sid, msgs, ASSISTANT_MSG_1, [1, 2, 3, 4], [10, 11])
        session = manager.sessions[sid]
        assert session.messages == [SYS_MSG, extra_sys, USER_MSG, ASSISTANT_MSG_1]

        t2_msgs = [SYS_MSG, extra_sys, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1]
        result = manager.prepare_pretokenized(sid, t2_msgs)
        assert result is not None
        assert result["input_ids"] == [1, 2, 3, 4, 10, 11]

    def test_not_append_only_rejects_user_message(self, manager: SingleUserTurnTrajectoryManager):
        """Appending a user message (not tool/system) is rejected."""
        sid = manager.create_session()
        manager.update_pretokenized_state(sid, [SYS_MSG, USER_MSG], ASSISTANT_MSG_1, [1, 2, 3], [10])

        bad = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1, {"role": "user", "content": "extra"}]
        with pytest.raises(MessageValidationError, match="user message at index 4.*after the first assistant"):
            manager.prepare_pretokenized(sid, bad)


class TestRollback:
    """Tests for session rollback to a previous assistant checkpoint."""

    def test_rollback_to_first_assistant(self, manager: SingleUserTurnTrajectoryManager):
        """After 2 completions, rolling back to the first assistant checkpoint works."""
        sid = manager.create_session()

        # Turn 1: [sys, user] -> assistant1
        t1_msgs = [SYS_MSG, USER_MSG]
        manager.update_pretokenized_state(sid, t1_msgs, ASSISTANT_MSG_1, [1, 2, 3], [10, 11])

        # Turn 2: [sys, user, asst1, tool1] -> assistant2
        t2_msgs = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1]
        manager.prepare_pretokenized(sid, t2_msgs)
        manager.update_pretokenized_state(sid, t2_msgs, ASSISTANT_MSG_2, [1, 2, 3, 10, 11, 20, 21], [30, 31])

        session = manager.sessions[sid]
        assert session.num_assistant == 2
        assert len(session.trajectory_token_ids) == 2

        # Rollback: send [sys, user, asst1, NEW_tool] — diverges after asst1
        new_tool = {"role": "tool", "content": '{"temperature": 99}', "tool_call_id": "call_1"}
        rollback_msgs = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, new_tool]
        result = manager.prepare_pretokenized(sid, rollback_msgs)
        assert result is not None

        # State should be rolled back to checkpoint 0
        assert session.num_assistant == 1
        assert len(session.trajectory_token_ids) == 1
        assert session.token_ids == [1, 2, 3, 10, 11]
        assert session.messages == [SYS_MSG, USER_MSG, ASSISTANT_MSG_1]

    def test_rollback_preserves_prefix_tokens(self, manager: SingleUserTurnTrajectoryManager):
        """After rollback, token_ids equals the checkpoint's tokens."""
        sid = manager.create_session()

        t1_msgs = [SYS_MSG, USER_MSG]
        t1_tokens = [1, 2, 3, 10, 11]
        manager.update_pretokenized_state(sid, t1_msgs, ASSISTANT_MSG_1, [1, 2, 3], [10, 11])

        t2_msgs = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1]
        manager.prepare_pretokenized(sid, t2_msgs)
        manager.update_pretokenized_state(sid, t2_msgs, ASSISTANT_MSG_2, [1, 2, 3, 10, 11, 20, 21], [30, 31])

        t3_msgs = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1, ASSISTANT_MSG_2, TOOL_MSG_2]
        manager.prepare_pretokenized(sid, t3_msgs)
        manager.update_pretokenized_state(
            sid, t3_msgs, ASSISTANT_MSG_FINAL, [1, 2, 3, 10, 11, 20, 21, 30, 31, 40], [50, 51]
        )

        assert len(manager.sessions[sid].trajectory_token_ids) == 3

        # Rollback to checkpoint 0 (first assistant)
        new_tool = {"role": "tool", "content": '{"alt": true}', "tool_call_id": "call_1"}
        manager.prepare_pretokenized(sid, [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, new_tool])

        session = manager.sessions[sid]
        assert session.token_ids == t1_tokens
        assert session.num_assistant == 1

    def test_rollback_then_continue_full_trajectory(self, manager: SingleUserTurnTrajectoryManager):
        """Rollback and then complete a full new trajectory from the checkpoint."""
        sid = manager.create_session()

        # Turn 1
        t1_msgs = [SYS_MSG, USER_MSG]
        manager.update_pretokenized_state(sid, t1_msgs, ASSISTANT_MSG_1, [1, 2, 3], [10, 11])

        # Turn 2
        t2_msgs = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1]
        manager.prepare_pretokenized(sid, t2_msgs)
        manager.update_pretokenized_state(sid, t2_msgs, ASSISTANT_MSG_2, [1, 2, 3, 10, 11, 20], [30])

        # Rollback to asst1, send different tool
        new_tool = {"role": "tool", "content": '{"retry": true}', "tool_call_id": "call_1"}
        rollback_msgs = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, new_tool]
        result = manager.prepare_pretokenized(sid, rollback_msgs)
        assert result is not None

        # Continue: complete a new turn from the rolled-back state
        manager.update_pretokenized_state(sid, rollback_msgs, ASSISTANT_MSG_FINAL, [1, 2, 3, 10, 11, 40, 41], [50, 51])

        session = manager.sessions[sid]
        assert session.num_assistant == 2
        assert len(session.trajectory_token_ids) == 2
        assert session.token_ids == [1, 2, 3, 10, 11, 40, 41, 50, 51]
        assert session.messages == [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, new_tool, ASSISTANT_MSG_FINAL]

    def test_rollback_fewer_messages_than_stored(self, manager: SingleUserTurnTrajectoryManager):
        """Rollback triggered when request has strictly fewer messages than stored."""
        sid = manager.create_session()

        # Turn 1: [sys, user] -> asst1
        manager.update_pretokenized_state(sid, [SYS_MSG, USER_MSG], ASSISTANT_MSG_1, [1, 2], [10])

        # Turn 2: [sys, user, asst1, tool1] -> asst2
        t2_msgs = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1]
        manager.prepare_pretokenized(sid, t2_msgs)
        manager.update_pretokenized_state(sid, t2_msgs, ASSISTANT_MSG_2, [1, 2, 10, 20], [30])
        # stored messages: [sys, user, asst1, tool1, asst2] (5 messages)

        # Agent retries with only [sys, user, asst1, sys_retry] (4 messages)
        retry_msgs = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, RETRY_SYS_MSG]
        result = manager.prepare_pretokenized(sid, retry_msgs)
        assert result is not None

        session = manager.sessions[sid]
        assert session.num_assistant == 1
        assert session.messages == [SYS_MSG, USER_MSG, ASSISTANT_MSG_1]

    def test_rollback_to_second_assistant(self, manager: SingleUserTurnTrajectoryManager):
        """Rollback to the second checkpoint (skipping the third)."""
        sid = manager.create_session()

        # 3 completions
        manager.update_pretokenized_state(sid, [SYS_MSG, USER_MSG], ASSISTANT_MSG_1, [1, 2], [10])

        t2 = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1]
        manager.prepare_pretokenized(sid, t2)
        manager.update_pretokenized_state(sid, t2, ASSISTANT_MSG_2, [1, 2, 10, 20], [30])

        t3 = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1, ASSISTANT_MSG_2, TOOL_MSG_2]
        manager.prepare_pretokenized(sid, t3)
        manager.update_pretokenized_state(sid, t3, ASSISTANT_MSG_FINAL, [1, 2, 10, 20, 30, 40], [50])

        session = manager.sessions[sid]
        assert session.num_assistant == 3

        # Rollback: keep up to asst2, diverge at tool2
        new_tool = {"role": "tool", "content": '{"alt": 1}', "tool_call_id": "call_2"}
        rollback_msgs = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1, ASSISTANT_MSG_2, new_tool]
        result = manager.prepare_pretokenized(sid, rollback_msgs)
        assert result is not None

        assert session.num_assistant == 2
        assert len(session.trajectory_token_ids) == 2
        assert session.token_ids == [1, 2, 10, 20, 30]
        assert session.messages == [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1, ASSISTANT_MSG_2]

    def test_no_rollback_when_append_only(self, manager: SingleUserTurnTrajectoryManager):
        """Normal append-only flow does not trigger rollback."""
        sid = manager.create_session()

        manager.update_pretokenized_state(sid, [SYS_MSG, USER_MSG], ASSISTANT_MSG_1, [1, 2], [10])

        # Append tool — not a rollback
        t2_msgs = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1]
        result = manager.prepare_pretokenized(sid, t2_msgs)
        assert result is not None

        session = manager.sessions[sid]
        # State should NOT have been rolled back
        assert session.num_assistant == 1
        assert len(session.trajectory_token_ids) == 1
        assert session.token_ids == [1, 2, 10]

    def test_rollback_no_assistant_in_prefix_raises(self, manager: SingleUserTurnTrajectoryManager):
        """Rollback raises if no assistant message exists in the matched prefix."""
        sid = manager.create_session()
        manager.update_pretokenized_state(sid, [SYS_MSG, USER_MSG], ASSISTANT_MSG_1, [1, 2], [10])

        # Diverge at user message (index 1) — only sys matched, no assistant
        bad_msgs = [SYS_MSG, {"role": "user", "content": "different question"}]
        with pytest.raises(MessageValidationError, match="rollback failed.*no assistant"):
            manager.prepare_pretokenized(sid, bad_msgs)

    def test_rollback_records_truncated(self, manager: SingleUserTurnTrajectoryManager):
        """Records are truncated in sync with trajectory_token_ids on rollback."""
        sid = manager.create_session()

        # Turn 1
        manager.update_pretokenized_state(sid, [SYS_MSG, USER_MSG], ASSISTANT_MSG_1, [1, 2], [10])
        r1 = SessionRecord(
            timestamp=1.0, method="POST", path="/v1/chat/completions", status_code=200, request={}, response={}
        )
        manager.append_session_record(sid, r1)

        # Turn 2
        t2 = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1]
        manager.prepare_pretokenized(sid, t2)
        manager.update_pretokenized_state(sid, t2, ASSISTANT_MSG_2, [1, 2, 10, 20], [30])
        r2 = SessionRecord(
            timestamp=2.0, method="POST", path="/v1/chat/completions", status_code=200, request={}, response={}
        )
        manager.append_session_record(sid, r2)

        session = manager.sessions[sid]
        assert len(session.records) == 2

        # Rollback to checkpoint 0
        new_tool = {"role": "tool", "content": '{"alt": 1}', "tool_call_id": "call_1"}
        manager.prepare_pretokenized(sid, [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, new_tool])

        assert len(session.records) == 1
        assert session.records[0].timestamp == 1.0


class TestUpdatePretokenizedStateMissingSession:
    """update_pretokenized_state raises SessionNotFoundError for unknown session."""

    def test_raises_on_missing_session(self, manager: SingleUserTurnTrajectoryManager):
        with pytest.raises(SessionNotFoundError, match="session not found"):
            manager.update_pretokenized_state(
                "nonexistent",
                [SYS_MSG, USER_MSG],
                ASSISTANT_MSG_1,
                [1, 2, 3],
                [10],
            )


class TestComputeSessionMismatch:
    """Tests for compute_session_mismatch."""

    def test_raises_for_missing_session(self, manager: SingleUserTurnTrajectoryManager):
        with pytest.raises(SessionNotFoundError):
            manager.compute_session_mismatch("nonexistent")

    def test_returns_none_for_empty_token_ids(self, manager: SingleUserTurnTrajectoryManager):
        sid = manager.create_session()
        assert manager.compute_session_mismatch(sid) is None

    @patch("miles.rollout.session.single_user_turn_trajectory.apply_chat_template")
    def test_returns_empty_list_when_no_mismatch(self, mock_template, manager: SingleUserTurnTrajectoryManager):
        sid = manager.create_session()
        manager.update_pretokenized_state(sid, [SYS_MSG, USER_MSG], ASSISTANT_MSG_1, [1, 2, 3], [10, 11])

        # Simulate: template returns same IDs as stored
        mock_template.return_value = [1, 2, 3, 10, 11]

        # Need a real comparator; replace the None one
        mock_comparator = MagicMock()
        mock_comparator.compare_sequences.return_value = []
        manager.comparator = mock_comparator

        result = manager.compute_session_mismatch(sid)
        assert result == []
        mock_comparator.compare_sequences.assert_called_once_with([1, 2, 3, 10, 11], [1, 2, 3, 10, 11])

    @patch("miles.rollout.session.single_user_turn_trajectory.apply_chat_template")
    def test_returns_mismatch_dicts(self, mock_template, manager: SingleUserTurnTrajectoryManager):
        sid = manager.create_session()
        manager.update_pretokenized_state(sid, [SYS_MSG, USER_MSG], ASSISTANT_MSG_1, [1, 2, 3], [10, 11])

        mock_template.return_value = [1, 2, 99, 10, 11]

        @dataclass
        class FakeMismatch:
            position: int

            def to_dict(self):
                return {"position": self.position, "detail": "mismatch"}

        mock_comparator = MagicMock()
        mock_comparator.compare_sequences.return_value = [FakeMismatch(position=2)]
        manager.comparator = mock_comparator

        result = manager.compute_session_mismatch(sid)
        assert result == [{"position": 2, "detail": "mismatch"}]

    @patch("miles.rollout.session.single_user_turn_trajectory.apply_chat_template")
    def test_raises_tokenization_error_on_exception(self, mock_template, manager: SingleUserTurnTrajectoryManager):
        sid = manager.create_session()
        manager.update_pretokenized_state(sid, [SYS_MSG, USER_MSG], ASSISTANT_MSG_1, [1, 2, 3], [10, 11])

        mock_template.side_effect = RuntimeError("tokenizer failed")

        with pytest.raises(TokenizationError, match="tokenizer failed"):
            manager.compute_session_mismatch(sid)

    @patch("miles.rollout.session.single_user_turn_trajectory.apply_chat_template")
    def test_uses_tools_from_last_record(self, mock_template, manager: SingleUserTurnTrajectoryManager):
        sid = manager.create_session()
        manager.update_pretokenized_state(sid, [SYS_MSG, USER_MSG], ASSISTANT_MSG_1, [1, 2], [10])

        tools = [{"type": "function", "function": {"name": "get_weather"}}]
        record = SessionRecord(
            timestamp=1.0,
            method="POST",
            path="/v1/chat/completions",
            status_code=200,
            request={"tools": tools},
            response={},
        )
        manager.append_session_record(sid, record)

        mock_template.return_value = [1, 2, 10]
        mock_comparator = MagicMock()
        mock_comparator.compare_sequences.return_value = []
        manager.comparator = mock_comparator

        manager.compute_session_mismatch(sid)

        # Verify tools were passed to apply_chat_template
        _, kwargs = mock_template.call_args
        assert kwargs["tools"] == tools
