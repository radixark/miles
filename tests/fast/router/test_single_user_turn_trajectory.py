"""Unit tests for SingleUserTurnTrajectoryManager.

Tests the trajectory manager's session CRUD and pretokenized state management
logic in isolation (no HTTP server, no real tokenizer).
"""

from types import SimpleNamespace

import pytest

from miles.router.session.session_types import SessionRecord
from miles.router.session.single_user_turn_trajectory import SingleUserTurnTrajectoryManager


@pytest.fixture
def manager():
    args = SimpleNamespace()
    return SingleUserTurnTrajectoryManager(args, tokenizer=None)


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
        records = manager.get_session_records_by_id("nonexistent")
        assert records is None

    def test_delete_session_by_id(self, manager: SingleUserTurnTrajectoryManager):
        session_id = manager.create_session()
        assert manager.delete_session_by_id(session_id) is True
        assert session_id not in manager.sessions
        assert manager.delete_session_by_id(session_id) is None

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
        appended = manager.append_session_record("missing", record)
        assert appended is None


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
    """Test try_prepare_pretokenized and update_pretokenized_state across turns."""

    def test_first_turn_returns_none(self, manager: SingleUserTurnTrajectoryManager):
        """First turn has no prior token_ids, so try_prepare returns None."""
        sid = manager.create_session()
        messages = [SYS_MSG, USER_MSG]
        result = manager.try_prepare_pretokenized(sid, messages)
        assert result is None

    def test_two_turn_trajectory(self, manager: SingleUserTurnTrajectoryManager):
        """Full 2-turn: user -> assistant(tool_call) -> tool -> final answer."""
        sid = manager.create_session()

        # --- Turn 1: [sys, user] -> assistant with tool_call ---
        turn1_messages = [SYS_MSG, USER_MSG]
        assert manager.try_prepare_pretokenized(sid, turn1_messages) is None

        turn1_prompt_ids = [1, 2, 3, 4, 5]
        turn1_completion_ids = [10, 11, 12]
        manager.update_pretokenized_state(sid, turn1_messages, ASSISTANT_MSG_1, turn1_prompt_ids, turn1_completion_ids)

        session = manager.sessions[sid]
        assert session.messages == [SYS_MSG, USER_MSG, ASSISTANT_MSG_1]
        assert session.token_ids == [1, 2, 3, 4, 5, 10, 11, 12]

        # --- Turn 2: [sys, user, assistant, tool] -> final answer ---
        turn2_messages = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1]
        result = manager.try_prepare_pretokenized(sid, turn2_messages)
        assert result is not None
        assert result["pretokenized_token_ids"] == [1, 2, 3, 4, 5, 10, 11, 12]
        assert result["pretokenized_num_message"] == 3  # [sys, user, assistant]

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
        result = manager.try_prepare_pretokenized(sid, t2_msgs)
        assert result == {"pretokenized_token_ids": [1, 2, 3, 10, 11], "pretokenized_num_message": 3}

        manager.update_pretokenized_state(sid, t2_msgs, ASSISTANT_MSG_2, [1, 2, 3, 10, 11, 20, 21], [30, 31])

        # Turn 3
        t3_msgs = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1, ASSISTANT_MSG_2, TOOL_MSG_2]
        result = manager.try_prepare_pretokenized(sid, t3_msgs)
        assert result == {
            "pretokenized_token_ids": [1, 2, 3, 10, 11, 20, 21, 30, 31],
            "pretokenized_num_message": 5,  # [sys, user, ass1, tool1, ass2]
        }

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

        with pytest.raises(AssertionError, match="pretokenized prefix mismatch"):
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
        with pytest.raises(ValueError, match="not append-only"):
            manager.try_prepare_pretokenized(sid, bad_messages)

    def test_multiple_user_messages_raises(self, manager: SingleUserTurnTrajectoryManager):
        """try_prepare raises when messages contain multiple user messages."""
        sid = manager.create_session()
        manager.update_pretokenized_state(sid, [SYS_MSG, USER_MSG], ASSISTANT_MSG_1, [1, 2, 3], [10])

        bad_messages = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1, {"role": "user", "content": "second"}]
        with pytest.raises(ValueError, match="invalid message structure"):
            manager.try_prepare_pretokenized(sid, bad_messages)

    def test_session_not_found_raises(self, manager: SingleUserTurnTrajectoryManager):
        with pytest.raises(ValueError, match="session not found"):
            manager.try_prepare_pretokenized("nonexistent", [SYS_MSG, USER_MSG])

    def test_no_system_message(self, manager: SingleUserTurnTrajectoryManager):
        """Works without system message (system is optional)."""
        sid = manager.create_session()
        msgs = [USER_MSG]
        manager.update_pretokenized_state(sid, msgs, ASSISTANT_MSG_1, [1, 2], [10])

        t2_msgs = [USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1]
        result = manager.try_prepare_pretokenized(sid, t2_msgs)
        assert result == {"pretokenized_token_ids": [1, 2, 10], "pretokenized_num_message": 2}

    def test_append_system_message_allowed(self, manager: SingleUserTurnTrajectoryManager):
        """Appending a system message after tool messages is allowed (e.g. retry prompt)."""
        sid = manager.create_session()
        manager.update_pretokenized_state(sid, [SYS_MSG, USER_MSG], ASSISTANT_MSG_1, [1, 2, 3], [10, 11])

        messages = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1, RETRY_SYS_MSG]
        result = manager.try_prepare_pretokenized(sid, messages)
        assert result is not None
        assert result["pretokenized_token_ids"] == [1, 2, 3, 10, 11]
        assert result["pretokenized_num_message"] == 3

    def test_append_system_then_assistant_trajectory(self, manager: SingleUserTurnTrajectoryManager):
        """Full trajectory with a retry system message between tool-call turns."""
        sid = manager.create_session()

        # Turn 1: [sys, user] -> assistant(tool_call)
        t1_msgs = [SYS_MSG, USER_MSG]
        manager.update_pretokenized_state(sid, t1_msgs, ASSISTANT_MSG_1, [1, 2, 3], [10, 11])

        # Turn 2: append tool + system_retry -> assistant(tool_call)
        t2_msgs = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1, RETRY_SYS_MSG]
        result = manager.try_prepare_pretokenized(sid, t2_msgs)
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
        result = manager.try_prepare_pretokenized(sid, t3_msgs)
        assert result is not None
        assert result["pretokenized_num_message"] == 6

    def test_multiple_system_messages_at_start(self, manager: SingleUserTurnTrajectoryManager):
        """Multiple system messages before the user message are allowed."""
        sid = manager.create_session()
        extra_sys = {"role": "system", "content": "Extra instructions."}
        msgs = [SYS_MSG, extra_sys, USER_MSG]
        result = manager.try_prepare_pretokenized(sid, msgs)
        assert result is None  # first turn, no prior tokens

        manager.update_pretokenized_state(sid, msgs, ASSISTANT_MSG_1, [1, 2, 3, 4], [10, 11])
        session = manager.sessions[sid]
        assert session.messages == [SYS_MSG, extra_sys, USER_MSG, ASSISTANT_MSG_1]

        t2_msgs = [SYS_MSG, extra_sys, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1]
        result = manager.try_prepare_pretokenized(sid, t2_msgs)
        assert result is not None
        assert result["pretokenized_token_ids"] == [1, 2, 3, 4, 10, 11]

    def test_not_append_only_rejects_user_message(self, manager: SingleUserTurnTrajectoryManager):
        """Appending a user message (not tool/system) is rejected."""
        sid = manager.create_session()
        manager.update_pretokenized_state(sid, [SYS_MSG, USER_MSG], ASSISTANT_MSG_1, [1, 2, 3], [10])

        bad = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1, {"role": "user", "content": "extra"}]
        with pytest.raises(ValueError, match="invalid message structure"):
            manager.try_prepare_pretokenized(sid, bad)
