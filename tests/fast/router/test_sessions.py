from types import SimpleNamespace
from unittest.mock import patch

import pytest
import requests

from miles.router.router import MilesRouter
from miles.router.session.session_types import SessionRecord
from miles.router.session.single_user_turn_trajectory import SingleUserTurnTrajectoryManager
from miles.utils.http_utils import find_available_port
from miles.utils.test_utils.mock_sglang_server import MockSGLangServer, ProcessResult, with_mock_server
from miles.utils.test_utils.uvicorn_thread_server import UvicornThreadServer


@pytest.fixture
def single_user_turn_manager():
    """Create a SingleUserTurnTrajectoryManager with a dummy tokenizer."""
    args = SimpleNamespace()
    return SingleUserTurnTrajectoryManager(args, tokenizer=None)


class TestSingleUserTurnTrajectoryManager:
    def test_create_session(self, single_user_turn_manager: SingleUserTurnTrajectoryManager):
        session_id = single_user_turn_manager.create_session()
        assert session_id is not None
        assert len(session_id) == 32
        assert session_id in single_user_turn_manager.sessions

    def test_get_session_records_by_id(self, single_user_turn_manager: SingleUserTurnTrajectoryManager):
        session_id = single_user_turn_manager.create_session()
        records = single_user_turn_manager.get_session_records_by_id(session_id)
        assert records == []

    def test_get_session_records_by_id_not_found(self, single_user_turn_manager: SingleUserTurnTrajectoryManager):
        records = single_user_turn_manager.get_session_records_by_id("nonexistent")
        assert records is None

    def test_delete_session_by_id(self, single_user_turn_manager: SingleUserTurnTrajectoryManager):
        session_id = single_user_turn_manager.create_session()
        assert single_user_turn_manager.delete_session_by_id(session_id) is True
        assert session_id not in single_user_turn_manager.sessions
        assert single_user_turn_manager.delete_session_by_id(session_id) is None

    def test_append_session_record(self, single_user_turn_manager: SingleUserTurnTrajectoryManager):
        session_id = single_user_turn_manager.create_session()
        record = SessionRecord(
            timestamp=0.0,
            method="POST",
            path="/v1/chat/completions",
            status_code=200,
            request={"messages": [{"role": "user", "content": "hello"}]},
            response={"choices": []},
        )

        appended = single_user_turn_manager.append_session_record(session_id, record)

        assert appended is True
        records = single_user_turn_manager.get_session_records_by_id(session_id)
        assert records is not None
        assert len(records) == 1
        assert records[0].path == record.path

    def test_append_session_record_missing_session(self, single_user_turn_manager: SingleUserTurnTrajectoryManager):
        record = SessionRecord(
            timestamp=0.0,
            method="POST",
            path="/v1/chat/completions",
            status_code=200,
            request={},
            response={},
        )
        appended = single_user_turn_manager.append_session_record("missing", record)
        assert appended is None


# ---------------------------------------------------------------------------
# Messages / token helpers for multi-turn pretokenized tests
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


class TestPretokenizedMultiTurn:
    """Test try_prepare_pretokenized and update_pretokenized_state across turns."""

    def test_first_turn_returns_none(self, single_user_turn_manager: SingleUserTurnTrajectoryManager):
        """First turn has no prior token_ids, so try_prepare returns None."""
        sid = single_user_turn_manager.create_session()
        messages = [SYS_MSG, USER_MSG]
        result = single_user_turn_manager.try_prepare_pretokenized(sid, messages)
        assert result is None

    def test_two_turn_trajectory(self, single_user_turn_manager: SingleUserTurnTrajectoryManager):
        """Full 2-turn: user → assistant(tool_call) → tool → final answer."""
        sid = single_user_turn_manager.create_session()

        # --- Turn 1: [sys, user] → assistant with tool_call ---
        turn1_messages = [SYS_MSG, USER_MSG]
        assert single_user_turn_manager.try_prepare_pretokenized(sid, turn1_messages) is None

        turn1_prompt_ids = [1, 2, 3, 4, 5]
        turn1_completion_ids = [10, 11, 12]
        single_user_turn_manager.update_pretokenized_state(
            sid, turn1_messages, ASSISTANT_MSG_1, turn1_prompt_ids, turn1_completion_ids
        )

        # Verify internal state
        session = single_user_turn_manager.sessions[sid]
        assert session.messages == [SYS_MSG, USER_MSG, ASSISTANT_MSG_1]
        assert session.token_ids == [1, 2, 3, 4, 5, 10, 11, 12]

        # --- Turn 2: [sys, user, assistant, tool] → final answer ---
        turn2_messages = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1]
        result = single_user_turn_manager.try_prepare_pretokenized(sid, turn2_messages)
        assert result is not None
        assert result["pretokenized_token_ids"] == [1, 2, 3, 4, 5, 10, 11, 12]
        assert result["pretokenized_num_message"] == 3  # [sys, user, assistant]

        turn2_prompt_ids = [1, 2, 3, 4, 5, 10, 11, 12, 20, 21]
        turn2_completion_ids = [30, 31, 32]
        single_user_turn_manager.update_pretokenized_state(
            sid, turn2_messages, ASSISTANT_MSG_FINAL, turn2_prompt_ids, turn2_completion_ids
        )

        assert session.messages == [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1, ASSISTANT_MSG_FINAL]
        assert session.token_ids == [1, 2, 3, 4, 5, 10, 11, 12, 20, 21, 30, 31, 32]

    def test_three_turn_trajectory(self, single_user_turn_manager: SingleUserTurnTrajectoryManager):
        """Full 3-turn: user → ass(tool) → tool → ass(tool) → tool → final."""
        sid = single_user_turn_manager.create_session()

        # Turn 1
        t1_msgs = [SYS_MSG, USER_MSG]
        single_user_turn_manager.update_pretokenized_state(sid, t1_msgs, ASSISTANT_MSG_1, [1, 2, 3], [10, 11])

        # Turn 2
        t2_msgs = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1]
        result = single_user_turn_manager.try_prepare_pretokenized(sid, t2_msgs)
        assert result == {"pretokenized_token_ids": [1, 2, 3, 10, 11], "pretokenized_num_message": 3}

        single_user_turn_manager.update_pretokenized_state(
            sid, t2_msgs, ASSISTANT_MSG_2, [1, 2, 3, 10, 11, 20, 21], [30, 31]
        )

        # Turn 3
        t3_msgs = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1, ASSISTANT_MSG_2, TOOL_MSG_2]
        result = single_user_turn_manager.try_prepare_pretokenized(sid, t3_msgs)
        assert result == {
            "pretokenized_token_ids": [1, 2, 3, 10, 11, 20, 21, 30, 31],
            "pretokenized_num_message": 5,  # [sys, user, ass1, tool1, ass2]
        }

        single_user_turn_manager.update_pretokenized_state(
            sid, t3_msgs, ASSISTANT_MSG_FINAL, [1, 2, 3, 10, 11, 20, 21, 30, 31, 40], [50, 51]
        )

        session = single_user_turn_manager.sessions[sid]
        assert len(session.messages) == 7  # sys, user, ass1, tool1, ass2, tool2, final
        assert session.token_ids == [1, 2, 3, 10, 11, 20, 21, 30, 31, 40, 50, 51]

    def test_prefix_mismatch_raises(self, single_user_turn_manager: SingleUserTurnTrajectoryManager):
        """update_pretokenized_state asserts stored token_ids is prefix of new."""
        sid = single_user_turn_manager.create_session()
        single_user_turn_manager.update_pretokenized_state(
            sid, [SYS_MSG, USER_MSG], ASSISTANT_MSG_1, [1, 2, 3], [10, 11]
        )

        with pytest.raises(AssertionError, match="pretokenized prefix mismatch"):
            single_user_turn_manager.update_pretokenized_state(
                sid,
                [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1],
                ASSISTANT_MSG_FINAL,
                [9, 9, 9, 20, 21],  # does NOT start with [1,2,3,10,11]
                [30],
            )

    def test_not_append_only_raises(self, single_user_turn_manager: SingleUserTurnTrajectoryManager):
        """try_prepare raises when new messages modify stored prefix."""
        sid = single_user_turn_manager.create_session()
        single_user_turn_manager.update_pretokenized_state(sid, [SYS_MSG, USER_MSG], ASSISTANT_MSG_1, [1, 2, 3], [10])

        # Append an assistant message instead of tool — violates append-only
        bad_messages = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, {"role": "assistant", "content": "oops"}]
        with pytest.raises(ValueError, match="not append-only"):
            single_user_turn_manager.try_prepare_pretokenized(sid, bad_messages)

    def test_multiple_user_messages_raises(self, single_user_turn_manager: SingleUserTurnTrajectoryManager):
        """try_prepare raises when messages contain multiple user messages."""
        sid = single_user_turn_manager.create_session()
        single_user_turn_manager.update_pretokenized_state(sid, [SYS_MSG, USER_MSG], ASSISTANT_MSG_1, [1, 2, 3], [10])

        bad_messages = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1, {"role": "user", "content": "second"}]
        with pytest.raises(ValueError, match="invalid message structure"):
            single_user_turn_manager.try_prepare_pretokenized(sid, bad_messages)

    def test_session_not_found_raises(self, single_user_turn_manager: SingleUserTurnTrajectoryManager):
        with pytest.raises(ValueError, match="session not found"):
            single_user_turn_manager.try_prepare_pretokenized("nonexistent", [SYS_MSG, USER_MSG])

    def test_no_system_message(self, single_user_turn_manager: SingleUserTurnTrajectoryManager):
        """Works without system message (system is optional)."""
        sid = single_user_turn_manager.create_session()
        msgs = [USER_MSG]
        single_user_turn_manager.update_pretokenized_state(sid, msgs, ASSISTANT_MSG_1, [1, 2], [10])

        t2_msgs = [USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1]
        result = single_user_turn_manager.try_prepare_pretokenized(sid, t2_msgs)
        assert result == {"pretokenized_token_ids": [1, 2, 10], "pretokenized_num_message": 2}


@pytest.fixture(scope="class")
def router_env():
    """Create a MilesRouter with session routes and a mock backend."""

    def process_fn(prompt: str) -> ProcessResult:
        return ProcessResult(text=f"echo: {prompt}", finish_reason="stop")

    original_chat_response = MockSGLangServer._compute_chat_completions_response

    def patched_chat_response(self, payload: dict) -> dict:
        response = original_chat_response(self, payload)
        logprobs_content = response["choices"][0]["logprobs"]["content"]
        for item in logprobs_content:
            item["token_id"] = self.tokenizer.convert_tokens_to_ids(item["token"])
        return response

    with patch.object(MockSGLangServer, "_compute_chat_completions_response", new=patched_chat_response):
        with with_mock_server(process_fn=process_fn) as backend:
            args = SimpleNamespace(
                miles_router_max_connections=10,
                miles_router_timeout=30,
                miles_router_middleware_paths=[],
                rollout_health_check_interval=60,
                miles_router_health_check_failure_threshold=3,
                hf_checkpoint="Qwen/Qwen3-0.6B",
                trajectory_manager="single_user_turn_trajectory",
            )
            router = MilesRouter(args)

            port = find_available_port(31000)
            server = UvicornThreadServer(router.app, host="127.0.0.1", port=port)
            server.start()

            url = f"http://127.0.0.1:{port}"
            requests.post(f"{url}/add_worker", json={"url": backend.url}, timeout=5.0)

            try:
                yield SimpleNamespace(url=url)
            finally:
                server.stop()


class TestSessionRoutes:
    def test_create_session(self, router_env):
        response = requests.post(f"{router_env.url}/sessions", timeout=5.0)
        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
        assert len(data["session_id"]) == 32

    def test_get_session_initial_state(self, router_env):
        session_id = requests.post(f"{router_env.url}/sessions", timeout=5.0).json()["session_id"]

        get_resp = requests.get(f"{router_env.url}/sessions/{session_id}", timeout=5.0)
        assert get_resp.status_code == 200
        data = get_resp.json()
        assert data["session_id"] == session_id
        assert data["records"] == []

    def test_get_session_not_found(self, router_env):
        response = requests.get(f"{router_env.url}/sessions/nonexistent", timeout=5.0)
        assert response.status_code == 404
        assert response.json()["error"] == "session not found"

    def test_delete_session(self, router_env):
        session_id = requests.post(f"{router_env.url}/sessions", timeout=5.0).json()["session_id"]

        delete_resp = requests.delete(f"{router_env.url}/sessions/{session_id}", timeout=5.0)
        assert delete_resp.status_code == 204
        assert delete_resp.text == ""

        assert requests.delete(f"{router_env.url}/sessions/{session_id}", timeout=5.0).status_code == 404

    def test_delete_session_not_found(self, router_env):
        response = requests.delete(f"{router_env.url}/sessions/nonexistent", timeout=5.0)
        assert response.status_code == 404
        assert response.json()["error"] == "session not found"


class TestSessionProxy:
    def test_proxy_chat_appends_record(self, router_env):
        session_id = requests.post(f"{router_env.url}/sessions", timeout=5.0).json()["session_id"]

        payload = {
            "messages": [{"role": "user", "content": "What is 1+2?"}],
            "return_logprob": True,
        }
        resp = requests.post(
            f"{router_env.url}/sessions/{session_id}/v1/chat/completions",
            json=payload,
            timeout=10.0,
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "choices" in body
        assert body["choices"]

        get_resp = requests.get(f"{router_env.url}/sessions/{session_id}", timeout=5.0)
        records = get_resp.json()["records"]

        assert isinstance(records, list)
        assert len(records) == 1
        record = records[0]
        assert record["path"] == "/v1/chat/completions"
        assert record["status_code"] == 200
