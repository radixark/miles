from types import SimpleNamespace

import pytest
import requests

from miles.router.router import MilesRouter
from miles.router.session.naive_trajectory import NaiveTrajectoryManager, TokenInfo
from miles.utils.http_utils import find_available_port
from miles.utils.test_utils.mock_sglang_server import ProcessResult, with_mock_server
from miles.utils.test_utils.uvicorn_thread_server import UvicornThreadServer


class DummyTokenizer:
    """Minimal tokenizer stub for testing NaiveTrajectoryManager."""

    def apply_chat_template(
        self,
        messages,
        tokenize: bool = True,
        add_special_tokens: bool = False,
        add_generation_prompt: bool = True,
    ):
        """Return deterministic token ids based on message count."""
        base = len(messages) or 1
        return [base, base + 1, base + 2]


@pytest.fixture
def naive_manager():
    """Create a NaiveTrajectoryManager with a dummy tokenizer."""
    args = SimpleNamespace()
    tokenizer = DummyTokenizer()
    return NaiveTrajectoryManager(args, tokenizer)


class TestNaiveTrajectoryManager:
    def test_create_session(self, naive_manager: NaiveTrajectoryManager):
        session_id = naive_manager.create_session()
        assert session_id is not None
        assert len(session_id) == 32
        assert session_id in naive_manager.sessions

    def test_get_token_info_by_id(self, naive_manager: NaiveTrajectoryManager):
        session_id = naive_manager.create_session()
        token_info = naive_manager.get_token_info_by_id(session_id)
        assert isinstance(token_info, TokenInfo)
        assert token_info.token_ids == []
        assert token_info.log_probs == []
        assert token_info.loss_mask == []

    def test_get_token_info_by_id_not_found(self, naive_manager: NaiveTrajectoryManager):
        token_info = naive_manager.get_token_info_by_id("nonexistent")
        assert token_info is None

    def test_calc_prompt_tokens_for_existing_session(self, naive_manager: NaiveTrajectoryManager):
        session_id = naive_manager.create_session()
        messages = [{"role": "user", "content": "hello"}]

        token_info = naive_manager.calc_prompt_tokens(session_id, messages)

        assert isinstance(token_info, TokenInfo)
        assert token_info.token_ids == [1, 2, 3]
        assert len(token_info.log_probs) == len(token_info.token_ids)
        assert len(token_info.loss_mask) == len(token_info.token_ids)

    def test_calc_prompt_tokens_for_missing_session(self, naive_manager: NaiveTrajectoryManager):
        messages = [{"role": "user", "content": "hello"}]
        with pytest.raises(ValueError):
            naive_manager.calc_prompt_tokens("missing", messages)

    def test_delete_session_by_id(self, naive_manager: NaiveTrajectoryManager):
        session_id = naive_manager.create_session()
        assert naive_manager.delete_session_by_id(session_id) is True
        assert session_id not in naive_manager.sessions
        assert naive_manager.delete_session_by_id(session_id) is False

    def test_update_record(self, naive_manager: NaiveTrajectoryManager):
        session_id = naive_manager.create_session()
        messages = [{"role": "user", "content": "hello"}]
        token_info = TokenInfo(token_ids=[1], log_probs=[0.0], loss_mask=[1])

        updated = naive_manager.update_record(session_id, messages, token_info)

        assert updated is True
        stored = naive_manager.sessions[session_id]
        assert stored.messages == messages
        assert stored.token_info == token_info

    def test_update_record_missing_session(self, naive_manager: NaiveTrajectoryManager):
        messages = [{"role": "user", "content": "hello"}]
        token_info = TokenInfo(token_ids=[1], log_probs=[0.0], loss_mask=[1])
        with pytest.raises(ValueError):
            naive_manager.update_record("missing", messages, token_info)


@pytest.fixture(scope="class")
def router_env():
    """Create a MilesRouter with session routes and a mock backend."""

    def process_fn(prompt: str) -> ProcessResult:
        return ProcessResult(text=f"echo: {prompt}", finish_reason="stop")

    with with_mock_server(process_fn=process_fn) as backend:
        args = SimpleNamespace(
            miles_router_max_connections=10,
            miles_router_timeout=30,
            miles_router_middleware_paths=[],
            rollout_health_check_interval=60,
            miles_router_health_check_failure_threshold=3,
            hf_checkpoint="Qwen/Qwen3-0.6B",
            trajectory_manager="naive_trajectory",
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
        assert data["records"] == {"token_ids": [], "log_probs": [], "loss_mask": []}

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
    def test_proxy_chat_updates_token_info(self, router_env):
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

        assert isinstance(records["token_ids"], list)
        assert isinstance(records["log_probs"], list)
        assert isinstance(records["loss_mask"], list)
        assert len(records["token_ids"]) > 0
        assert len(records["token_ids"]) == len(records["log_probs"]) == len(records["loss_mask"])
        assert all(mask == 1 for mask in records["loss_mask"])
