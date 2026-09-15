"""HTTP validation and server-owned request fields."""

import pytest
import requests
from tests.fast.router.test_sessions import _create_session, _post_chat, _serve_router
from tests.fast.router.test_sessions_v2 import _serve_router as _serve_router_v2

from miles.utils.lora import LORA_ADAPTER_NAME

USER = {"role": "user", "content": "hi"}
LAUNCH_KWARGS = {"enable_thinking": False}  # both ``_serve_router`` helpers launch with this
THINKING_ON = {"enable_thinking": True}
TOOLS = [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {}}}}]
OTHER_TOOLS = [
    {"type": "function", "function": {"name": "get_time", "parameters": {"type": "object", "properties": {}}}}
]


def _serve(version: str, extra_args: dict | None = None):
    serve = _serve_router_v2 if version == "v2" else _serve_router
    return serve(extra_args)


def _records(url: str, session_id: str) -> list[dict]:
    return requests.get(f"{url}/sessions/{session_id}", timeout=5.0).json()["records"]


def _metadata(url: str, session_id: str) -> dict:
    return requests.get(f"{url}/sessions/{session_id}", timeout=5.0).json()["metadata"]


class TestForbiddenClientFields:
    @pytest.mark.parametrize(
        ("field", "value"),
        [("input_ids", [1, 2, 3]), ("routed_experts_start_len", 0), ("logprob_start_len", 0), ("lora_path", "x")],
    )
    def test_tito_control_fields_return_400_and_record_nothing(self, field, value):
        with _serve_router() as env:
            session_id = _create_session(env.url)
            resp = _post_chat(env.url, session_id, {"messages": [USER], field: value})
            assert resp.status_code == 400
            assert f"{field}={value!r} is not accepted" in resp.json()["error"]
            assert _records(env.url, session_id) == []

    def test_model_adapter_suffix_rejected_only_when_lora_rollout_is_enabled(self):
        with _serve_router({"lora_rank": 8}) as env:
            session_id = _create_session(env.url)
            resp = _post_chat(env.url, session_id, {"messages": [USER], "model": "base:adapter"})
            assert resp.status_code == 400
            assert "LoRA adapter" in resp.json()["error"]
        with _serve_router() as env:
            session_id = _create_session(env.url)
            assert _post_chat(env.url, session_id, {"messages": [USER], "model": "base:adapter"}).status_code == 200
            assert env.backend.request_log[-1]["model"] == "base:adapter"


class TestServerOwnedFields:
    def test_client_values_are_replaced_and_replay_flags_are_always_present(self):
        with _serve_router() as env:
            session_id = _create_session(env.url)
            resp = _post_chat(
                env.url,
                session_id,
                {
                    "messages": [USER],
                    "temperature": 0.7,
                    "logprobs": False,
                    "return_meta_info": False,
                    "no_stop_trim": True,
                    "return_routed_experts": True,
                    "return_indexer_topk": True,
                },
            )
            assert resp.status_code == 200
            wire = env.backend.request_log[-1]
            assert wire["logprobs"] is True
            assert wire["return_meta_info"] is True
            assert wire["no_stop_trim"] is False
            assert wire["return_routed_experts"] is False
            assert wire["return_indexer_topk"] is False
            assert "lora_path" not in wire
            assert wire["temperature"] == 0.7  # not in the table: the client's

    def test_lora_path_follows_lora_rollout_enabled(self):
        with _serve_router({"lora_rank": 8}) as env:
            session_id = _create_session(env.url)
            assert _post_chat(env.url, session_id, {"messages": [USER]}).status_code == 200
            assert env.backend.request_log[-1]["lora_path"] == LORA_ADAPTER_NAME
        with _serve_router({"lora_rank": 8, "lora_train_only": True}) as env:
            session_id = _create_session(env.url)
            assert _post_chat(env.url, session_id, {"messages": [USER]}).status_code == 200
            assert "lora_path" not in env.backend.request_log[-1]


class TestChatTemplateKwargs:
    def test_request_kwargs_override_the_launch_for_renderer_and_wire(self):
        with _serve_router() as env:
            session_id = _create_session(env.url)
            assert _post_chat(env.url, session_id, {"messages": [USER]}).status_code == 200
            launch_wire = env.backend.request_log[-1]
            assert launch_wire["chat_template_kwargs"] == LAUNCH_KWARGS

            session_id = _create_session(env.url)
            resp = _post_chat(env.url, session_id, {"messages": [USER], "chat_template_kwargs": THINKING_ON})
            assert resp.status_code == 200
            wire = env.backend.request_log[-1]
            assert wire["chat_template_kwargs"] == THINKING_ON
            assert wire["input_ids"] != launch_wire["input_ids"]  # the local render followed the request

    def test_non_object_chat_template_kwargs_is_400(self):
        with _serve_router() as env:
            session_id = _create_session(env.url)
            resp = _post_chat(env.url, session_id, {"messages": [USER], "chat_template_kwargs": "oops"})
            assert resp.status_code == 400
            assert resp.json()["error"] == "chat_template_kwargs must be an object"

    def test_tools_inside_chat_template_kwargs_is_400(self):
        with _serve_router() as env:
            session_id = _create_session(env.url)
            resp = _post_chat(env.url, session_id, {"messages": [USER], "chat_template_kwargs": {"tools": TOOLS}})
            assert resp.status_code == 400
            assert "tools belongs at the top level" in resp.json()["error"]
