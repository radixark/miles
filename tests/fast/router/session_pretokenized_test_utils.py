from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import requests
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from miles.rollout.session.session_server import SessionServer
from miles.utils.chat_template_utils import MismatchType, apply_chat_template, get_tito_tokenizer
from miles.utils.http_utils import find_available_port
from miles.utils.processing_utils import load_tokenizer
from miles.utils.test_utils.uvicorn_thread_server import UvicornThreadServer

FORBIDDEN_MISMATCH_TYPES: frozenset[str] = frozenset(
    {
        MismatchType.SPECIAL_TOKEN_COUNT.value,
        MismatchType.SPECIAL_TOKEN_TYPE.value,
        MismatchType.NON_ASSISTANT_TEXT.value,
    }
)


@dataclass(frozen=True)
class ScriptedBackendTurn:
    response_message: dict[str, Any]
    render_message: dict[str, Any]


def load_test_tokenizer(hf_checkpoint: str, chat_template_path: str | None):
    return load_tokenizer(
        hf_checkpoint,
        chat_template_path=chat_template_path,
        trust_remote_code=True,
    )


def make_router_env(
    backend,
    *,
    hf_checkpoint: str,
    chat_template_path: str | None,
    tito_model: str,
    allowed_append_roles: list[str],
):
    args = SimpleNamespace(
        miles_router_timeout=30,
        hf_checkpoint=hf_checkpoint,
        chat_template_path=chat_template_path,
        tito_model=tito_model,
        tito_allowed_append_roles=allowed_append_roles,
        use_rollout_routing_replay=False,
    )
    session_server = SessionServer(args, backend_url=backend.url)

    port = find_available_port(31000)
    server = UvicornThreadServer(session_server.app, host="127.0.0.1", port=port)
    server.start()

    return SimpleNamespace(
        url=f"http://127.0.0.1:{port}",
        backend=backend,
        server=server,
    )


def teardown_router_env(env) -> None:
    env.server.stop()
    env.backend.stop()


def fetch_session_payload(base_url: str, session_id: str) -> dict[str, Any]:
    response = requests.get(f"{base_url}/sessions/{session_id}", timeout=5.0)
    response.raise_for_status()
    return response.json()


def compute_local_session_mismatch(
    tokenizer,
    *,
    tito_model: str,
    allowed_append_roles: list[str],
    messages: list[dict[str, Any]],
    accumulated_token_ids: list[int],
    tools: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    comparator = get_tito_tokenizer(
        tokenizer,
        tokenizer_type=tito_model,
        allowed_append_roles=allowed_append_roles,
    ).create_comparator()
    expected_ids = apply_chat_template(
        messages,
        tokenizer=tokenizer,
        tools=tools,
        add_generation_prompt=False,
        tokenize=True,
    )
    return [m.to_dict() for m in comparator.compare_sequences(expected_ids, accumulated_token_ids)]


def forbidden_mismatches(mismatch: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [m for m in mismatch if m.get("type") in FORBIDDEN_MISMATCH_TYPES]


class ScriptedChatBackend:
    def __init__(self, tokenizer, scripted_turns: list[ScriptedBackendTurn]):
        self.tokenizer = tokenizer
        self._scripted_turns = scripted_turns
        self._call_count = 0
        self.request_log: list[dict[str, Any]] = []
        self.host = "127.0.0.1"
        self.port = find_available_port(32000)
        self.app = FastAPI()
        self._server = UvicornThreadServer(self.app, host=self.host, port=self.port)
        self._setup_routes()

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def start(self):
        self._server.start()

    def stop(self):
        self._server.stop()

    def reset_stats(self):
        self.request_log.clear()
        self._call_count = 0

    def _setup_routes(self):
        @self.app.get("/health")
        async def health():
            return JSONResponse(content={"status": "ok"})

        @self.app.post("/v1/chat/completions")
        async def chat_completions(request: Request):
            payload = await request.json()
            self.request_log.append(payload)

            idx = self._call_count
            assert idx < len(self._scripted_turns), f"Unexpected extra request #{idx + 1}"
            self._call_count += 1

            turn = self._scripted_turns[idx]
            messages = payload["messages"]
            tools = payload.get("tools")

            prompt_text = apply_chat_template(
                messages,
                tokenizer=self.tokenizer,
                tools=tools,
                add_generation_prompt=True,
                tokenize=False,
            )
            with_assistant_text = apply_chat_template(
                messages + [turn.render_message],
                tokenizer=self.tokenizer,
                tools=tools,
                add_generation_prompt=False,
                tokenize=False,
            )
            assert with_assistant_text.startswith(prompt_text), "Scripted assistant must extend prompt text"

            response_text = with_assistant_text[len(prompt_text) :]
            output_ids = self.tokenizer.encode(response_text, add_special_tokens=False)
            input_ids = payload.get("input_ids")
            prompt_ids = (
                list(input_ids)
                if input_ids is not None
                else apply_chat_template(
                    messages,
                    tokenizer=self.tokenizer,
                    tools=tools,
                    add_generation_prompt=True,
                    tokenize=True,
                )
            )

            return JSONResponse(
                content={
                    "id": f"scripted-{idx}",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "scripted-model",
                    "choices": [
                        {
                            "index": 0,
                            "message": turn.response_message,
                            "prompt_token_ids": prompt_ids,
                            "finish_reason": "tool_calls" if turn.response_message.get("tool_calls") else "stop",
                            "meta_info": {
                                "completion_tokens": len(output_ids),
                                "output_token_logprobs": [[-i / 128, tid] for i, tid in enumerate(output_ids)],
                            },
                        }
                    ],
                }
            )
