"""Standalone Session Server.

A lightweight FastAPI application that sits in front of an sglang instance,
manages session lifecycle, injects pretokenized prefixes for TITO, records
per-turn data, and transparently forwards everything else.

Usage (CLI):
    python -m miles.session_server.server \\
        --host 0.0.0.0 --port 8000 \\
        --upstream-url http://localhost:30000 \\
        --hf-checkpoint Qwen/Qwen3-0.6B

Environment variables:
    SESSION_SERVER_TIMEOUT  – HTTP proxy timeout in seconds (default: no timeout)
"""

import argparse
import json
import logging
import os
import time

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.responses import Response

from miles.router.session.session_types import GetSessionResponse, SessionRecord
from miles.router.session.single_user_turn_trajectory import SingleUserTurnTrajectoryManager
from miles.utils.processing_utils import load_tokenizer

logger = logging.getLogger(__name__)


def create_session_server_app(args) -> FastAPI:
    """Build and return the FastAPI application (useful for testing)."""
    app = FastAPI()

    upstream_url = args.upstream_url
    hf_checkpoint = getattr(args, "hf_checkpoint", None)

    timeout_env = os.environ.get("SESSION_SERVER_TIMEOUT")
    timeout = float(timeout_env) if timeout_env is not None else None

    client = httpx.AsyncClient(
        timeout=httpx.Timeout(timeout),
    )

    # ── session routes (only when hf_checkpoint is provided) ──────────
    if hf_checkpoint:
        tokenizer = load_tokenizer(
            hf_checkpoint,
            chat_template_path=getattr(args, "chat_template_path", None),
            trust_remote_code=True,
        )
        manager = SingleUserTurnTrajectoryManager(args, tokenizer)

        @app.post("/sessions")
        async def create_session():
            session_id = manager.create_session()
            return {"session_id": session_id}

        @app.get("/sessions/{session_id}")
        async def get_session(session_id: str):
            records = manager.get_session_records_by_id(session_id)
            if records is None:
                return JSONResponse(status_code=404, content={"error": "session not found"})
            return GetSessionResponse(session_id=session_id, records=records)

        @app.delete("/sessions/{session_id}")
        async def delete_session(session_id: str):
            deleted = manager.delete_session_by_id(session_id)
            if deleted is None:
                return JSONResponse(status_code=404, content={"error": "session not found"})
            return Response(status_code=204)

        @app.post("/sessions/{session_id}/v1/chat/completions")
        async def chat_completions(request: Request, session_id: str):
            body = await request.body()
            request_body = json.loads(body) if body else {}

            request_body.setdefault("logprobs", True)
            request_body.setdefault("return_prompt_token_ids", True)

            request_messages = request_body.get("messages", [])
            pretokenized = manager.try_prepare_pretokenized(session_id, request_messages)
            if pretokenized is not None:
                request_body["pretokenized_token_ids"] = pretokenized["pretokenized_token_ids"]
                request_body["pretokenized_num_message"] = pretokenized["pretokenized_num_message"]
                logger.debug(
                    "Using pretokenized input: %d tokens, %d messages",
                    len(pretokenized["pretokenized_token_ids"]),
                    pretokenized["pretokenized_num_message"],
                )

            result = await _do_proxy(
                client, request, upstream_url, "v1/chat/completions", body=json.dumps(request_body).encode()
            )

            response = json.loads(result["response_body"])
            choice = response.get("choices", [{}])[0]

            if "response_token_ids" not in choice:
                raise RuntimeError("response_token_ids must be in choice (requires logprobs=True)")

            prompt_token_ids = choice.get("prompt_token_ids", [])
            completion_token_ids = choice["response_token_ids"]
            assistant_message = choice.get("message", {})

            manager.update_pretokenized_state(
                session_id,
                request_messages,
                assistant_message,
                prompt_token_ids,
                completion_token_ids,
            )

            record = SessionRecord(
                timestamp=time.time(),
                method=request.method,
                path="/v1/chat/completions",
                status_code=result["status_code"],
                request=request_body,
                response=response,
            )
            appended = manager.append_session_record(session_id, record)
            if appended is None:
                return JSONResponse(status_code=404, content={"error": "session not found"})
            return _build_response(result)

        @app.api_route(
            "/sessions/{session_id}/{path:path}",
            methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
        )
        async def session_proxy(request: Request, session_id: str, path: str):
            result = await _do_proxy(client, request, upstream_url, path)
            return _build_response(result)

    # ── catch-all: forward everything else to sglang router ───────────
    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
    async def proxy_all(request: Request, path: str):
        result = await _do_proxy(client, request, upstream_url, path)
        return _build_response(result)

    return app


# ── internal helpers ──────────────────────────────────────────────────


async def _do_proxy(
    client: httpx.AsyncClient,
    request: Request,
    upstream_url: str,
    path: str,
    body: bytes | None = None,
) -> dict:
    """Forward a request to the upstream sglang router."""
    url = f"{upstream_url}/{path}"

    if body is None:
        body = await request.body()

    headers = dict(request.headers)
    headers = {k: v for k, v in headers.items() if k.lower() not in ("content-length", "transfer-encoding", "host")}

    response = await client.request(request.method, url, content=body, headers=headers)
    content = await response.aread()
    return {
        "response_body": content,
        "status_code": response.status_code,
        "headers": dict(response.headers),
    }


def _build_response(result: dict) -> Response:
    """Build an HTTP response from the proxy result dict."""
    content = result["response_body"]
    status_code = result["status_code"]
    headers = result["headers"]
    content_type = headers.get("content-type", "")
    try:
        data = json.loads(content)
        return JSONResponse(content=data, status_code=status_code, headers=headers)
    except Exception:
        return Response(content=content, status_code=status_code, headers=headers, media_type=content_type)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Session Server – standalone session management for sglang")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--upstream-url", type=str, required=True, help="Base URL of the sglang router")
    parser.add_argument(
        "--hf-checkpoint", type=str, default=None, help="HuggingFace model id (enables session routes)"
    )
    parser.add_argument("--chat-template-path", type=str, default=None)
    args = parser.parse_args()

    from types import SimpleNamespace

    ns = SimpleNamespace(
        upstream_url=args.upstream_url,
        hf_checkpoint=args.hf_checkpoint,
        chat_template_path=args.chat_template_path,
    )
    app = create_session_server_app(ns)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
