import json
import logging
import time
from typing import TYPE_CHECKING

from fastapi import Request
from fastapi.responses import JSONResponse, Response

from miles.router.session.session_types import GetSessionResponse, SessionRecord
from miles.router.session.single_user_turn_trajectory import SingleUserTurnTrajectoryManager
from miles.utils.processing_utils import load_tokenizer

if TYPE_CHECKING:
    from miles.router.router import MilesRouter

logger = logging.getLogger(__name__)


def setup_session_routes(app, router: "MilesRouter"):
    hf_checkpoint = getattr(router.args, "hf_checkpoint", None)
    if not hf_checkpoint:
        if getattr(router, "verbose", False):
            logger.info("[miles-router] Skipping session routes (hf_checkpoint not set).")
        return

    tokenizer = load_tokenizer(
        hf_checkpoint, chat_template_path=router.args.chat_template_path, trust_remote_code=True
    )

    manager = SingleUserTurnTrajectoryManager(router.args, tokenizer)

    @app.post("/sessions")
    async def create_session():
        session_id = manager.create_session()
        return {"session_id": session_id}

    @app.get("/sessions/{session_id}")
    async def get_session(session_id: str):
        records = manager.get_session_records_by_id(session_id)
        if records is None:
            return JSONResponse(status_code=404, content={"error": "session not found"})
        return GetSessionResponse(
            session_id=session_id,
            records=records,
        )

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

        # Ensure SGLang returns token IDs and logprobs for TITO, regardless
        # of whether the upstream agent (e.g. mini-swe-agent) requested them.
        request_body.setdefault("logprobs", True)
        request_body.setdefault("return_prompt_token_ids", True)

        # Try to inject pretokenized token IDs for prefix reuse.
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

        body = json.dumps(request_body).encode()

        result = await router._do_proxy(request, "v1/chat/completions", body=body)

        response = json.loads(result["response_body"])

        choice = response.get("choices", [{}])[0]

        if "logprobs" not in choice or "content" not in choice["logprobs"]:
            raise RuntimeError("logprobs must be in choice")
        logprobs_content = choice["logprobs"]["content"]
        for item in logprobs_content:
            if "token_id" not in item:
                raise RuntimeError("token_id must be in choice's logprobs content item")

        # Extract token IDs and update pretokenized state for next turn.
        prompt_token_ids = choice.get("prompt_token_ids", [])
        completion_token_ids = [item["token_id"] for item in logprobs_content]
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
        return router._build_proxy_response(result)

    @app.api_route("/sessions/{session_id}/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
    async def session_proxy(request: Request, session_id: str, path: str):
        result = await router._do_proxy(request, path)
        return router._build_proxy_response(result)
