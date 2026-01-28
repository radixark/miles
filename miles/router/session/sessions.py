import json
import logging
import time
from typing import TYPE_CHECKING

from fastapi import Request
from fastapi.responses import JSONResponse, Response
from transformers import AutoTokenizer

from miles.router.session.naive_trajectory import NaiveTrajectoryManager
from miles.router.session.session_types import GetSessionResponse, SessionRecord

if TYPE_CHECKING:
    from miles.router.router import MilesRouter

logger = logging.getLogger(__name__)


def setup_session_routes(app, router: "MilesRouter"):
    hf_checkpoint = getattr(router.args, "hf_checkpoint", None)
    if not hf_checkpoint:
        if getattr(router, "verbose", False):
            logger.info("[miles-router] Skipping session routes (hf_checkpoint not set).")
        return

    tokenizer = AutoTokenizer.from_pretrained(hf_checkpoint, trust_remote_code=True)
    manager = NaiveTrajectoryManager(router.args, tokenizer)

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

        if router.args.miles_router_enable_token_input_for_chat_completions:
            if "messages" in request_body and "input_ids" not in request_body:
                prompt_token_ids = manager.calc_prompt_tokens(session_id, request_body["messages"])
                if prompt_token_ids is None:
                    return JSONResponse(status_code=404, content={"error": "session not found"})
                request_body["input_ids"] = prompt_token_ids
                body = json.dumps(request_body).encode("utf-8")

        result = await router._do_proxy(request, "v1/chat/completions", body=body)

        response = json.loads(result["response_body"])

        choice = response.get("choices", [{}])[0]
        # messages = request_body["messages"] + [choice["message"]]

        if "logprobs" not in choice or "content" not in choice["logprobs"]:
            raise RuntimeError("logprobs must be in choice")
        logprobs_content = choice["logprobs"]["content"]

        for item in logprobs_content:
            if "token_id" not in item:
                raise RuntimeError("token_id must be in item")
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
