import json
import time
from typing import TYPE_CHECKING

from fastapi import Request
from fastapi.responses import JSONResponse, Response
from transformers import AutoTokenizer

from miles.router.session.session_types import GetSessionResponse, SessionRecord

if TYPE_CHECKING:
    from miles.router.router import MilesRouter


def setup_session_routes(app, router: "MilesRouter"):

    tokenizer = AutoTokenizer.from_pretrained(router.args.hf_checkpoint, trust_remote_code=True)
    if router.args.trajectory_manager == "naive_trajectory":
        from miles.router.session.naive_trajectory import NaiveTrajectoryManager

        manager = NaiveTrajectoryManager(router.args, tokenizer)
    else:
        raise ValueError(f"Invalid trajectory manager: {router.args.trajectory_manager}")

    @app.post("/sessions")
    async def create_session():
        session_id = manager.create_session()
        return {"session_id": session_id}

    @app.get("/sessions/{session_id}")
    async def get_session(session_id: str):
        try:
            records = manager.get_session_records_by_id(session_id)
        except ValueError:
            return JSONResponse(status_code=404, content={"error": "session not found"})
        return GetSessionResponse(
            session_id=session_id,
            records=records,
        )

    @app.delete("/sessions/{session_id}")
    async def delete_session(session_id: str):
        try:
            manager.delete_session_by_id(session_id)
        except ValueError:
            return JSONResponse(status_code=404, content={"error": "session not found"})
        return Response(status_code=204)

    @app.api_route("/sessions/{session_id}/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
    async def session_proxy(request: Request, session_id: str, path: str):
        body = await request.body()
        request_body = json.loads(body) if body else {}

        if "messages" in request_body and "input_ids" not in request_body:
            try:
                prompt_token_ids = manager.calc_prompt_tokens(session_id, request_body["messages"])
                request_body["input_ids"] = prompt_token_ids
            except ValueError:
                return JSONResponse(status_code=404, content={"error": "session not found"})
            body = json.dumps(request_body).encode("utf-8")

        result = await router._do_proxy(request, path, body=body)

        response = json.loads(result["response_body"])

        choice = response.get("choices", [{}])[0]
        # messages = request_body["messages"] + [choice["message"]]

        assert "logprobs" in choice and "content" in choice["logprobs"], "logprobs must be in choice"
        logprobs_content = choice["logprobs"]["content"]

        for item in logprobs_content:
            if "token" in item and "token_id" not in item:
                item["token_id"] = tokenizer.convert_tokens_to_ids(item["token"])

        record = SessionRecord(
            timestamp=time.time(),
            method=request.method,
            path=path,
            status_code=result["status_code"],
            request=request_body,
            response=response,
        )
        try:
            manager.append_session_record(session_id, record)
        except ValueError:
            return JSONResponse(status_code=404, content={"error": "session not found"})
        return router._build_proxy_response(result)
