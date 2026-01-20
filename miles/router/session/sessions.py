import json
from typing import TYPE_CHECKING

from fastapi import Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel
from transformers import AutoTokenizer

from miles.router.session.naive_trajectory import NaiveTrajectoryManager, TokenInfo

if TYPE_CHECKING:
    from miles.router.router import MilesRouter


class SessionRecord(BaseModel):
    timestamp: float
    method: str
    path: str
    request: dict
    response: dict
    status_code: int


class GetSessionResponse(BaseModel):
    session_id: str
    records: dict


def setup_session_routes(app, router: "MilesRouter"):

    tokenizer = AutoTokenizer.from_pretrained(router.args.hf_checkpoint, trust_remote_code=True)
    if router.args.trajectory_manager == "naive_trajectory":
        manager = NaiveTrajectoryManager(router.args, tokenizer)
    else:
        raise ValueError(f"Invalid trajectory manager: {router.args.trajectory_manager}")

    @app.post("/sessions")
    async def create_session():
        session_id = manager.create_session()
        return {"session_id": session_id}

    @app.get("/sessions/{session_id}")
    async def get_session(session_id: str):
        token_info = manager.get_token_info_by_id(session_id)
        if token_info is None:
            return JSONResponse(status_code=404, content={"error": "session not found"})
        return GetSessionResponse(
            session_id=session_id,
            records=token_info.model_dump(),
        )

    @app.delete("/sessions/{session_id}")
    async def delete_session(session_id: str):
        status = manager.delete_session_by_id(session_id)
        if not status:
            return JSONResponse(status_code=404, content={"error": "session not found"})
        return Response(status_code=204)

    @app.api_route("/sessions/{session_id}/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
    async def session_proxy(request: Request, session_id: str, path: str):
        body = await request.body()
        request_body = json.loads(body) if body else {}

        prompt_token_info = TokenInfo()
        response_token_info = TokenInfo()
        if "messages" in request_body and "input_ids" not in request_body:
            prompt_token_info = manager.calc_prompt_tokens(session_id, request_body["messages"])
            if prompt_token_info is None:
                return JSONResponse(status_code=404, content={"error": "session not found"})
            request_body["input_ids"] = prompt_token_info.token_ids
            body = json.dumps(request_body).encode("utf-8")

        result = await router._do_proxy(request, path, body=body)

        response = json.loads(result["response_body"])

        choice = response.get("choices", [{}])[0]
        messages = request_body["messages"] + [choice["message"]]

        assert "logprobs" in choice and "content" in choice["logprobs"], "logprobs must be in choice"
        logprobs_content = choice["logprobs"]["content"]

        for item in logprobs_content:
            if "token" in item and "token_id" not in item:
                item["token_id"] = tokenizer.convert_tokens_to_ids(item["token"])
            response_token_info.append(item["token_id"], item["logprob"], 1)

        manager.update_record(
            session_id,
            messages,
            response_token_info,
        )
        return router._build_proxy_response(result)
