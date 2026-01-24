import json
import time
import uuid
from typing import TYPE_CHECKING

from fastapi import Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel
from transformers import AutoTokenizer

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
    records: list[SessionRecord]


class SessionManager:
    def __init__(self):
        self.sessions: dict[str, list[SessionRecord]] = {}

    def create_session(self) -> str:
        session_id = uuid.uuid4().hex
        self.sessions[session_id] = []
        return session_id

    def get_session(self, session_id: str) -> list[SessionRecord] | None:
        return self.sessions.get(session_id)

    def delete_session(self, session_id: str) -> list[SessionRecord]:
        assert session_id in self.sessions
        return self.sessions.pop(session_id)

    def add_record(self, session_id: str, record: SessionRecord):
        assert session_id in self.sessions
        self.sessions[session_id].append(record)


def setup_session_routes(app, router: "MilesRouter"):
    manager = SessionManager()

    # TODO temporary hack before @guapisolo implements TITO
    # ============================= HACK START ===============================
    # Lazy load tokenizer only when needed (for tests that don't have hf_checkpoint)
    tokenizer = None

    def get_tokenizer():
        nonlocal tokenizer
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(router.args.hf_checkpoint, trust_remote_code=True)
        return tokenizer

    # ============================= HACK END ===============================

    @app.post("/sessions")
    async def create_session():
        session_id = manager.create_session()
        return {"session_id": session_id}

    @app.get("/sessions/{session_id}")
    async def get_session(session_id: str):
        records = manager.get_session(session_id)
        if records is None:
            return JSONResponse(status_code=404, content={"error": "session not found"})
        return GetSessionResponse(session_id=session_id, records=records)

    @app.delete("/sessions/{session_id}")
    async def delete_session(session_id: str):
        if session_id not in manager.sessions:
            return JSONResponse(status_code=404, content={"error": "session not found"})
        manager.delete_session(session_id)
        return Response(status_code=204)

    @app.api_route("/sessions/{session_id}/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
    async def session_proxy(request: Request, session_id: str, path: str):
        if session_id not in manager.sessions:
            return JSONResponse(status_code=404, content={"error": "session not found"})

        result = await router._do_proxy(request, path)

        request_body = json.loads(result["request_body"])
        response_body = json.loads(result["response_body"])

        # TODO: remove this hack when @guapisolo implements the real TITO
        # ============================= HACK START ===============================
        if "messages" in request_body and "input_ids" not in request_body:
            request_body["input_ids"] = get_tokenizer().apply_chat_template(
                request_body["messages"],
                add_generation_prompt=True,
                add_special_tokens=False,
                tools=request_body.get("tools"),
            )
        if (
            "logprobs" in response_body.get("choices", [{}])[0]
            and "content" in response_body["choices"][0]["logprobs"]
        ):
            logprobs_content = response_body["choices"][0]["logprobs"]["content"]
            for item in logprobs_content:
                if "token" in item and "token_id" not in item:
                    item["token_id"] = get_tokenizer().convert_tokens_to_ids(item["token"])
        # ============================= HACK END ===============================

        record = SessionRecord(
            timestamp=time.time(),
            method=request.method,
            path=path,
            request=request_body,
            response=response_body,
            status_code=result["status_code"],
        )
        manager.add_record(session_id, record)

        return router._build_proxy_response(result)
