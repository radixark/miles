import json
import time
import uuid
from typing import TYPE_CHECKING

from fastapi import Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

if TYPE_CHECKING:
    from miles.router.router import MilesRouter


# TODO refine after @guapisolo's implementation
class SessionRecordChatCompletionsExtras(BaseModel):
    input_ids: list[int]
    output_ids: list[int]


class SessionRecord(BaseModel):
    timestamp: float
    method: str
    path: str
    request: dict
    response: dict
    extras: SessionRecordChatCompletionsExtras | None
    status_code: int


class DeleteSessionResponse(BaseModel):
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

    @app.post("/sessions")
    async def create_session():
        session_id = manager.create_session()
        return {"session_id": session_id}

    @app.delete("/sessions/{session_id}")
    async def delete_session(session_id: str) -> JSONResponse | DeleteSessionResponse:
        if session_id not in manager.sessions:
            return JSONResponse(status_code=404, content={"error": "session not found"})
        records = manager.delete_session(session_id)
        return DeleteSessionResponse(session_id=session_id, records=records)

    @app.post("/sessions/{session_id}/v1/chat/completions")
    async def session_chat_completions(request: Request, session_id: str):
        if session_id not in manager.sessions:
            return JSONResponse(status_code=404, content={"error": "session not found"})

        # TODO may need to pass `session_id` for token-id-consistent oai endpoint processing
        result = await router._do_proxy(request, "v1/chat/completions")

        record = SessionRecord(
            timestamp=time.time(),
            method=request.method,
            path=path,
            request=json.loads(result["request_body"]),
            response=json.loads(result["response_body"]),
            extras=SessionRecordChatCompletionsExtras(
                input_ids=TODO,
                output_ids=TODO,
            ),
            status_code=result["status_code"],
        )
        manager.add_record(session_id, record)

        return router._build_proxy_response(result)
