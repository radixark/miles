import json
import time
import uuid
from dataclasses import dataclass, asdict
from typing import TYPE_CHECKING

from fastapi import Request
from fastapi.responses import JSONResponse

if TYPE_CHECKING:
    from miles.router.router import MilesRouter


@dataclass
class SessionRecord:
    timestamp: float
    method: str
    path: str
    request_json: dict | None
    response_json: dict | None
    status_code: int


class SessionManager:
    def __init__(self):
        self.sessions: dict[str, list[SessionRecord]] = {}

    def create_session(self) -> str:
        session_id = uuid.uuid4().hex
        self.sessions[session_id] = []
        return session_id

    def get_session(self, session_id: str) -> list[SessionRecord] | None:
        return self.sessions.get(session_id)

    def delete_session(self, session_id: str) -> list[SessionRecord] | None:
        return self.sessions.pop(session_id, None)

    def add_record(self, session_id: str, record: SessionRecord):
        if session_id not in self.sessions:
            raise KeyError(f"session not found: {session_id}")
        self.sessions[session_id].append(record)


def setup_session_routes(app, router: "MilesRouter"):
    manager = SessionManager()

    @app.post("/sessions")
    async def create_session():
        session_id = manager.create_session()
        return {"session_id": session_id}

    @app.get("/sessions/{session_id}")
    async def get_session(session_id: str):
        records = manager.get_session(session_id)
        if records is None:
            return JSONResponse(status_code=404, content={"error": "session not found"})
        return {"session_id": session_id, "records": [asdict(r) for r in records]}

    @app.delete("/sessions/{session_id}")
    async def delete_session(session_id: str):
        records = manager.delete_session(session_id)
        if records is None:
            return JSONResponse(status_code=404, content={"error": "session not found"})
        return {"session_id": session_id, "records": [asdict(r) for r in records]}

    @app.api_route("/sessions/{session_id}/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
    async def session_proxy(request: Request, session_id: str, path: str):
        if session_id not in manager.sessions:
            return JSONResponse(status_code=404, content={"error": "session not found"})

        result = await router._do_proxy(request, path)

        request_json = None
        if result["request_body"]:
            try:
                request_json = json.loads(result["request_body"])
            except Exception:
                pass

        response_json = None
        try:
            response_json = json.loads(result["response_body"])
        except Exception:
            pass

        record = SessionRecord(
            timestamp=time.time(),
            method=request.method,
            path=path,
            request_json=request_json,
            response_json=response_json,
            status_code=result["status_code"],
        )
        manager.add_record(session_id, record)

        return router._build_response(result)
