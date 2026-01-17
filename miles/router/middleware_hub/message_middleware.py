from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from .seq_trajectory import SeqTrajectory


class MessageMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, *, router):
        super().__init__(app)
        self.router = router
        self.session_trajectories: dict[str, SeqTrajectory] = {}

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if path.startswith("/sessions/"):
            return await self._dispatch_sessions(request, call_next)
        return await call_next(request)

    async def _dispatch_sessions(self, request: Request, call_next):
        parts = request.url.path.strip("/").split("/")
        if len(parts) == 2 and parts[0] == "sessions" and request.method == "POST":
            return await self._register_session(parts[1])
        if len(parts) == 3 and parts[0] == "sessions" and parts[2] == "retrieve" and request.method == "GET":
            return await self._retrieve_session(parts[1])
        if (
            len(parts) == 5
            and parts[0] == "sessions"
            and parts[2] == "v1"
            and parts[3] == "chat"
            and parts[4] == "completions"
            and request.method == "POST"
        ):
            return await self._session_chat_completions(request, call_next, parts[1])
        return await call_next(request)

    async def _register_session(self, session_id: str):
        self.session_trajectories[session_id] = SeqTrajectory()
        return JSONResponse({"session_id": session_id, "status": "ok"}, status_code=201)

    async def _session_chat_completions(self, request: Request, call_next, session_id: str):
        # TODO: Fill chat completion request with cached token ids and update trajectory from response.
        return JSONResponse({"error": "not implemented", "session_id": session_id}, status_code=501)

    async def _retrieve_session(self, session_id: str):
        if session_id not in self.session_trajectories:
            return JSONResponse({"error": "session not found", "session_id": session_id}, status_code=404)
        trajectory = self.session_trajectories[session_id]
        return JSONResponse(
            {
                "session_id": session_id,
                "token_ids": trajectory.token_ids,
                "log_probs": trajectory.log_probs,
                "loss_mask": trajectory.loss_mask,
            }
        )

    async def _delete_session(self, session_id: str):
        if session_id not in self.session_trajectories:
            return JSONResponse({"error": "session not found", "session_id": session_id}, status_code=404)
        del self.session_trajectories[session_id]
        return JSONResponse({"session_id": session_id, "status": "ok"}, status_code=200)
