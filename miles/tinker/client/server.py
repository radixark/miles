import asyncio
import socket
import uuid
from contextlib import asynccontextmanager, contextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from pydantic import ValidationError

from miles.tinker.client.rendering import ChatRequest
from miles.tinker.client.session import ChatSession


class _EmbeddedServer(uvicorn.Server):
    @contextmanager
    def capture_signals(self):
        # The cookbook process owns signals; several rollout servers may coexist.
        yield


class SessionServer:
    """An OAI adapter owned by the cookbook rollout process."""

    def __init__(self):
        self.sessions: dict[str, ChatSession] = {}
        self.app = FastAPI()
        self.app.post("/sessions/{session_id}/v1/chat/completions")(self.complete)

    @asynccontextmanager
    async def session(self, session: ChatSession):
        session_id = uuid.uuid4().hex
        self.sessions[session_id] = session
        try:
            yield f"/sessions/{session_id}"
        finally:
            del self.sessions[session_id]
            async with session.lock:
                session.closed = True

    async def complete(self, session_id: str, request: Request):
        # An unguessable URL grants access to exactly one trial's policy.
        session = self.sessions.get(session_id)
        if session is None:
            raise HTTPException(404, "unknown session")
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > 16 * 1024 * 1024:
                raise HTTPException(413, "request body too large")
        try:
            chat_request = ChatRequest.model_validate_json(body)
            return await session.complete(chat_request)
        except ValidationError as error:
            raise HTTPException(400, str(error)) from error
        except ValueError as error:
            raise HTTPException(400, str(error)) from error

    @asynccontextmanager
    async def serve(self, host: str = "127.0.0.1"):
        # Bind first so parallel rollout groups each get their own race-free port.
        with socket.socket() as sock:
            sock.bind((host, 0))
            sock.listen()
            server = _EmbeddedServer(
                uvicorn.Config(self.app, log_level="warning", lifespan="off", timeout_graceful_shutdown=30)
            )
            task = asyncio.create_task(server.serve(sockets=[sock]))
            try:
                while not server.started:
                    if task.done():
                        await task
                        raise RuntimeError("session server exited before startup")
                    await asyncio.sleep(0.01)
                yield sock.getsockname()[1]
            finally:
                server.should_exit = True
                await task
