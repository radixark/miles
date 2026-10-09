"""Mock agent process and hooks for the full CPU rollout benchmark."""

import os
from contextlib import asynccontextmanager

import httpx

_client = None


async def run_agent(base_url, **kwargs):
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=120)
    response = await _client.post(os.environ["MILES_BENCH_AGENT_URL"] + "/run", json={"base_url": base_url})
    response.raise_for_status()
    return response.json()


async def reward(args, sample, **kwargs):
    return [1.0] * len(sample) if isinstance(sample, list) else 1.0


async def close_client():
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None


def serve_agent(request_bodies, port):
    import uvicorn
    from fastapi import FastAPI, Request

    @asynccontextmanager
    async def lifespan(app):
        async with httpx.AsyncClient(timeout=120) as client:
            app.state.client = client
            yield

    app = FastAPI(lifespan=lifespan)

    @app.post("/run")
    async def run(request: Request):
        base_url = (await request.json())["base_url"]
        for body in request_bodies:
            response = await app.state.client.post(
                base_url + "/v1/chat/completions", content=body, headers={"content-type": "application/json"}
            )
            response.raise_for_status()
        return {"agent_metrics": {"turns_ok": len(request_bodies)}}

    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


def serve_backend(response_bodies, port):
    import uvicorn
    from fastapi import FastAPI, Request
    from starlette.responses import Response

    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        payload = await request.json()
        turn = (len(payload["messages"]) - 1) // 2
        return Response(response_bodies[turn], media_type="application/json")

    @app.get("/list_workers")
    async def list_workers():
        return {"urls": [f"http://127.0.0.1:{port}"]}

    @app.post("/abort_request")
    async def abort():
        return {"status": "ok"}

    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
