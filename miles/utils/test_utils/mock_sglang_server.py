import asyncio
import base64
import random
import socket
import threading
import time
from collections.abc import Callable
from contextlib import asynccontextmanager, contextmanager
from typing import Any

import numpy as np
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from miles.utils.http_utils import find_available_port


class MockSGLangServer:
    def __init__(
        self,
        tokenizer,
        process_fn: Callable[[str], str] | None = None,
        host: str = "127.0.0.1",
        port: int | None = None,
        finish_reason: str = "stop",
        cached_tokens: int = 0,
        weight_version: str | None = None,
        num_layers: int = 32,
        moe_router_topk: int = 2,
        num_experts: int = 8,
    ):
        self.tokenizer = tokenizer
        self.process_fn = process_fn or (lambda x: "This is a mock response.")
        self.host = host
        self.port = port or find_available_port(30000)
        self.finish_reason = finish_reason
        self.cached_tokens = cached_tokens
        self.weight_version = weight_version
        self.num_layers = num_layers
        self.moe_router_topk = moe_router_topk
        self.num_experts = num_experts

        self.requests: list[dict[str, Any]] = []
        self.app = FastAPI()
        self.server: uvicorn.Server | None = None
        self.server_thread: threading.Thread | None = None

        self._setup_routes()

    def _setup_routes(self):
        @self.app.post("/generate")
        async def generate(request: Request):
            payload = await request.json()
            self.requests.append(payload)

            return_logprob = payload.get("return_logprob", False)
            return_routed_experts = payload.get("return_routed_experts", False)
            input_ids = payload.get("input_ids", [])

            prompt_str = self.tokenizer.decode(input_ids, skip_special_tokens=False)
            response_str = self.process_fn(prompt_str)
            output_ids = self.tokenizer.encode(response_str, add_special_tokens=False)

            prompt_tokens = len(input_ids)
            completion_tokens = len(output_ids)

            response = {
                "text": response_str,
                "meta_info": {
                    "finish_reason": {"type": self.finish_reason},
                    "prompt_tokens": prompt_tokens,
                    "cached_tokens": min(self.cached_tokens, prompt_tokens),
                    "completion_tokens": completion_tokens,
                },
            }

            if self.finish_reason == "length":
                response["meta_info"]["finish_reason"]["length"] = completion_tokens

            if return_logprob:
                output_token_logprobs = [
                    (random.uniform(-10.0, -0.1), token_id) for token_id in output_ids
                ]
                response["meta_info"]["output_token_logprobs"] = output_token_logprobs

            if return_routed_experts:
                total_tokens = prompt_tokens + completion_tokens
                num_tokens_for_routing = max(1, total_tokens - 1)
                routed_experts_array = np.random.randint(
                    0, self.num_experts,
                    size=(num_tokens_for_routing, self.num_layers, self.moe_router_topk),
                    dtype=np.int32,
                )
                routed_experts_b64 = base64.b64encode(routed_experts_array.tobytes()).decode("ascii")
                response["meta_info"]["routed_experts"] = routed_experts_b64

            if self.weight_version is not None:
                response["meta_info"]["weight_version"] = self.weight_version

            return JSONResponse(content=response)

    def start(self):
        config = uvicorn.Config(self.app, host=self.host, port=self.port, log_level="error")
        self.server = uvicorn.Server(config)

        def run_server():
            asyncio.run(self.server.serve())

        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()

        for _ in range(50):
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                result = sock.connect_ex((self.host, self.port))
                sock.close()
                if result == 0:
                    break
            except Exception:
                pass
            time.sleep(0.1)
        else:
            raise RuntimeError(f"Failed to start server on {self.host}:{self.port}")

    def stop(self):
        if self.server:
            self.server.should_exit = True
        if self.server_thread and self.server_thread.is_alive():
            self.server_thread.join(timeout=2.0)

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def clear_requests(self):
        self.requests.clear()


@contextmanager
def start_mock_server(
    tokenizer,
    process_fn: Callable[[str], str] | None = None,
    host: str = "127.0.0.1",
    port: int | None = None,
    finish_reason: str = "stop",
    **kwargs,
):
    server = MockSGLangServer(
        tokenizer=tokenizer,
        process_fn=process_fn,
        host=host,
        port=port,
        finish_reason=finish_reason,
        **kwargs,
    )
    try:
        server.start()
        yield server
    finally:
        server.stop()


@asynccontextmanager
async def start_mock_server_async(
    tokenizer,
    process_fn: Callable[[str], str] | None = None,
    host: str = "127.0.0.1",
    port: int | None = None,
    finish_reason: str = "stop",
    **kwargs,
):
    server = MockSGLangServer(
        tokenizer=tokenizer,
        process_fn=process_fn,
        host=host,
        port=port,
        finish_reason=finish_reason,
        **kwargs,
    )
    try:
        server.start()
        yield server
    finally:
        server.stop()
