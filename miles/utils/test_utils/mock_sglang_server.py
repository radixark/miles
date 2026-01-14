import asyncio
import random
import re
import socket
import threading
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from transformers import AutoTokenizer

from miles.utils.http_utils import find_available_port


@dataclass(frozen=True)
class ProcessResult:
    text: str
    finish_reason: str


ProcessFn = Callable[[str], ProcessResult]


class MockSGLangServer:
    def __init__(
        self,
        model_name: str,
        process_fn: ProcessFn,
        host: str,
        port: int,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.process_fn = process_fn
        self.host = host
        self.port = port or find_available_port(30000)

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

            input_ids = payload.get("input_ids", [])

            prompt_str = self.tokenizer.decode(input_ids, skip_special_tokens=False)
            process_result = self.process_fn(prompt_str)
            output_ids = self.tokenizer.encode(process_result.text, add_special_tokens=False)

            prompt_tokens = len(input_ids)
            completion_tokens = len(output_ids)

            finish_reason_dict = {"type": process_result.finish_reason}
            if process_result.finish_reason == "length":
                finish_reason_dict["length"] = completion_tokens

            output_token_logprobs = [
                (-1 / 128 * i, token_id)
                for i, token_id in enumerate(output_ids)
            ]

            response = {
                "text": process_result.text,
                "meta_info": {
                    "finish_reason": finish_reason_dict,
                    "prompt_tokens": prompt_tokens,
                    "cached_tokens": 0,
                    "completion_tokens": completion_tokens,
                    "output_token_logprobs": output_token_logprobs,
                },
            }

            return JSONResponse(content=response)

    def start(self):
        config = uvicorn.Config(self.app, host=self.host, port=self.port, log_level="info")
        self.server = uvicorn.Server(config)

        def run_server():
            asyncio.run(self.server.serve())

        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()

        self._wait_for_server_to_start()

    def _wait_for_server_to_start(self):
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


def default_process_fn(prompt: str) -> ProcessResult:
    match = re.search(r"What is 1\+(\d+)\?", prompt)
    if match:
        num = int(match.group(1))
        ans = 1 + num
        return ProcessResult(text=f"It is {ans}.", finish_reason="stop")
    return ProcessResult(text="I don't understand.", finish_reason="stop")


@contextmanager
def with_mock_server(
    model_name: str = "Qwen/Qwen3-0.6B",
    process_fn: ProcessFn = default_process_fn,
    host: str = "127.0.0.1",
    port: int | None = None,
    **kwargs,
):
    server = MockSGLangServer(
        model_name=model_name,
        process_fn=process_fn,
        host=host,
        port=port,
        **kwargs,
    )
    try:
        server.start()
        yield server
    finally:
        server.stop()
