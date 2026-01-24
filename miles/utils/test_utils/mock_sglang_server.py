import asyncio
import re
import time
import uuid
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import asdict, dataclass

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import TypeAdapter
from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from transformers import AutoTokenizer

from miles.utils.http_utils import find_available_port
from miles.utils.test_utils.uvicorn_thread_server import UvicornThreadServer


@dataclass(frozen=True)
class ProcessResultMetaInfo:
    weight_version: str | None = None
    routed_experts: str | None = None
    spec_accept_token_num: int | None = None
    spec_draft_token_num: int | None = None
    spec_verify_ct: int | None = None

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass(frozen=True)
class ProcessResult:
    text: str
    finish_reason: str = "stop"
    cached_tokens: int = 0
    meta_info: ProcessResultMetaInfo = ProcessResultMetaInfo()


ProcessFn = Callable[[str], ProcessResult]


class MockSGLangServer:
    def __init__(
        self,
        model_name: str,
        process_fn: ProcessFn,
        host: str,
        port: int,
        latency: float = 0.0,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.process_fn = process_fn
        self.host = host
        self.port = port or find_available_port(30000)
        self.latency = latency

        self.app = FastAPI()
        self._server: UvicornThreadServer | None = None

        self.request_log: list[dict] = []
        self._concurrency = Counter()

        self._setup_routes()

    @property
    def max_concurrent(self) -> int:
        return self._concurrency.max_value

    def reset_stats(self):
        self.request_log.clear()
        self._concurrency.reset()

    def start(self):
        self._server = UvicornThreadServer(self.app, host=self.host, port=self.port)
        self._server.start()

    def stop(self):
        if self._server is not None:
            self._server.stop()

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def _setup_routes(self):
        @self.app.post("/generate")
        async def generate(request: Request):
            return await self._handle_generate_like_request(request, self._compute_generate_response)

        @self.app.post("/v1/chat/completions")
        async def chat_completions(request: Request):
            return await self._handle_generate_like_request(request, self._compute_chat_completions_response)

        @self.app.get("/health")
        async def health():
            return JSONResponse(content={"status": "ok"})

        @self.app.post("/abort_request")
        async def abort_request(_request: Request):
            return JSONResponse(content={"status": "ok"})

    async def _handle_generate_like_request(self, request: Request, compute_fn: Callable[[dict], dict]):
        payload = await request.json()
        self.request_log.append(payload)
        with self._concurrency.track():
            if self.latency > 0:
                await asyncio.sleep(self.latency)
            response = compute_fn(payload)
        return JSONResponse(content=response)

    def _compute_generate_response(self, payload: dict) -> dict:
        assert payload.get("return_logprob", True) is True, "MockSGLangServer requires return_logprob=True"
        input_ids = payload.get("input_ids", [])

        prompt_str = self.tokenizer.decode(input_ids, skip_special_tokens=False)
        process_result = self.process_fn(prompt_str)
        output_ids = self.tokenizer.encode(process_result.text, add_special_tokens=False)

        prompt_tokens = len(input_ids)
        completion_tokens = len(output_ids)

        finish_reason_dict = {"type": process_result.finish_reason}
        if process_result.finish_reason == "length":
            finish_reason_dict["length"] = completion_tokens

        output_token_logprobs = [(-1 / 128 * i, token_id) for i, token_id in enumerate(output_ids)]

        meta_info = {
            "finish_reason": finish_reason_dict,
            "prompt_tokens": prompt_tokens,
            "cached_tokens": process_result.cached_tokens,
            "completion_tokens": completion_tokens,
            "output_token_logprobs": output_token_logprobs,
            **process_result.meta_info.to_dict(),
        }

        return {"text": process_result.text, "meta_info": meta_info}

    def _compute_chat_completions_response(self, payload: dict) -> dict:
        messages = payload.get("messages", [])
        tools = payload.get("tools")

        prompt_str = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, tools=tools
        )

        process_result = self.process_fn(prompt_str)
        output_ids = self.tokenizer.encode(process_result.text, add_special_tokens=False)

        logprobs_content = [
            {"token": self.tokenizer.convert_ids_to_tokens(tid), "logprob": -1 / 128 * i}
            for i, tid in enumerate(output_ids)
        ]

        finish_reason = process_result.finish_reason
        tool_calls = None
        if tools and finish_reason == "stop":
            parser = FunctionCallParser(
                tools=TypeAdapter(list[Tool]).validate_python(tools),
                tool_call_parser="qwen25",
            )
            message_content, parsed_calls = parser.parse_non_stream(process_result.text)
            if parsed_calls:
                finish_reason = "tool_calls"
                tool_calls = [
                    {
                        "id": f"call{i:05d}",
                        "type": "function",
                        "function": {"name": call.name, "arguments": call.parameters or "{}"},
                    }
                    for i, call in enumerate(parsed_calls)
                ]
        else:
            message_content = process_result.text

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "mock-model",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": message_content,
                        "tool_calls": tool_calls,
                    },
                    "logprobs": {"content": logprobs_content},
                    "finish_reason": finish_reason,
                }
            ],
        }


class Counter:
    def __init__(self):
        self._current = 0
        self._max = 0

    @property
    def max_value(self) -> int:
        return self._max

    def reset(self):
        self._current = 0
        self._max = 0

    @contextmanager
    def track(self):
        self._current += 1
        self._max = max(self._max, self._current)
        try:
            yield
        finally:
            self._current -= 1


def default_process_fn(prompt: str) -> ProcessResult:
    match = re.search(r"What is 1\+(\d+)\?", prompt)
    if match:
        num = int(match.group(1))
        ans = 1 + num
        return ProcessResult(text=f"\\boxed{{{ans}}}", finish_reason="stop")
    return ProcessResult(text="I don't understand.", finish_reason="stop")


@contextmanager
def with_mock_server(
    model_name: str = "Qwen/Qwen3-0.6B",
    process_fn: ProcessFn = default_process_fn,
    host: str = "127.0.0.1",
    port: int | None = None,
    latency: float = 0.0,
):
    server = MockSGLangServer(
        model_name=model_name,
        process_fn=process_fn,
        host=host,
        port=port,
        latency=latency,
    )
    try:
        server.start()
        yield server
    finally:
        server.stop()
