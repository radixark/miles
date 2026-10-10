import asyncio
import time
import uuid
from dataclasses import dataclass, field

import tinker
from tinker_cookbook.completers import TinkerTokenCompleter
from tinker_cookbook.renderers import Renderer

from miles.tinker.client.rendering import ChatRequest, parse_completion, render_prompt, sampling_params
from miles.tinker.core.token_trace import TokenTrace


@dataclass
class ChatSession:
    policy: TinkerTokenCompleter
    renderer: Renderer
    max_datum_tokens: int
    max_turns: int = 512
    trace: TokenTrace = field(default_factory=TokenTrace)
    closed: bool = False
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def complete(self, request: ChatRequest) -> dict:
        async with self.lock:
            if self.closed:
                raise ValueError("session is closed")
            if len(self.trace.turns) >= self.max_turns:
                raise ValueError("session turn limit reached")
            input_ids = await asyncio.to_thread(render_prompt, self.renderer, request)
            params = sampling_params(self.renderer, request)
            params["max_tokens"] = min(params.get("max_tokens", self.policy.max_tokens), self.policy.max_tokens)
            params.setdefault("temperature", self.policy.temperature)
            if self.policy.context_window is not None:
                params["max_tokens"] = min(params["max_tokens"], self.policy.context_window - len(input_ids))
            if params["max_tokens"] <= 0 or len(input_ids) + params["max_tokens"] - 1 > self.max_datum_tokens:
                raise ValueError("prompt plus completion exceeds the per-datum token budget")
            response = await self.policy.sampling_client.sample_async(
                prompt=tinker.ModelInput.from_ints(input_ids),
                num_samples=1,
                sampling_params=tinker.SamplingParams(**params),
            )
            # Parsing is a display operation; its failure must not erase a completed sample.
            turn = self.trace.record(f"chatcmpl-{uuid.uuid4().hex}", input_ids, response.sequences[0].model_dump())
            message = parse_completion(self.renderer, turn)
            return {
                "id": turn.id,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": request.model or "tinker",
                "choices": [{
                    "index": 0,
                    "message": message,
                    "finish_reason": "tool_calls" if turn.stop_reason == "stop" and message.get("tool_calls") else turn.stop_reason,
                }],
                "usage": {
                    "prompt_tokens": len(turn.input_ids),
                    "completion_tokens": len(turn.output_ids),
                    "total_tokens": len(turn.input_ids) + len(turn.output_ids),
                },
            }
