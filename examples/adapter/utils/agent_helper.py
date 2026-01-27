import asyncio
import logging
import time
from typing import Any

import httpx
from eval_protocol import InitRequest
from sglang.srt.entrypoints.openai.protocol import ChatCompletionResponse


logger = logging.getLogger(__name__)
_HTTP_CLIENT: httpx.AsyncClient | None = None
_HTTP_BASE_URL: str | None = None
_HTTP_TIMEOUT_SECONDS = 60.0
_HTTP_MAX_RETRIES = 60
_HTTP_RETRY_INTERVAL_SECONDS = 1.0
_HTTP_CLIENT_LOCK = asyncio.Lock()
HTTP_CLIENT_CONCURRENCY = 2048


async def _get_http_client(base_url: str) -> httpx.AsyncClient:
    global _HTTP_CLIENT, _HTTP_BASE_URL
    async with _HTTP_CLIENT_LOCK:
        if _HTTP_CLIENT is None or _HTTP_BASE_URL != base_url:
            if _HTTP_CLIENT is not None:
                await _HTTP_CLIENT.aclose()
            _HTTP_BASE_URL = base_url
            _HTTP_CLIENT = httpx.AsyncClient(
                limits=httpx.Limits(max_connections=HTTP_CLIENT_CONCURRENCY),
                timeout=httpx.Timeout(None),
            )
        return _HTTP_CLIENT


async def call_llm(request: InitRequest, messages: list[dict[str, Any]]) -> ChatCompletionResponse:
    base_url = request.model_base_url
    completion_params = dict(request.completion_params or {})

    payload = {
        "model": "default",
        "messages": messages,  # notice: we do not use request.messages here
        "tools": request.tools,
        **completion_params,
    }

    url = f"{base_url}/v1/chat/completions"
    client = await _get_http_client(url)
    response = None
    for attempt in range(1, _HTTP_MAX_RETRIES + 1):
        try:
            http_response = await client.post(url, json=payload)
            http_response.raise_for_status()
            response = http_response.json()
            break
        except httpx.RequestError:
            if attempt == _HTTP_MAX_RETRIES:
                raise
            logger.warning("OpenAI API error, retrying... (attempt %s/%s)", attempt, _HTTP_MAX_RETRIES)
            time.sleep(_HTTP_RETRY_INTERVAL_SECONDS)
    response = ChatCompletionResponse.model_validate(response)
    return response
