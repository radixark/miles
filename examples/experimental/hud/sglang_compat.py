"""An AsyncOpenAI client that lets stock sglang serve HUD's token-id contract.

HUD's ``OpenAIChatAgent`` (with ``extra_body={"return_token_ids": True}``)
expects each choice to carry ``token_ids`` (output) and ``prompt_token_ids``.
sglang has both, under different names and shapes:

  request:  return_prompt_token_ids=True  -> choice.prompt_token_ids
            return_meta_info=True         -> choice.meta_info.output_token_logprobs
                                             as (logprob, token_id, text) triples
  (an input ``prompt_token_ids`` continuation field is not accepted; sglang
   re-renders the chat template per turn. Verified: the rendered prompt of
   turn k+1 starts with prompt+output of turn k, so exact stitching survives.)

This transport rewrites the request flags on the way out and synthesizes
``choice.token_ids`` on the way back. No sglang changes, no HUD changes.
"""

from __future__ import annotations

import json

import httpx
from openai import AsyncOpenAI


class _SglangTokenIds(httpx.AsyncBaseTransport):
    def __init__(self) -> None:
        self._inner = httpx.AsyncHTTPTransport()

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        is_chat = request.url.path.endswith("/chat/completions")
        if is_chat:
            body = json.loads(request.content or b"{}")
            if body.pop("return_token_ids", None):
                body["return_prompt_token_ids"] = True
                body["return_meta_info"] = True
            # HUD's KV-continuation field; sglang re-renders the template instead.
            body.pop("prompt_token_ids", None)
            content = json.dumps(body).encode()
            headers = {k: v for k, v in request.headers.items() if k.lower() != "content-length"}
            request = httpx.Request(request.method, request.url, headers=headers, content=content)

        response = await self._inner.handle_async_request(request)

        if is_chat and response.status_code == 200:
            data = json.loads(await response.aread())
            for choice in data.get("choices", []):
                triples = (choice.get("meta_info") or {}).get("output_token_logprobs") or []
                if triples and "token_ids" not in choice:
                    choice["token_ids"] = [t[1] for t in triples]
            content = json.dumps(data).encode()
            headers = {
                k: v
                for k, v in response.headers.items()
                if k.lower() not in ("content-length", "content-encoding", "transfer-encoding")
            }
            response = httpx.Response(response.status_code, headers=headers, content=content, request=request)
        return response


def sglang_client(base_url: str, api_key: str = "EMPTY") -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
        http_client=httpx.AsyncClient(transport=_SglangTokenIds(), timeout=600),
    )
