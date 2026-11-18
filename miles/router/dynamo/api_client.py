import json
import re
from collections.abc import AsyncIterable, AsyncIterator
from typing import Any

import httpx

from miles.utils.http_utils import ResponseDecodeError, post

_NEWLINE = re.compile(b"\r\n|\r|\n")


class GenerateProtocolError(ResponseDecodeError):
    pass


class RetryableGenerateError(GenerateProtocolError):
    retryable = True


def _raise_generate_error(error: Any, *, status_code=None) -> None:
    retryable = None
    if isinstance(error, dict):
        retryable = error.get("retryable")
        status_code = error.get("code", error.get("status_code", status_code))
    if retryable is None:
        retryable = isinstance(status_code, int) and (status_code in (408, 429) or 500 <= status_code < 600)
    # Unknown error events fail fast; only explicit producer signals permit retry.
    error_type = RetryableGenerateError if retryable is True else GenerateProtocolError
    detail = json.dumps(error, ensure_ascii=False) if isinstance(error, dict) else str(error)
    raise error_type(f"Dynamo generation failed: {detail}")


async def _generate_events(chunks: AsyncIterable[bytes], *, max_event_bytes: int) -> AsyncIterator[dict[str, Any]]:
    terminal = False
    async for event, data in _sse_events(chunks, max_event_bytes=max_event_bytes):
        if event == "error":
            try:
                error = json.loads(data)
            except ValueError:
                error = data
            _raise_generate_error(error.get("error", error) if isinstance(error, dict) else error)
        if data == "[DONE]":
            if not terminal:
                raise GenerateProtocolError("stream ended without a terminal response")
            return
        if terminal:
            raise GenerateProtocolError("received data after a terminal response")
        try:
            value = json.loads(data)
        except ValueError as error:
            raise GenerateProtocolError("invalid SSE JSON") from error
        if not isinstance(value, dict):
            raise GenerateProtocolError("expected a native generation response")
        if "error" in value:
            _raise_generate_error(value["error"])
        meta = value.get("meta_info")
        if not isinstance(meta, dict):
            raise GenerateProtocolError("response is missing meta_info")
        finish = meta.get("finish_reason")
        if isinstance(finish, dict) and isinstance(status := finish.get("status_code"), int) and status >= 400:
            _raise_generate_error(finish, status_code=status)
        terminal = meta.get("finish_reason") is not None
        yield value
    raise RetryableGenerateError("stream closed before [DONE]")


async def _sse_events(chunks: AsyncIterable[bytes], *, max_event_bytes: int) -> AsyncIterator[tuple[str, str]]:
    buffer = b""
    data: list[bytes] = []
    event = b""
    size = 0
    async for chunk in chunks:
        buffer += chunk
        while (match := _NEWLINE.search(buffer)) is not None:
            # A CR at the chunk boundary may be the first half of CRLF.
            if match[0] == b"\r" and match.end() == len(buffer):
                break
            line, buffer = buffer[: match.start()], buffer[match.end() :]
            size += len(line) + len(match[0])
            if size > max_event_bytes:
                raise GenerateProtocolError("SSE event exceeds max_event_bytes")
            if not line:
                if data:
                    try:
                        yield event.decode("utf-8"), b"\n".join(data).decode("utf-8")
                    except UnicodeDecodeError as error:
                        raise GenerateProtocolError("invalid SSE UTF-8") from error
                data, event, size = [], b"", 0
            else:
                field, separator, value = line.partition(b":")
                value = value.removeprefix(b" ") if separator else b""
                if field == b"data":
                    data.append(value)
                elif field == b"event":
                    event = value
        if size + len(buffer) > max_event_bytes:
            raise GenerateProtocolError("SSE event exceeds max_event_bytes")
    if buffer == b"\r":
        if data:
            try:
                yield event.decode("utf-8"), b"\n".join(data).decode("utf-8")
            except UnicodeDecodeError as error:
                raise GenerateProtocolError("invalid SSE UTF-8") from error
        buffer, data = b"", []
    if buffer or data:
        raise RetryableGenerateError("incomplete SSE event")


async def post_generate(args, url: str, payload: dict, *, headers=None) -> dict:
    config = args.dynamo_config
    if config is None:
        return await post(url, payload, headers=headers)
    # Caller headers authenticate the frontend hop; each sidecar owns its engine credentials.
    # Native response assembly requires streaming even when the rollout caller requests a complete result.
    return await post(
        url,
        {**payload, "stream": True},
        headers=headers,
        response_decoder=decode_response,
        response_timeout=config.request_timeout_seconds,
    )


async def decode_response(response: httpx.Response) -> dict:
    if response.headers.get("content-type", "").split(";")[0].strip().lower() != "text/event-stream":
        raise GenerateProtocolError("expected text/event-stream")
    texts, ids = [], []
    arrays = {}
    request_id = None
    final = None
    async for event in _generate_events(response.aiter_bytes(), max_event_bytes=8 * 1024 * 1024):
        meta = event["meta_info"]
        if request_id is not None and meta.get("id") != request_id:
            raise GenerateProtocolError("request identity changed within the stream")
        request_id = meta.get("id")
        if not request_id or not isinstance(event.get("text"), str) or not isinstance(event.get("output_ids"), list):
            raise GenerateProtocolError("missing native response identity, text or tokens")
        chunk_ids = event["output_ids"]
        if (meta.get("finish_reason") or {}).get("type") == "abort":
            # sglang-miles abort echoes cumulative text/logprobs, even in incremental mode.
            pairs = meta.get("output_token_logprobs", [])
            if len(pairs) != meta.get("completion_tokens"):
                raise GenerateProtocolError("aborted response has incomplete token logprobs")
            ids = [pair[1] for pair in pairs]
            texts = [event["text"]]
            arrays = {}
            final = event
            continue
        pairs = meta.get("output_token_logprobs")
        if pairs is not None and (len(pairs) != len(chunk_ids) or [pair[1] for pair in pairs] != chunk_ids):
            raise GenerateProtocolError("incremental tokens and logprobs disagree")
        ids.extend(chunk_ids)
        texts.append(event["text"])
        if meta.get("completion_tokens") != len(ids):
            raise GenerateProtocolError("expected incremental token output")
        for name in (
            "output_token_logprobs",
            "output_top_logprobs",
            "output_token_ids_logprobs",
            "output_token_sampling_mask",
            "output_token_sampling_logprobs",
        ):
            if meta.get(name) is not None:
                arrays.setdefault(name, []).extend(meta[name])
        final = event
    if final is None:
        raise GenerateProtocolError("empty generation response")
    # Version spans and usage are cumulative; token-aligned arrays and text are deltas.
    return {**final, "text": "".join(texts), "output_ids": ids, "meta_info": {**final["meta_info"], **arrays}}
