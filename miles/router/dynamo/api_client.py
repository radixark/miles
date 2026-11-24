import json
import re
from collections.abc import AsyncIterable, AsyncIterator
from typing import Any

from miles.utils.http_utils import ResponseDecodeError

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
