"""Black-box OpenAI streaming client for e2e agents.

Consumes a ``stream: true`` chat completions response as SSE and rebuilds the
non-streaming response shape, so agent code can stay wire-format agnostic.
The accumulator follows the OpenAI delta semantics (fragments concatenate,
tool_calls merge by ``index``) and therefore works for both the session
server's single-chunk fake streaming and a real multi-chunk stream.

The rebuilt assistant message must survive the session server's
``message_matches`` prefix check on the next turn, so tool_call fragments
keep every upstream key except the streaming-only ``index``.
"""

import json


def accumulate_chat_chunks(chunks: list[dict], *, label: str = "") -> dict:
    """Fold ``chat.completion.chunk`` dicts into a non-streaming response."""
    assert chunks, f"{label}: SSE stream contained no chunks before [DONE]"
    message: dict = {"role": "assistant", "content": None}
    tool_calls_by_index: dict[int, dict] = {}
    finish_reason = None
    usage = None
    for chunk in chunks:
        assert chunk.get("object") == "chat.completion.chunk", f"{label}: unexpected SSE payload {chunk!r}"
        [choice] = chunk["choices"]
        delta = choice.get("delta") or {}
        if delta.get("role"):
            message["role"] = delta["role"]
        if delta.get("content") is not None:
            message["content"] = (message["content"] or "") + delta["content"]
        if delta.get("reasoning_content") is not None:
            message["reasoning_content"] = (message.get("reasoning_content") or "") + delta["reasoning_content"]
        for fragment in delta.get("tool_calls") or []:
            entry = tool_calls_by_index.setdefault(fragment["index"], {"function": {}})
            for key, value in fragment.items():
                if key in ("index", "function") or value is None:
                    continue
                entry[key] = value
            function = fragment.get("function") or {}
            if function.get("name") is not None:
                entry["function"]["name"] = function["name"]
            if function.get("arguments") is not None:
                entry["function"]["arguments"] = entry["function"].get("arguments", "") + function["arguments"]
        if choice.get("finish_reason"):
            finish_reason = choice["finish_reason"]
        if chunk.get("usage") is not None:
            usage = chunk["usage"]
    if tool_calls_by_index:
        message["tool_calls"] = [tool_calls_by_index[i] for i in sorted(tool_calls_by_index)]
    first = chunks[0]
    response = {
        "id": first.get("id"),
        "object": "chat.completion",
        "created": first.get("created"),
        "model": first.get("model"),
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
    }
    if usage is not None:
        response["usage"] = usage
    return response


async def stream_chat_completions(client, url: str, payload: dict, *, label: str = "") -> dict:
    """POST *payload* with ``stream: true`` via httpx and rebuild the response."""
    chunks: list[dict] = []
    done = False
    async with client.stream("POST", url, json={**payload, "stream": True}) as resp:
        if resp.status_code != 200:
            body = await resp.aread()
            raise AssertionError(f"{label} failed ({resp.status_code}): {body.decode(errors='replace')}")
        content_type = resp.headers.get("content-type", "")
        assert content_type.startswith(
            "text/event-stream"
        ), f"{label}: expected SSE, got content-type {content_type!r}"
        async for line in resp.aiter_lines():
            if not line or line.startswith(":"):
                continue
            assert line.startswith("data: "), f"{label}: unexpected SSE line {line!r}"
            assert not done, f"{label}: SSE event after [DONE]"
            data = line[len("data: ") :]
            if data == "[DONE]":
                done = True
                continue
            chunks.append(json.loads(data))
    assert done, f"{label}: stream ended without data: [DONE]"
    return accumulate_chat_chunks(chunks, label=label)
