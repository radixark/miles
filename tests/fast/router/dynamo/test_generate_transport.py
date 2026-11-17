import asyncio
import json

import httpx
import pytest

from miles.router.dynamo.api_client import GenerateProtocolError, _generate_events
from miles.utils import http_utils

pytestmark = pytest.mark.asyncio


def _event(*, terminal=False):
    return {"text": "你好", "output_ids": [12], "meta_info": {"finish_reason": {"type": "stop"} if terminal else None}}


def _wire(value):
    return b"data: " + json.dumps(value, ensure_ascii=False).encode() + b"\n\n"


async def _collect(chunks, *, max_event_bytes=8 * 1024 * 1024):
    async def source():
        for chunk in chunks:
            yield chunk

    return [event async for event in _generate_events(source(), max_event_bytes=max_event_bytes)]


@pytest.mark.parametrize("split", [1, 2, 7, 4096])
async def test_chunk_boundaries(split):
    wire = b": heartbeat\r\n\r\n" + _wire(_event()) + _wire(_event(terminal=True)) + b"data: [DONE]\n\n"
    assert await _collect([wire[i : i + split] for i in range(0, len(wire), split)]) == [
        _event(),
        _event(terminal=True),
    ]


async def test_multiline_data_and_cr_delimiters():
    wire = b'event: message\rdata: {"meta_info":\rdata: {"finish_reason":{"type":"stop"}}}\r\rdata: [DONE]\r\r'
    assert len(await _collect([wire, b"\n"])) == 1
    assert len(await _collect([wire])) == 1


@pytest.mark.parametrize(
    "wire",
    [
        b"data: [DONE]\n\n",
        b"data: broken\n\n",
        b"data: []\n\n",
        b'data: {"error":"bad"}\n\n',
        b"event: error\ndata: rejected\n\n",
        b"data: \xff\n\n",
        _wire(_event()),
        _wire(_event(terminal=True)),
        _wire(_event(terminal=True)) + _wire(_event()),
        b'data: {"meta_info":{}}',
    ],
)
async def test_protocol_failures(wire):
    with pytest.raises(GenerateProtocolError):
        await _collect([wire])


async def test_bounded_event():
    for wire in [b"data: " + b"x" * 100, b"data:x\n" * 100]:
        with pytest.raises(GenerateProtocolError, match="exceeds"):
            await _collect([wire], max_event_bytes=32)


class _EventStream(httpx.AsyncByteStream):
    def __init__(self, chunks):
        self.chunks = chunks
        self.closed = False

    async def __aiter__(self):
        for chunk in self.chunks:
            yield chunk

    async def aclose(self):
        self.closed = True


async def _decode_events(response):
    return [event async for event in _generate_events(response.aiter_bytes(), max_event_bytes=1024)]


async def test_retry_discards_partial_generation(monkeypatch):
    first = _EventStream([_wire(_event())])
    second = _EventStream([_wire(_event(terminal=True)), b"data: [DONE]\n\n"])
    streams = iter([first, second])
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(200, stream=next(streams))

    async def no_wait(_):
        pass

    monkeypatch.setattr(http_utils.asyncio, "sleep", no_wait)
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        result = await http_utils._post(client, "http://engine/generate", {}, response_decoder=_decode_events)

    assert result == [_event(terminal=True)]
    assert len(requests) == 2
    assert first.closed and second.closed


@pytest.mark.parametrize("stage", ["headers", "body"])
async def test_attempt_timeout_covers_headers_and_body(stage):
    started = asyncio.Event()
    stream = _EventStream([])

    async def wait_forever():
        started.set()
        await asyncio.Event().wait()

    async def respond(request):
        if stage == "headers":
            await wait_forever()
        return httpx.Response(200, stream=stream)

    async def decode(response):
        await wait_forever()

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:

        async def run_attempt():
            with pytest.raises(TimeoutError):
                await http_utils._post(
                    client,
                    "http://engine/generate",
                    {},
                    max_retries=1,
                    response_decoder=decode,
                    response_timeout=0.02,
                )

        await asyncio.wait_for(run_attempt(), timeout=1)
    assert started.is_set()
    if stage == "body":
        assert stream.closed


async def test_cancellation_closes_stream_without_retry():
    started = asyncio.Event()
    stream = _EventStream([])
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(200, stream=stream)

    async def decode(response):
        started.set()
        await asyncio.Event().wait()

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        task = asyncio.create_task(http_utils._post(client, "http://engine/generate", {}, response_decoder=decode))
        await asyncio.wait_for(started.wait(), timeout=1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert len(requests) == 1
    assert stream.closed
