import asyncio
import json
import traceback
from types import SimpleNamespace

import httpx
import pytest
from ray.exceptions import RayTaskError

from miles.router.dynamo.api_client import GenerateProtocolError, RetryableGenerateError, _generate_events
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


async def _request_events(client, monkeypatch, *, distributed, max_retries=3):
    if not distributed:
        return await http_utils._post(
            client, "http://engine/generate", {}, max_retries=max_retries, response_decoder=_decode_events
        )

    async def remote(*args, **kwargs):
        try:
            return await http_utils._post(client, *args, **kwargs)
        except Exception as error:
            raise RayTaskError("do_post", traceback.format_exc(), error).as_instanceof_cause() from error

    def no_fallback():
        pytest.fail("a decoder failure must not replay generation through the local fallback")

    monkeypatch.setattr(http_utils, "_distributed_post_enabled", True)
    monkeypatch.setattr(http_utils, "_post_actors", [SimpleNamespace(do_post=SimpleNamespace(remote=remote))])
    monkeypatch.setattr(http_utils, "_rollout_client", no_fallback)
    return await http_utils.post(
        "http://engine/generate", {}, max_retries=max_retries, response_decoder=_decode_events
    )


@pytest.mark.parametrize("distributed", [False, True])
@pytest.mark.parametrize(
    "body,message",
    [
        (_wire({"error": {"message": "input too long", "code": 400, "retryable": False}}), "input too long"),
        (
            _wire({"error": {"message": "permanent backend failure", "code": 503, "retryable": False}}),
            "permanent backend failure",
        ),
        (b'event: error\ndata: {"error":{"message":"invalid prompt","code":400}}\n\n', "invalid prompt"),
        (b"event: error\ndata: unknown engine failure\n\n", "unknown engine failure"),
        (b"data: broken\n\n", "invalid SSE JSON"),
        (b"data: [DONE]\n\n", "without a terminal response"),
        (
            _wire(
                {"meta_info": {"finish_reason": {"type": "abort", "status_code": 400, "message": "invalid sampling"}}}
            ),
            "invalid sampling",
        ),
    ],
)
async def test_permanent_stream_errors_fail_once_with_original_reason(monkeypatch, distributed, body, message):
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(200, stream=_EventStream([body]))

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        with pytest.raises(GenerateProtocolError, match=message):
            await _request_events(client, monkeypatch, distributed=distributed)
    assert len(requests) == 1


@pytest.mark.parametrize("distributed", [False, True])
@pytest.mark.parametrize(
    "error",
    [
        {"message": "worker unavailable", "code": 503},
        {"message": "overloaded", "code": 429},
        {"message": "request timed out", "code": 408},
        {"message": "explicit retry", "retryable": True},
    ],
)
async def test_explicit_transient_stream_errors_retry(monkeypatch, distributed, error):
    requests = []

    def respond(request):
        requests.append(request)
        body = _wire({"error": error}) if len(requests) == 1 else _wire(_event(terminal=True)) + b"data: [DONE]\n\n"
        return httpx.Response(200, stream=_EventStream([body]))

    async def no_wait(_):
        pass

    monkeypatch.setattr(http_utils.asyncio, "sleep", no_wait)
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        assert await _request_events(client, monkeypatch, distributed=distributed) == [_event(terminal=True)]
    assert len(requests) == 2


async def test_exhausted_decoder_retries_do_not_restart_locally(monkeypatch):
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(200, stream=_EventStream([_wire(_event())]))

    async def no_wait(_):
        pass

    monkeypatch.setattr(http_utils.asyncio, "sleep", no_wait)
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        with pytest.raises(RetryableGenerateError, match="before \\[DONE\\]"):
            await _request_events(client, monkeypatch, distributed=True)
    assert len(requests) == 3


@pytest.mark.parametrize("failure", [TimeoutError, httpx.ReadTimeout, httpx.ReadError])
async def test_exhausted_stream_transport_retries_do_not_restart_locally(monkeypatch, failure):
    requests = []

    def respond(request):
        requests.append(request)
        raise failure("stream transport stalled")

    async def no_wait(_):
        pass

    monkeypatch.setattr(http_utils.asyncio, "sleep", no_wait)
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        with pytest.raises(failure, match="stream transport stalled"):
            await _request_events(client, monkeypatch, distributed=True)
    assert len(requests) == 3


async def test_actor_outage_still_falls_back_to_local_transport(monkeypatch):
    async def unavailable(*args, **kwargs):
        raise RuntimeError("poster actor unavailable")

    monkeypatch.setattr(http_utils, "_distributed_post_enabled", True)
    monkeypatch.setattr(http_utils, "_post_actors", [SimpleNamespace(do_post=SimpleNamespace(remote=unavailable))])
    body = _wire(_event(terminal=True)) + b"data: [DONE]\n\n"
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, stream=_EventStream([body])))
    ) as client:
        monkeypatch.setattr(http_utils, "_rollout_client", lambda: client)
        assert await http_utils.post("http://engine/generate", {}, response_decoder=_decode_events) == [
            _event(terminal=True)
        ]


@pytest.mark.parametrize("action", ["get", "delete", "put"])
async def test_streaming_rejects_non_post_before_sending(action):
    def refuse_request(request):
        pytest.fail("invalid streaming methods must be rejected before sending")

    async with httpx.AsyncClient(transport=httpx.MockTransport(refuse_request)) as client:
        with pytest.raises(http_utils.ResponseDecodeError, match="only action='post'"):
            await http_utils._post(
                client, "http://engine/generate", {}, action=action, response_decoder=_decode_events
            )


async def test_streamed_redirect_preserves_http_error_body():
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(302, stream=_EventStream([b"wrong endpoint"])))
    ) as client:
        with pytest.raises(httpx.HTTPStatusError) as error:
            await http_utils._post(
                client, "http://engine/generate", {}, max_retries=1, response_decoder=_decode_events
            )
    assert error.value.response.text == "wrong endpoint"


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
