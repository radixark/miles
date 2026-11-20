import asyncio
import base64
import json
import struct
from types import SimpleNamespace
from unittest.mock import AsyncMock
import httpx
import pytest
from miles.router.dynamo.api_client import GenerateProtocolError, decode_response, post_generate
from miles.router.dynamo.config import DynamoConfig
from miles.utils.http_utils import _post


def chunk(token, count, *, text="x", terminal=False):
    return (
        "data: "
        + json.dumps(
            {
                "output_ids": [token],
                "text": text,
                "meta_info": {
                    "id": "request",
                    "completion_tokens": count,
                    "prompt_tokens": 1,
                    "output_token_logprobs": [[-0.5, token, text]],
                    "weight_version": "2",
                    "finish_reason": {"type": "stop"} if terminal else None,
                },
            }
        )
        + "\n\n"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("first_failure", ["truncated", "status", "transient_error"])
async def test_native_retry_discards_incomplete_attempt(monkeypatch, first_failure):
    attempts = []

    def handle(request):
        attempts.append(request)
        if len(attempts) == 1:
            if first_failure == "status":
                return httpx.Response(503)
            body = chunk(99, 1)
            if first_failure == "transient_error":
                body += 'event: error\ndata: {"error":{"message":"worker lost","code":503}}\n\n'
        else:
            body = chunk(10, 1, text="a") + chunk(11, 2, text="b", terminal=True) + "data: [DONE]\n\n"
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=body)

    monkeypatch.setattr("miles.utils.http_utils.asyncio.sleep", AsyncMock())
    payload = {"input_ids": [1], "stream": True}
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        result = await _post(client, "http://frontend/generate", payload, response_decoder=decode_response)
    assert len(attempts) == 2
    assert result["output_ids"] == [10, 11]
    assert result["text"] == "ab"
    assert result["meta_info"]["output_token_logprobs"] == [[-0.5, 10, "a"], [-0.5, 11, "b"]]
    assert payload == {"input_ids": [1], "stream": True}


@pytest.mark.asyncio
async def test_native_transport_keeps_sixty_attempts(monkeypatch):
    calls = []

    def handle(request):
        calls.append(request)
        return httpx.Response(503)

    monkeypatch.setattr("miles.utils.http_utils.asyncio.sleep", AsyncMock())
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        with pytest.raises(httpx.HTTPStatusError):
            await _post(client, "http://frontend/generate", {}, response_decoder=decode_response)
    assert len(calls) == 60


@pytest.mark.asyncio
async def test_cancel_does_not_retry():
    calls = []

    async def handle(request):
        calls.append(request)
        raise asyncio.CancelledError

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        with pytest.raises(asyncio.CancelledError):
            await _post(client, "http://frontend/generate", {}, response_decoder=decode_response)
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_timeout_also_covers_response_headers():
    async def handle(request):
        await asyncio.Event().wait()

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        with pytest.raises(TimeoutError):
            await _post(
                client,
                "http://frontend/generate",
                {},
                max_retries=1,
                response_decoder=decode_response,
                response_timeout=0.01,
            )


@pytest.mark.asyncio
async def test_abort_echo_is_not_appended_twice():
    abort = {
        "output_ids": [10],
        "text": "a",
        "meta_info": {
            "id": "request",
            "completion_tokens": 1,
            "output_token_logprobs": [[-0.5, 10, "a"]],
            "finish_reason": {"type": "abort"},
        },
    }
    response = httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        text=chunk(10, 1, text="a") + "data: " + json.dumps(abort) + "\n\ndata: [DONE]\n\n",
    )
    output = await decode_response(response)
    assert output["text"] == "a"
    assert output["output_ids"] == [10]
    assert output["meta_info"]["output_token_logprobs"] == [[-0.5, 10, "a"]]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "field,value,error",
    [
        ("id", "another-request", "identity changed"),
        ("completion_tokens", 3, "incremental token output"),
        ("output_token_logprobs", [[-0.5, 99, "b"]], "tokens and logprobs disagree"),
        ("output_token_logprobs", [[-0.5]], "malformed token logprobs"),
        ("finish_reason", "invalid", "invalid finish_reason"),
        ("output_token_sampling_mask", 17, "expected output_token_sampling_mask array"),
    ],
)
async def test_rejects_inconsistent_incremental_responses(field, value, error):
    final = json.loads(chunk(11, 2, text="b", terminal=True).removeprefix("data: "))
    final["meta_info"][field] = value
    response = httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        text=chunk(10, 1, text="a") + "data: " + json.dumps(final) + "\n\ndata: [DONE]\n\n",
    )
    with pytest.raises(GenerateProtocolError, match=error):
        await decode_response(response)


@pytest.mark.asyncio
@pytest.mark.parametrize("frontend_key", [None, "frontend-key"])
async def test_generation_wrapper_preserves_caller_inputs(monkeypatch, frontend_key):
    send = AsyncMock(return_value={"text": "complete"})
    monkeypatch.setattr("miles.router.dynamo.api_client.post", send)
    args = SimpleNamespace(dynamo_config=SimpleNamespace(request_timeout_seconds=7), sglang_api_key="engine-key")
    payload = {"input_ids": [1], "stream": False, "return_logprob": True}
    headers = {"X-Request-ID": "request"}
    if frontend_key:
        headers["Authorization"] = f"Bearer {frontend_key}"
    original_headers = dict(headers)

    assert await post_generate(args, "http://frontend/generate", payload, headers=headers) == {"text": "complete"}
    assert send.call_args.args == (
        "http://frontend/generate",
        {"input_ids": [1], "stream": True, "return_logprob": True},
    )
    assert send.call_args.kwargs["headers"] == original_headers
    assert "engine-key" not in str(send.call_args.kwargs["headers"])
    assert send.call_args.kwargs["response_timeout"] == 7
    assert payload == {"input_ids": [1], "stream": False, "return_logprob": True}
    assert headers == original_headers


@pytest.mark.asyncio
@pytest.mark.parametrize("setting", ["missing", None, False])
async def test_generation_requires_logprobs_before_dispatch(monkeypatch, setting):
    send = AsyncMock()
    monkeypatch.setattr("miles.router.dynamo.api_client.post", send)
    payload = {"input_ids": [1]}
    if setting != "missing":
        payload["return_logprob"] = setting
    args = SimpleNamespace(dynamo_config=SimpleNamespace(request_timeout_seconds=None))
    with pytest.raises(GenerateProtocolError, match="requires return_logprob=True"):
        await post_generate(args, "http://frontend/generate", payload)
    send.assert_not_awaited()


@pytest.fixture
def dynamo_config():
    return DynamoConfig(
        namespace="test-run",
        discovery={"backend": "etcd", "endpoints": [{"host": "localhost", "port": 2379}]},
        pins={"dynamo": "a" * 40, "sglang": "b" * 40},
    )


@pytest.mark.asyncio
async def test_default_deadline_allows_long_generation_and_pause(monkeypatch, dynamo_config):
    loop = asyncio.get_running_loop()
    clock = [loop.time()]
    calls = []

    class PausedStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield chunk(10, 1, text="a").encode()
            clock[0] += 601
            # Let an old 600-second total deadline fire if it is still scheduled.
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            yield (chunk(11, 2, text="b", terminal=True) + "data: [DONE]\n\n").encode()

    def handle(request):
        calls.append(request)
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=PausedStream())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:

        async def send(url, payload, **kwargs):
            return await _post(client, url, payload, max_retries=1, **kwargs)

        monkeypatch.setattr("miles.router.dynamo.api_client.post", send)
        with monkeypatch.context() as timing:
            timing.setattr(loop, "time", lambda: clock[0])
            output = await post_generate(
                SimpleNamespace(dynamo_config=dynamo_config), "http://frontend/generate", {"return_logprob": True}
            )
    assert output["text"] == "ab"
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_explicit_deadline_still_bounds_generation(monkeypatch, dynamo_config):
    async def handle(request):
        await asyncio.Event().wait()

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:

        async def send(url, payload, **kwargs):
            return await _post(client, url, payload, max_retries=1, **kwargs)

        monkeypatch.setattr("miles.router.dynamo.api_client.post", send)
        with pytest.raises(TimeoutError):
            await post_generate(
                SimpleNamespace(dynamo_config=dynamo_config.model_copy(update={"request_timeout_seconds": 0.01})),
                "http://frontend/generate",
                {"return_logprob": True},
            )


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["content_type", "incremental_count"])
async def test_deterministic_decode_failures_are_not_retried(monkeypatch, failure):
    calls = []

    def handle(request):
        calls.append(request)
        return httpx.Response(
            200,
            headers={"content-type": "application/json" if failure == "content_type" else "text/event-stream"},
            text=chunk(10, 2, terminal=True) + "data: [DONE]\n\n",
        )

    monkeypatch.setattr("miles.utils.http_utils.asyncio.sleep", AsyncMock())
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        with pytest.raises(GenerateProtocolError):
            await _post(client, "http://frontend/generate", {}, response_decoder=decode_response)
    assert len(calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [400, 503])
async def test_failed_abort_is_an_error_not_a_partial_sample(monkeypatch, status):
    calls = []

    def handle(request):
        calls.append(request)
        if len(calls) == 1:
            result = {
                "text": "",
                "output_ids": [],
                "meta_info": {
                    "id": "request",
                    "completion_tokens": 0,
                    "output_token_logprobs": [],
                    "finish_reason": {"type": "abort", "status_code": status, "message": "engine rejected request"},
                },
            }
            body = "data: " + json.dumps(result) + "\n\n"
        else:
            body = chunk(10, 1, terminal=True)
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=body + "data: [DONE]\n\n")

    monkeypatch.setattr("miles.utils.http_utils.asyncio.sleep", AsyncMock())
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        if status == 400:
            with pytest.raises(GenerateProtocolError, match="engine rejected request"):
                await _post(client, "http://frontend/generate", {}, response_decoder=decode_response)
            assert len(calls) == 1
        else:
            result = await _post(client, "http://frontend/generate", {}, response_decoder=decode_response)
            assert result["output_ids"] == [10]
            assert result["meta_info"]["finish_reason"] == {"type": "stop"}
            assert len(calls) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("finish_type", ["stop", "abort"])
async def test_native_request_metadata_survives_multichunk_generation(finish_type):
    # Contract fixture from sglang-miles 318ffdf2: _slice_streaming_output_meta_info slices five
    # output arrays; input logprobs persist in ReqState, routed experts and spec metrics arrive at finish.
    prompt_logprobs = [[None, 1, None], [-0.1, 2, "prompt"]]
    routed_bytes = struct.pack("<6i", 1, 2, 3, 4, 5, 6)  # Three token positions, one layer, top-k two.
    routed_experts = base64.b64encode(routed_bytes).decode()
    first = json.loads(chunk(10, 1, text="a").removeprefix("data: "))
    first["meta_info"].update(input_token_logprobs=prompt_logprobs, cached_tokens=2)
    final = json.loads(chunk(11, 2, text="b", terminal=True).removeprefix("data: "))
    final["meta_info"].update(
        input_token_logprobs=prompt_logprobs,
        cached_tokens=2,
        routed_experts=routed_experts,
        spec_verify_ct=4,
        spec_num_correct_drafts=3,
        spec_num_proposed_drafts=8,
        spec_accept_rate=3 / 8,
        weight_versions=[{"version": "7", "start": 0, "end": 2}],
        finish_reason={"type": finish_type},
    )
    if finish_type == "abort":
        final.update(text="ab", output_ids=[10, 11])
        final["meta_info"]["output_token_logprobs"] = [[-0.5, 10, "a"], [-0.5, 11, "b"]]
    response = httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        text="".join("data: " + json.dumps(event) + "\n\n" for event in (first, final)) + "data: [DONE]\n\n",
    )

    output = await decode_response(response)

    assert output["text"] == "ab"
    assert output["output_ids"] == [10, 11]
    assert output["meta_info"]["output_token_logprobs"] == [[-0.5, 10, "a"], [-0.5, 11, "b"]]
    for name in ("input_token_logprobs", "cached_tokens", "routed_experts", "weight_versions"):
        assert output["meta_info"][name] == final["meta_info"][name]
    assert base64.b64decode(output["meta_info"]["routed_experts"]) == routed_bytes
    assert output["meta_info"]["spec_verify_ct"] == 4
    assert output["meta_info"]["spec_num_correct_drafts"] == 3
    assert output["meta_info"]["spec_num_proposed_drafts"] == 8
    assert output["meta_info"]["spec_accept_rate"] == 3 / 8


@pytest.mark.asyncio
async def test_generation_wrapper_uses_existing_post_without_dynamo(monkeypatch):
    send = AsyncMock(return_value={"text": "complete"})
    monkeypatch.setattr("miles.router.dynamo.api_client.post", send)
    payload = {"input_ids": [1]}
    headers = {"X-Request-ID": "request"}

    assert await post_generate(
        SimpleNamespace(dynamo_config=None), "http://engine/generate", payload, headers=headers
    ) == {"text": "complete"}
    send.assert_awaited_once_with("http://engine/generate", payload, headers=headers)
