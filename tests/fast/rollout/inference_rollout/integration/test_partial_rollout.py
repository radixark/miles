import asyncio
import threading

import pytest
from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from tests.fast.rollout.inference_rollout.integration.utils import integration_env_config

from miles.ray.rollout.rollout_executor import _compute_rollout_function_config
from miles.ray.rollout.train_data_conversion import convert_samples_to_train_data
from miles.rollout.base_types import RolloutFnConstructorInput, RolloutFnTrainInput
from miles.rollout.inference_rollout.compatibility import call_rollout_function, load_rollout_function
from miles.utils.test_utils.mock_sglang_server import ProcessResult
from miles.utils.types import Sample

_DATA_ROWS = [
    {"input": "What is 1+7?", "label": "8"},
    {"input": "What is 1+8?", "label": "9"},
]


async def _wait_threading_event(event: threading.Event, timeout: float = 5.0) -> None:
    await asyncio.wait_for(asyncio.to_thread(event.wait), timeout=timeout)


@pytest.mark.parametrize(
    "rollout_env,mask_offpolicy",
    [
        pytest.param(
            integration_env_config(
                [
                    "--partial-rollout",
                    "--mask-offpolicy-in-partial-rollout",
                    "--rollout-batch-size",
                    "1",
                    "--over-sampling-batch-size",
                    "2",
                    "--no-rollout-shuffle",
                ],
                data_rows=_DATA_ROWS,
            ),
            True,
            id="masked",
        ),
        pytest.param(
            integration_env_config(
                [
                    "--partial-rollout",
                    "--rollout-batch-size",
                    "1",
                    "--over-sampling-batch-size",
                    "2",
                    "--no-rollout-shuffle",
                ],
                data_rows=_DATA_ROWS,
            ),
            False,
            id="unmasked",
        ),
    ],
    indirect=["rollout_env"],
)
def test_partial_rollout_resumes_buffered_sample_and_converts_loss_mask(rollout_env, mask_offpolicy: bool) -> None:
    env = rollout_env
    second_prompt_started = threading.Event()
    abort_received = threading.Event()

    partial_text = "\\boxed"
    remaining_text = "{9}"
    partial_token_len = len(env.mock_server.tokenizer.encode(partial_text, add_special_tokens=False))
    remaining_token_len = len(env.mock_server.tokenizer.encode(remaining_text, add_special_tokens=False))

    # Replace /generate and /abort_request with async coordination so the mock server's event loop never blocks.
    env.mock_server.app.router.routes = [
        route
        for route in env.mock_server.app.router.routes
        if not (isinstance(route, APIRoute) and route.path in {"/generate", "/abort_request"})
    ]

    @env.mock_server.app.post("/abort_request")
    async def abort_request(_request: Request) -> JSONResponse:
        abort_received.set()
        return JSONResponse(content={"status": "ok"})

    @env.mock_server.app.post("/generate")
    async def generate(request: Request) -> JSONResponse:
        payload = await request.json()
        env.mock_server.request_log.append(payload)
        prompt = env.mock_server.tokenizer.decode(payload["input_ids"], skip_special_tokens=False)
        if "1+7?" in prompt:
            await _wait_threading_event(second_prompt_started)
            result = ProcessResult(text="\\boxed{8}", finish_reason="stop")
        elif partial_text not in prompt:
            second_prompt_started.set()
            await _wait_threading_event(abort_received)
            result = ProcessResult(text=partial_text, finish_reason="abort")
        else:
            result = ProcessResult(text=remaining_text, finish_reason="stop")
        env.mock_server.process_fn = lambda _p, _r=result: _r
        return JSONResponse(content=env.mock_server._compute_generate_response(payload))

    fn = load_rollout_function(
        RolloutFnConstructorInput(
            args=_compute_rollout_function_config(env.args, env.args.rollout_function_path),
            data_source=env.data_source,
        ),
        env.args.rollout_function_path,
    )

    out0 = call_rollout_function(fn, RolloutFnTrainInput(rollout_id=0))

    assert len(out0.samples) == 1
    assert out0.samples[0][0].prompt == "What is 1+7?"
    assert env.data_source.get_buffer_length() == 1
    buffered_sample = env.data_source.buffer[0][0]
    assert buffered_sample.status == Sample.Status.ABORTED
    assert buffered_sample.response == partial_text
    assert buffered_sample.metadata["start_rollout_id"] == 0

    out1 = call_rollout_function(fn, RolloutFnTrainInput(rollout_id=1))
    assert len(out1.samples) == 1
    resumed = out1.samples[0][0]
    assert resumed.prompt == "What is 1+8?"
    assert resumed.status == Sample.Status.COMPLETED
    assert resumed.response == partial_text + remaining_text
    assert resumed.response_length == partial_token_len + remaining_token_len
    expected_mask = (
        [0] * partial_token_len + [1] * remaining_token_len
        if mask_offpolicy
        else [1] * (partial_token_len + remaining_token_len)
    )
    if mask_offpolicy:
        assert resumed.loss_mask == expected_mask
    else:
        assert resumed.loss_mask is None
    resumed.validate()

    train_data = convert_samples_to_train_data(env.args, [resumed], {}, None, None)
    assert train_data["loss_masks"] == [expected_mask]
