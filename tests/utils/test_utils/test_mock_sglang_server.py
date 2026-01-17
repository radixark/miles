import asyncio
import concurrent.futures
import time

import pytest
import requests

from miles.utils.test_utils.mock_sglang_server import Counter, ProcessResult, default_process_fn, with_mock_server


@pytest.fixture(scope="module")
def mock_server():
    with with_mock_server() as server:
        yield server


def test_basic_server_start_stop(mock_server):
    assert mock_server.port > 0
    assert f"http://{mock_server.host}:{mock_server.port}" == mock_server.url


def test_generate_endpoint_basic(mock_server):
    prompt = "What is 1+7?"
    input_ids = mock_server.tokenizer.encode(prompt, add_special_tokens=False)
    assert input_ids == [3838, 374, 220, 16, 10, 22, 30]

    response = requests.post(
        f"{mock_server.url}/generate",
        json={
            "input_ids": input_ids,
            "sampling_params": {"temperature": 0.7, "max_new_tokens": 10},
            "return_logprob": True,
        },
        timeout=5.0,
    )
    assert response.status_code == 200
    data = response.json()

    assert data == {
        "text": "\\boxed{8}",
        "meta_info": {
            "finish_reason": {"type": "stop"},
            "prompt_tokens": len(input_ids),
            "cached_tokens": 0,
            "completion_tokens": 5,
            "output_token_logprobs": [
                [-0.0, 59],
                [-0.0078125, 79075],
                [-0.015625, 90],
                [-0.0234375, 23],
                [-0.03125, 92],
            ],
        },
    }


def test_process_fn_receives_decoded_prompt():
    received_prompts = []

    def process_fn(prompt: str) -> ProcessResult:
        received_prompts.append(prompt)
        return ProcessResult(text="response", finish_reason="stop")

    with with_mock_server(process_fn=process_fn) as server:
        requests.post(f"{server.url}/generate", json={"input_ids": [1, 2, 3], "sampling_params": {}}, timeout=5.0)

    assert len(received_prompts) == 1
    assert isinstance(received_prompts[0], str)


def test_default_process_fn():
    assert default_process_fn("What is 1+5?") == ProcessResult(text="\\boxed{6}", finish_reason="stop")
    assert default_process_fn("What is 1+10?") == ProcessResult(text="\\boxed{11}", finish_reason="stop")
    assert default_process_fn("Hello") == ProcessResult(text="I don't understand.", finish_reason="stop")


def test_request_log_and_reset_stats(mock_server):
    mock_server.reset_stats()
    assert len(mock_server.request_log) == 0

    payload = {"input_ids": [1, 2, 3], "sampling_params": {"temperature": 0.5}, "return_logprob": True}
    requests.post(f"{mock_server.url}/generate", json=payload, timeout=5.0)
    assert len(mock_server.request_log) == 1
    assert mock_server.request_log[0] == payload

    mock_server.reset_stats()
    assert len(mock_server.request_log) == 0
    assert mock_server.max_concurrent == 0


@pytest.mark.parametrize("latency,min_time,max_time", [(0.0, 0.0, 0.3), (0.5, 0.5, 1.0)])
def test_latency(latency, min_time, max_time):
    with with_mock_server(latency=latency) as server:
        start = time.time()
        requests.post(f"{server.url}/generate", json={"input_ids": [1], "sampling_params": {}}, timeout=5.0)
        elapsed = time.time() - start
        assert min_time <= elapsed < max_time


def test_max_concurrent_with_latency():
    with with_mock_server(latency=0.1) as server:

        def send_request():
            requests.post(f"{server.url}/generate", json={"input_ids": [1], "sampling_params": {}}, timeout=5.0)

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(send_request) for _ in range(3)]
            concurrent.futures.wait(futures)

        assert server.max_concurrent == 3


def test_counter_tracks_max():
    counter = Counter()
    assert counter.max_value == 0

    with counter.track():
        assert counter.max_value == 1
        with counter.track():
            assert counter.max_value == 2

    counter.reset()
    assert counter.max_value == 0


def test_counter_concurrent_tasks():
    counter = Counter()

    async def task():
        with counter.track():
            await asyncio.sleep(0.1)

    async def run_all():
        await asyncio.gather(task(), task(), task())

    asyncio.run(run_all())
    assert counter.max_value == 3
