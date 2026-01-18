import pytest
import requests

from miles.utils.test_utils.mock_sglang_server import ProcessResult, default_process_fn, with_mock_server


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


def test_process_fn_receives_decoded_prompt(mock_server):
    received_prompts = []

    def process_fn(prompt: str) -> ProcessResult:
        received_prompts.append(prompt)
        return ProcessResult(text="response", finish_reason="stop")

    with with_mock_server(process_fn=process_fn) as server:
        input_ids = [1, 2, 3]
        requests.post(f"{server.url}/generate", json={"input_ids": input_ids, "sampling_params": {}}, timeout=5.0)

        assert len(received_prompts) == 1
        assert isinstance(received_prompts[0], str)


def test_default_process_fn():
    result = default_process_fn("What is 1+5?")
    assert result.text == "\\boxed{6}"
    assert result.finish_reason == "stop"

    result = default_process_fn("What is 1+10?")
    assert result.text == "\\boxed{11}"
    assert result.finish_reason == "stop"

    result = default_process_fn("Hello")
    assert result.text == "I don't understand."
    assert result.finish_reason == "stop"
