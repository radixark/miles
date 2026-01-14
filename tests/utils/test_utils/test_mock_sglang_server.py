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
    input_ids = [1, 2, 3, 4, 5]
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
        "text": "I don't understand.",
        "meta_info": {
            "finish_reason": {"type": "stop"},
            "prompt_tokens": 5,
            "cached_tokens": 0,
            "completion_tokens": 5,
            "output_token_logprobs": [
                [-0.0, 40],
                [-0.0078125, 1513],
                [-0.015625, 944],
                [-0.0234375, 3535],
                [-0.03125, 13],
            ],
        },
    }


def test_finish_reason_stop(mock_server):
    def process_fn(prompt: str) -> ProcessResult:
        return ProcessResult(text="Complete response", finish_reason="stop")

    with with_mock_server(process_fn=process_fn) as server:
        response = requests.post(
            f"{server.url}/generate", json={"input_ids": [1, 2, 3], "sampling_params": {}}, timeout=5.0
        )
        assert response.status_code == 200
        data = response.json()

        assert data["meta_info"]["finish_reason"]["type"] == "stop"
        assert "length" not in data["meta_info"]["finish_reason"]


def test_finish_reason_length(mock_server):
    def process_fn(prompt: str) -> ProcessResult:
        return ProcessResult(text="Truncated", finish_reason="length")

    with with_mock_server(process_fn=process_fn) as server:
        response = requests.post(
            f"{server.url}/generate", json={"input_ids": [1, 2, 3], "sampling_params": {}}, timeout=5.0
        )
        assert response.status_code == 200
        data = response.json()

        assert data["meta_info"]["finish_reason"]["type"] == "length"
        assert "length" in data["meta_info"]["finish_reason"]


def test_finish_reason_abort(mock_server):
    def process_fn(prompt: str) -> ProcessResult:
        return ProcessResult(text="Aborted", finish_reason="abort")

    with with_mock_server(process_fn=process_fn) as server:
        response = requests.post(
            f"{server.url}/generate", json={"input_ids": [1, 2, 3], "sampling_params": {}}, timeout=5.0
        )
        assert response.status_code == 200
        data = response.json()

        assert data["meta_info"]["finish_reason"]["type"] == "abort"


def test_return_logprob(mock_server):
    def process_fn(prompt: str) -> ProcessResult:
        return ProcessResult(text="Test", finish_reason="stop")

    with with_mock_server(process_fn=process_fn) as server:
        response = requests.post(
            f"{server.url}/generate",
            json={"input_ids": [1, 2, 3], "sampling_params": {}, "return_logprob": True},
            timeout=5.0,
        )
        assert response.status_code == 200
        data = response.json()

        assert "output_token_logprobs" in data["meta_info"]
        logprobs = data["meta_info"]["output_token_logprobs"]
        assert isinstance(logprobs, list)
        assert len(logprobs) > 0
        assert isinstance(logprobs[0], list)
        assert len(logprobs[0]) == 2
        assert isinstance(logprobs[0][0], float)
        assert isinstance(logprobs[0][1], int)


def test_request_recording(mock_server):
    request1 = {"input_ids": [1, 2, 3], "sampling_params": {"temperature": 0.7}}
    request2 = {"input_ids": [4, 5, 6], "sampling_params": {"temperature": 0.9}, "return_logprob": True}

    requests.post(f"{mock_server.url}/generate", json=request1, timeout=5.0)
    requests.post(f"{mock_server.url}/generate", json=request2, timeout=5.0)

    assert len(mock_server.requests) >= 2
    assert mock_server.requests[-2] == request1
    assert mock_server.requests[-1] == request2

    mock_server.clear_requests()
    assert len(mock_server.requests) == 0


def test_context_manager():
    def process_fn(prompt: str) -> ProcessResult:
        return ProcessResult(text="Context test response", finish_reason="stop")

    with with_mock_server(process_fn=process_fn) as server:
        response = requests.post(
            f"{server.url}/generate", json={"input_ids": [1, 2, 3], "sampling_params": {}}, timeout=5.0
        )
        assert response.status_code == 200
        data = response.json()
        assert data["text"] == "Context test response"


def test_prompt_tokens_calculated_from_input_ids(mock_server):
    input_ids = [10, 20, 30, 40, 50, 60, 70]
    response = requests.post(
        f"{mock_server.url}/generate",
        json={"input_ids": input_ids, "sampling_params": {}},
        timeout=5.0,
    )
    assert response.status_code == 200
    data = response.json()

    assert data["meta_info"]["prompt_tokens"] == len(input_ids)


def test_completion_tokens_calculated_from_output(mock_server):
    def process_fn(prompt: str) -> ProcessResult:
        return ProcessResult(text="Short", finish_reason="stop")

    with with_mock_server(process_fn=process_fn) as server:
        response = requests.post(
            f"{server.url}/generate",
            json={"input_ids": [1, 2, 3], "sampling_params": {}},
            timeout=5.0,
        )
        assert response.status_code == 200
        data = response.json()

        assert data["meta_info"]["completion_tokens"] > 0


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
    assert result.text == "It is 6."
    assert result.finish_reason == "stop"

    result = default_process_fn("What is 1+10?")
    assert result.text == "It is 11."
    assert result.finish_reason == "stop"

    result = default_process_fn("Hello")
    assert result.text == "I don't understand."
    assert result.finish_reason == "stop"


def test_default_process_fn_integration(mock_server):
    tokenizer = mock_server.tokenizer
    prompt_text = "What is 1+7?"
    input_ids = tokenizer.encode(prompt_text, add_special_tokens=False)

    response = requests.post(
        f"{mock_server.url}/generate",
        json={"input_ids": input_ids, "sampling_params": {}},
        timeout=5.0,
    )
    assert response.status_code == 200
    data = response.json()

    assert "It is 8." in data["text"] or "8" in data["text"]
