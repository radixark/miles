import math
import os

import pytest
import requests
from tests.e2e.sglang_patch.sglang_server import start_sglang_server
from transformers import AutoTokenizer

MODEL_PATH = os.environ.get("SGLANG_E2E_MODEL_PATH", "Qwen/Qwen3-0.6B")
SEED = 1234
TEMPERATURE = 1.0
TOP_P = 1.0
MAX_COMPLETION_TOKENS = 64
LOGPROB_TOL = 1e-6


@pytest.fixture(scope="module")
def sglang_server():
    server = start_sglang_server(model_path=MODEL_PATH)
    try:
        yield server
    finally:
        server.stop()


@pytest.mark.system
def test_chat_completions_input_ids_equivalence(sglang_server):
    """Validate that providing input_ids yields the same completion as raw messages."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    messages = _build_messages()
    # Build the same prompt two ways: message list vs. explicit input_ids.
    input_ids = _build_input_ids(tokenizer, messages)

    # Request completions for both payload variants.
    response_a = _post_chat(sglang_server.base_url, _build_payload(messages))
    response_b = _post_chat(sglang_server.base_url, _build_payload(messages, input_ids))

    choice_a = response_a["choices"][0]
    choice_b = response_b["choices"][0]

    # The generated content and finish reason should match across variants.
    assert choice_a["message"]["content"] == choice_b["message"]["content"]
    assert choice_a["finish_reason"] == choice_b["finish_reason"]

    # Compare token ids and per-token logprobs for exact equivalence.
    token_ids_a, logprobs_a = _extract_tokens_and_logprobs(choice_a)
    token_ids_b, logprobs_b = _extract_tokens_and_logprobs(choice_b)

    assert token_ids_a == token_ids_b
    assert len(logprobs_a) == len(logprobs_b)

    for index, (a_val, b_val) in enumerate(zip(logprobs_a, logprobs_b, strict=True)):
        assert math.isclose(a_val, b_val, abs_tol=LOGPROB_TOL), f"logprob mismatch at {index}: {a_val} vs {b_val}"


@pytest.mark.system
def test_chat_completions_input_logprobs_prompt_ids_match(sglang_server):
    """Ensure input_ids are echoed exactly in input_token_ids and logprobs are present."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    messages = _build_messages()
    input_ids = _build_input_ids(tokenizer, messages)

    response = _post_chat(sglang_server.base_url, _build_payload(messages, input_ids))
    choice = response["choices"][0]

    input_token_ids = _extract_input_token_ids(choice)

    assert input_token_ids == input_ids
    assert choice.get("logprobs", {}).get("content"), "logprobs content is missing"


def _post_chat(base_url: str, payload: dict) -> dict:
    response = requests.post(f"{base_url}/v1/chat/completions", json=payload, timeout=120)
    print(f"response: {response.json()}", flush=True)
    assert response.status_code == 200, response.text
    return response.json()


def _build_messages() -> list[dict]:
    return [
        {"role": "system", "content": "You are a concise assistant."},
        {"role": "user", "content": "Answer with one word: 2+2?"},
    ]


def _build_input_ids(tokenizer, messages: list[dict]) -> list[int]:
    return tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)


def _build_payload(messages: list[dict], input_ids: list[int] | None = None) -> dict:
    payload = {
        "model": MODEL_PATH,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_completion_tokens": MAX_COMPLETION_TOKENS,
        "seed": SEED,
        "logprobs": True,
        "messages": messages,
        "logprob_start_len": 0,
    }
    if input_ids is not None:
        payload["input_ids"] = input_ids
    return payload


def _extract_tokens_and_logprobs(choice: dict) -> tuple[list[int], list[float]]:
    logprobs = choice.get("logprobs", {}).get("content")
    assert logprobs, "logprobs content is missing"

    token_ids = []
    for item in logprobs:
        token_ids.append(item["token_id"])
    values = [item["logprob"] for item in logprobs]
    return token_ids, values


def _extract_input_token_ids(choice: dict) -> list[int]:
    token_ids = choice.get("input_token_ids")
    assert token_ids is not None, "input_token_ids is missing in response"

    print(f"input_token_ids: {token_ids}", flush=True)
    return token_ids
