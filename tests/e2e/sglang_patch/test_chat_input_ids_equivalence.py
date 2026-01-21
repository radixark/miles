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
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    messages = [
        {"role": "system", "content": "You are a concise assistant."},
        {"role": "user", "content": "Answer with one word: 2+2?"},
    ]

    input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)

    base_payload = {
        "model": MODEL_PATH,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_completion_tokens": MAX_COMPLETION_TOKENS,
        "seed": SEED,
        "logprobs": True,
    }

    payload_a = {**base_payload, "messages": messages}
    payload_b = {**base_payload, "messages": messages, "input_ids": input_ids}

    response_a = _post_chat(sglang_server.base_url, payload_a)
    response_b = _post_chat(sglang_server.base_url, payload_b)

    choice_a = response_a["choices"][0]
    choice_b = response_b["choices"][0]

    assert choice_a["message"]["content"] == choice_b["message"]["content"]
    assert choice_a["finish_reason"] == choice_b["finish_reason"]

    token_ids_a, logprobs_a = _extract_logprobs(choice_a, tokenizer)
    token_ids_b, logprobs_b = _extract_logprobs(choice_b, tokenizer)

    assert token_ids_a == token_ids_b
    assert len(logprobs_a) == len(logprobs_b)

    for index, (a_val, b_val) in enumerate(zip(logprobs_a, logprobs_b, strict=True)):
        assert math.isclose(a_val, b_val, abs_tol=LOGPROB_TOL), f"logprob mismatch at {index}: {a_val} vs {b_val}"


def _post_chat(base_url: str, payload: dict) -> dict:
    response = requests.post(f"{base_url}/v1/chat/completions", json=payload, timeout=120)
    assert response.status_code == 200, response.text
    return response.json()


def _extract_logprobs(choice: dict, tokenizer) -> tuple[list[int], list[float]]:
    logprobs = choice.get("logprobs", {}).get("content")
    assert logprobs, "logprobs content is missing"

    tokens = [item["token"] for item in logprobs]
    token_ids = [_token_id_from_token_string(tokenizer, token) for token in tokens]
    values = [item["logprob"] for item in logprobs]
    return token_ids, values


def _token_id_from_token_string(tokenizer, token: str) -> int:
    token_id = tokenizer.convert_tokens_to_ids(token)
    if token_id is not None:
        return token_id

    encoded = tokenizer.encode(token, add_special_tokens=False)
    assert len(encoded) == 1, f"token_id conversion failed for token: {token!r}"
    return encoded[0]
