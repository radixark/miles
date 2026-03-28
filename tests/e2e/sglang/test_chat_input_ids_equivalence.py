import math
import os

import pytest
import requests
from tests.e2e.sglang.utils.sglang_server import start_sglang_server
from transformers import AutoTokenizer

MODEL_PATH = os.environ.get("SGLANG_E2E_MODEL_PATH", "Qwen/Qwen3-0.6B")
SEED = 1234
MAX_NEW_TOKENS = 100
LOGPROB_TOL = 1e-6


@pytest.fixture(scope="module")
def sglang_server():
    server = start_sglang_server(model_path=MODEL_PATH)
    try:
        yield server
    finally:
        server.stop()


def _build_messages() -> list[dict]:
    return [
        {"role": "system", "content": "You are a concise assistant."},
        {"role": "user", "content": "Answer with one word: 2+2?"},
    ]


@pytest.mark.system
def test_generate_and_chat_completions_equivalence(sglang_server):
    """The /generate (token) and /v1/chat/completions (message) endpoints
    must produce identical output given the same prompt."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    messages = _build_messages()
    template_out = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    input_ids = template_out["input_ids"] if hasattr(template_out, "keys") else template_out

    gen_resp = _post_generate(sglang_server.base_url, input_ids)
    chat_resp = _post_chat(sglang_server.base_url, messages)

    chat_choice = chat_resp["choices"][0]

    # --- output text ---
    gen_text = gen_resp["text"]
    chat_text = chat_choice["message"]["content"]
    assert gen_text == chat_text, f"text mismatch:\n  generate: {gen_text!r}\n  chat:     {chat_text!r}"

    # --- output token ids (via generate logprobs vs tokenizing chat text) ---
    gen_token_ids = [t[1] for t in gen_resp["meta_info"]["output_token_logprobs"]]
    chat_token_ids = list(tokenizer.encode(chat_text, add_special_tokens=False))
    assert (
        gen_token_ids == chat_token_ids
    ), f"output token ids mismatch:\n  generate: {gen_token_ids[:20]}...\n  chat(enc): {chat_token_ids[:20]}..."

    # --- output logprobs ---
    gen_logprobs = [t[0] for t in gen_resp["meta_info"]["output_token_logprobs"]]
    chat_logprobs_content = (chat_choice.get("logprobs") or {}).get("content")
    assert chat_logprobs_content is not None, "logprobs.content missing in chat response"
    chat_logprobs = [t["logprob"] for t in chat_logprobs_content]
    assert len(gen_logprobs) == len(
        chat_logprobs
    ), f"logprob length mismatch: generate={len(gen_logprobs)} chat={len(chat_logprobs)}"
    for i, (g, c) in enumerate(zip(gen_logprobs, chat_logprobs, strict=True)):
        assert math.isclose(g, c, abs_tol=LOGPROB_TOL), f"logprob mismatch at {i}: {g} vs {c}"

    # --- prompt token ids ---
    chat_prompt_ids = chat_choice.get("prompt_token_ids")
    assert chat_prompt_ids is not None, "prompt_token_ids missing in chat response"
    assert chat_prompt_ids == input_ids, "prompt_token_ids from chat != local apply_chat_template"


def _post_generate(base_url: str, input_ids: list[int]) -> dict:
    payload = {
        "input_ids": input_ids,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": MAX_NEW_TOKENS,
            "sampling_seed": SEED,
        },
        "return_logprob": True,
    }
    resp = requests.post(f"{base_url}/generate", json=payload, timeout=120)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    print(f"generate response text: {data['text']!r}", flush=True)
    return data


def _post_chat(base_url: str, messages: list[dict]) -> dict:
    payload = {
        "model": MODEL_PATH,
        "messages": messages,
        "temperature": 0,
        "max_completion_tokens": MAX_NEW_TOKENS,
        "seed": SEED,
        "logprobs": True,
        "return_prompt_token_ids": True,
    }
    resp = requests.post(f"{base_url}/v1/chat/completions", json=payload, timeout=120)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    print(f"chat response text: {data['choices'][0]['message']['content']!r}", flush=True)
    return data
