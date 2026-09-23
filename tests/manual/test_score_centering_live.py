"""Exercise the production score-centering contract against a live SGLang server.

Set MILES_LIVE_SCORE_CENTERING_ENDPOINT and MILES_LIVE_SCORE_CENTERING_MODEL.
MILES_LIVE_SCORE_CENTERING_TOP_K optionally selects the candidate count (default 128).
This probe generates short responses; it does not train or modify the server.
"""

import json
import os
from argparse import Namespace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import requests
from transformers import AutoTokenizer

from miles.rollout.generate_utils.score_centering import (
    append_score_centering_topk,
    configure_score_centering_request,
    validate_score_centering_sample,
)
from miles.utils.types import Sample


@pytest.fixture(scope="module")
def live_responses(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    endpoint = os.environ.get("MILES_LIVE_SCORE_CENTERING_ENDPOINT")
    model = os.environ.get("MILES_LIVE_SCORE_CENTERING_MODEL")
    if not endpoint or not model:
        pytest.skip("set the live score-centering endpoint and local model path to opt in")
    endpoint = endpoint.rstrip("/")
    top_k = int(os.environ.get("MILES_LIVE_SCORE_CENTERING_TOP_K", "128"))
    assert top_k > 0, "The live probe requires a positive candidate count"
    artifact_dir = Path(
        os.environ.get("MILES_LIVE_SCORE_CENTERING_ARTIFACT_DIR") or tmp_path_factory.mktemp("score-centering-live")
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model)
    messages = [{"role": "user", "content": "Compute 17 times 23. Show your reasoning and put the answer in a box."}]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=False,
        add_generation_prompt=True,
        enable_thinking=True,
        preserve_thinking=True,
    )
    # Warm the same prefix before comparing temperatures: GDN cached and
    # uncached prefills can use different kernels and produce different logits.
    warmup = requests.post(
        f"{endpoint}/generate",
        json={"input_ids": prompt, "sampling_params": {"max_new_tokens": 1, "temperature": 1.0}},
        timeout=300,
    )
    warmup.raise_for_status()
    outputs: dict[str, Any] = {"prompt": prompt, "top_k": top_k}
    for temperature in (0.7, 1.0, 1.3):
        args = Namespace(loss_type="score_centering", score_centering_top_k=top_k, rollout_temperature=temperature)
        payload = {
            "input_ids": prompt,
            "return_logprob": True,
            "sampling_params": {"max_new_tokens": 32, "temperature": temperature},
        }
        configure_score_centering_request(args, payload)
        response = requests.post(f"{endpoint}/generate", json=payload, timeout=300)
        response.raise_for_status()
        outputs[str(temperature)] = response.json()
    args = Namespace(loss_type="score_centering", score_centering_top_k=top_k, rollout_temperature=1.0)
    payload = {
        "model": os.environ.get("MILES_LIVE_SCORE_CENTERING_SERVED_MODEL", "default"),
        "messages": messages,
        "max_tokens": 32,
        "logprobs": True,
        "return_meta_info": True,
        "chat_template_kwargs": {"enable_thinking": True, "preserve_thinking": True},
    }
    configure_score_centering_request(args, payload, openai=True)
    response = requests.post(f"{endpoint}/v1/chat/completions", json=payload, timeout=300)
    response.raise_for_status()
    outputs["openai"] = response.json()
    (artifact_dir / "responses.json").write_text(json.dumps(outputs))
    return outputs


def _validate_metadata(meta: dict[str, Any], prompt: list[int], top_k: int) -> None:
    generated = meta["output_token_logprobs"]
    assert generated, "The server must generate at least one token"
    sample = Sample(
        tokens=prompt + [item[1] for item in generated],
        response_length=len(generated),
        rollout_log_probs=[item[0] for item in generated],
    )
    append_score_centering_topk(sample, meta, top_k)
    validate_score_centering_sample(sample, top_k)
    assert np.all(sample.rollout_topk_token_ids >= 0), "This model should return the requested candidates"
    assert np.all(np.isfinite(sample.rollout_topk_log_probs))


@pytest.mark.parametrize("temperature", [0.7, 1.0, 1.3])
def test_native_sampler_contract(live_responses: dict[str, Any], temperature: float) -> None:
    _validate_metadata(
        live_responses[str(temperature)]["meta_info"], live_responses["prompt"], live_responses["top_k"]
    )


def test_openai_sampler_contract(live_responses: dict[str, Any]) -> None:
    choice = live_responses["openai"]["choices"][0]
    _validate_metadata(choice["meta_info"], live_responses["prompt"], live_responses["top_k"])


@pytest.mark.parametrize("temperature", [0.7, 1.3])
def test_candidate_probabilities_include_temperature(live_responses: dict[str, Any], temperature: float) -> None:
    # For the same first-token logits, T*log(q_T) - log(q_1) is constant.
    # Testing differences cancels the unknown full-vocabulary partition sums.
    base = {entry[1]: entry[0] for entry in live_responses["1.0"]["meta_info"]["output_top_logprobs"][0]}
    scaled = {entry[1]: entry[0] for entry in live_responses[str(temperature)]["meta_info"]["output_top_logprobs"][0]}
    base_meta = live_responses["1.0"]["meta_info"]
    scaled_meta = live_responses[str(temperature)]["meta_info"]
    assert base_meta.get("cached_tokens") == scaled_meta.get(
        "cached_tokens"
    ), "Temperature comparison requires the same prefill cache layout"
    common = sorted(base.keys() & scaled.keys())
    assert len(common) >= min(100, max(1, 3 * live_responses["top_k"] // 4))
    differences = np.asarray([temperature * scaled[token] - base[token] for token in common])
    assert (
        np.ptp(differences) < 5e-3
    ), f"Top-k logprobs do not match the temperature-adjusted sampler: spread={np.ptp(differences)}"
