"""Check a running full-training gateway with the Tinker SDK (two optimizer steps).

python tests/manual/tinker/validate_full_training.py --base-url http://localhost:10613 --base-model Qwen/Qwen3-0.6B
"""

import argparse
import json
import time

import httpx
import numpy as np
import tinker
from tinker import types


def create_full_training_client(base_url: str, base_model: str) -> tinker.TrainingClient:
    service = tinker.ServiceClient(base_url=base_url, api_key="tml-full-training-validation")
    model_seq_id = service.holder.get_training_client_id()
    with httpx.Client(base_url=base_url, headers={"X-API-Key": "tml-full-training-validation"}, timeout=60) as http:
        response = http.post(
            "/api/v1/create_model",
            json={
                "session_id": service.holder.get_session_id(),
                "model_seq_id": model_seq_id,
                "base_model": base_model,
                "parameterization": {"type": "full"},
            },
        )
        assert response.is_success, response.text
        created = response.json()
        deadline = time.monotonic() + 300
        while True:
            response = http.post("/api/v1/retrieve_future", json={"request_id": created["request_id"]})
            response.raise_for_status()
            result = response.json()
            assert "error" not in result, result
            if result.get("type") != "try_again":
                break
            assert time.monotonic() < deadline, "model creation timed out"
    client = tinker.TrainingClient(service.holder, model_seq_id=model_seq_id, model_id=created["model_id"])
    assert client.get_info().is_lora is False
    return client


def validate(base_url: str, base_model: str) -> dict:
    client = create_full_training_client(base_url, base_model)
    rows = []
    for offset in (0, 100):
        tokens = list(range(200 + offset, 264 + offset))
        rows.append(
            types.Datum(
                model_input=types.ModelInput.from_ints(tokens[:-1]),
                loss_fn_inputs={
                    "target_tokens": types.TensorData(data=tokens[1:], dtype="int64", shape=[63]),
                    "weights": types.TensorData(data=[1 / 126] * 63, dtype="float32", shape=[63]),
                },
            )
        )
    adam = types.AdamParams(learning_rate=1e-4, grad_clip_norm=1.0)

    def score():
        result = client.forward(rows, "cross_entropy").result()
        return np.array([x for row in result.loss_fn_outputs for x in row["logprobs"].data])

    initial = score()
    initial_checkpoint = client.save_state("before-training").result().path
    client.load_state_with_optimizer(initial_checkpoint).result()
    np.testing.assert_allclose(score(), initial, atol=1e-4, rtol=0)
    for row in rows:
        client.forward_backward([row], "cross_entropy").result()
    first_step = client.optim_step(adam).result()
    first = score()
    assert np.max(np.abs(first - initial)) > 1e-4, "training did not change logprobs"
    checkpoint = client.save_state("after-one").result().path
    sampler_a = client.save_weights_and_get_sampling_client("after-one")
    prompt = types.ModelInput.from_ints([200, 201, 202, 203])
    params = types.SamplingParams(max_tokens=8, temperature=0, seed=7)
    sample_a = sampler_a.sample(prompt, num_samples=1, sampling_params=params).result()
    client.forward_backward(rows, "cross_entropy").result()
    second_step = client.optim_step(adam).result()
    second = score()
    sampler_b = client.save_weights_and_get_sampling_client("after-two")
    sampler_b.sample(prompt, num_samples=1, sampling_params=params).result()
    sample_a_again = sampler_a.sample(prompt, num_samples=1, sampling_params=params).result()
    assert sample_a.sequences[0].tokens == sample_a_again.sequences[0].tokens
    np.testing.assert_allclose(sample_a.sequences[0].logprobs, sample_a_again.sequences[0].logprobs, atol=1e-5, rtol=0)
    client.load_state_with_optimizer(checkpoint).result()
    restored = score()
    np.testing.assert_allclose(restored, first, atol=1e-4, rtol=0)
    for row in rows:
        client.forward_backward([row], "cross_entropy").result()
    replay_step = client.optim_step(adam).result()
    replayed = score()
    np.testing.assert_allclose(replayed, second, atol=1e-2, rtol=0)
    result = {
        "initial_mean_logprob": float(initial.mean()),
        "step_one_mean_logprob": float(first.mean()),
        "step_two_mean_logprob": float(second.mean()),
        "checkpoint_max_logprob_diff": float(abs(restored - first).max()),
        "split_vs_combined_max_logprob_diff": float(abs(replayed - second).max()),
        "first_step": first_step.metrics,
        "second_step": second_step.metrics,
        "replayed_step": replay_step.metrics,
        "named_sampler_survives_weight_switch": True,
    }
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--base-model", required=True)
    cli = parser.parse_args()
    validate(cli.base_url, cli.base_model)
