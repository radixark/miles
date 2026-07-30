"""End-to-end smoke test using only the public Tinker Python SDK.

Run this against a live ``train_tinker.py`` service. The test covers model
creation, forward/backward, Adam, exact optimizer-state restore, weights-only
restore, named and ephemeral sampler snapshots, sampling, and prompt logprobs.
"""

from __future__ import annotations

import argparse
import logging
import math
import uuid

import tinker
from tinker import types
from transformers import AutoTokenizer

logging.getLogger("tinker.lib.api_future_impl").setLevel(logging.ERROR)


def _cross_entropy_datum(tokenizer, text: str) -> tuple[types.Datum, list[int]]:
    tokens = tokenizer.encode(text, add_special_tokens=True)
    if len(tokens) < 2:
        raise ValueError("smoke-test text must tokenize to at least two tokens")
    model_tokens = tokens[:-1]
    datum = types.Datum(
        model_input=types.ModelInput.from_ints(model_tokens),
        loss_fn_inputs={
            "target_tokens": tokens[1:],
            "weights": [1.0] * len(model_tokens),
        },
    )
    return datum, model_tokens


def _assert_forward_output(output: types.ForwardBackwardOutput, expected_values: int) -> None:
    assert len(output.loss_fn_outputs) == 1
    logprobs = output.loss_fn_outputs[0]["logprobs"].data
    assert len(logprobs) == expected_values
    assert all(math.isfinite(value) for value in logprobs)
    assert output.metrics is not None
    assert math.isfinite(output.metrics["loss:sum"])


def _assert_sample(output: types.SampleResponse, expected_samples: int) -> None:
    assert len(output.sequences) == expected_samples
    for sequence in output.sequences:
        assert sequence.tokens
        assert sequence.logprobs is not None
        assert len(sequence.tokens) == len(sequence.logprobs)
        assert all(math.isfinite(value) for value in sequence.logprobs)


def _assert_values_close(left: list[float], right: list[float]) -> None:
    assert len(left) == len(right)
    for left_value, right_value in zip(left, right, strict=True):
        assert math.isclose(left_value, right_value, rel_tol=1e-6, abs_tol=1e-6)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8068")
    parser.add_argument("--api-key", default="tml-local")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--rank", type=int, default=8)
    args = parser.parse_args()

    service = tinker.ServiceClient(base_url=args.base_url.rstrip("/"), api_key=args.api_key)
    capabilities = service.get_server_capabilities()
    supported = {model.model_name for model in capabilities.supported_models}
    assert args.base_model in supported

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    datum, prompt_tokens = _cross_entropy_datum(
        tokenizer,
        "Tinker-compatible training in Miles updates a LoRA adapter.",
    )
    data = [datum]
    adam = types.AdamParams(
        learning_rate=1e-4,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
        weight_decay=0.0,
        grad_clip_norm=1.0,
    )

    print("1/8 create LoRA training client")
    training = service.create_lora_training_client(
        base_model=args.base_model,
        rank=args.rank,
        seed=7,
        train_mlp=True,
        train_attn=True,
        train_unembed=True,
    )
    info = training.get_info()
    assert info.is_lora
    assert info.lora_rank == args.rank

    print("2/8 forward all public losses and run forward_backward")
    forward = training.forward(data, "cross_entropy").result()
    _assert_forward_output(forward, len(prompt_tokens))
    targets = datum.loss_fn_inputs["target_tokens"].tolist()
    baseline_logprobs = forward.loss_fn_outputs[0]["logprobs"].tolist()
    advantages = [1.0 if index % 2 == 0 else -0.5 for index in range(len(prompt_tokens))]
    rl_data = [
        types.Datum(
            model_input=datum.model_input,
            loss_fn_inputs={
                "target_tokens": targets,
                "logprobs": baseline_logprobs,
                "advantages": advantages,
            },
        )
    ]
    for loss_fn, config in [
        ("importance_sampling", None),
        ("ppo", {"clip_low_threshold": 0.8, "clip_high_threshold": 1.2}),
        ("cispo", {"clip_low_threshold": 0.0, "clip_high_threshold": 4.0}),
        ("dro", {"beta": 0.05}),
    ]:
        _assert_forward_output(
            training.forward(rl_data, loss_fn, config).result(),
            len(prompt_tokens),
        )

    topk_data = [
        types.Datum(
            model_input=datum.model_input,
            loss_fn_inputs={
                "target_tokens": [[target, target] for target in targets],
                "weights": [[1.0, 0.0] for _ in targets],
            },
        )
    ]
    _assert_forward_output(
        training.forward(topk_data, "cross_entropy").result(),
        2 * len(prompt_tokens),
    )
    _assert_forward_output(
        training.forward_backward(data, "cross_entropy").result(),
        len(prompt_tokens),
    )

    def custom_loss(_data, logprobs):
        loss = -sum(logprob.mean() for logprob in logprobs)
        return loss, {"custom_loss": float(loss.detach())}

    custom_output = training.forward_backward_custom(data, custom_loss).result()
    assert custom_output.metrics is not None
    assert math.isfinite(custom_output.metrics["custom_loss"])

    print("3/8 save retained gradients and optimizer state")
    suffix = uuid.uuid4().hex[:8]
    training_state = training.save_state(f"smoke-training-{suffix}").result()
    first_step = training.optim_step(adam).result()
    assert first_step.metrics is not None
    first_grad_norm = first_step.metrics["grad_norm"]
    assert math.isfinite(first_grad_norm)
    assert first_grad_norm > 0

    print("4/8 create a second client from exact training state")
    restored = service.create_training_client_from_state_with_optimizer(training_state.path)
    restored_step = restored.optim_step(adam).result()
    assert restored_step.metrics is not None
    restored_grad_norm = restored_step.metrics["grad_norm"]
    assert math.isfinite(restored_grad_norm)
    print(f"    retained-gradient norm: source={first_grad_norm:.9g}, restored={restored_grad_norm:.9g}")
    assert math.isclose(restored_grad_norm, first_grad_norm, rel_tol=1e-6, abs_tol=1e-8)

    # The first checkpoint was intentionally taken before Adam had moments.
    # Take another after one step, restore it across slots, and prove that the
    # FP32 masters, moments, step clock, and retained gradients continue
    # identically by comparing the next update's model outputs.
    training.forward_backward(data, "cross_entropy").result()
    adam_state = training.save_state(f"smoke-adam-{suffix}").result()
    restored.load_state_with_optimizer(adam_state.path).result()
    second_step = training.optim_step(adam).result()
    restored_second_step = restored.optim_step(adam).result()
    assert second_step.metrics is not None
    assert restored_second_step.metrics is not None
    assert math.isclose(
        second_step.metrics["grad_norm"],
        restored_second_step.metrics["grad_norm"],
        rel_tol=1e-6,
        abs_tol=1e-8,
    )
    source_logprobs = training.forward(data, "cross_entropy").result().loss_fn_outputs[0]["logprobs"].tolist()
    restored_logprobs = restored.forward(data, "cross_entropy").result().loss_fn_outputs[0]["logprobs"].tolist()
    _assert_values_close(source_logprobs, restored_logprobs)

    print("5/8 weights-only restore resets optimizer and retained gradients")
    training.load_state(training_state.path).result()
    training.forward_backward(data, "cross_entropy").result()
    training.optim_step(adam).result()

    print("6/8 named sampler snapshot, sampling, and prompt logprobs")
    sampler_state = training.save_weights_for_sampler(f"smoke-sampler-{suffix}").result()
    sampler = service.create_sampling_client(model_path=sampler_state.path)
    sample_params = types.SamplingParams(
        max_tokens=4,
        seed=11,
        temperature=0.8,
        top_k=20,
        top_p=0.95,
    )
    sample = sampler.sample(
        prompt=types.ModelInput.from_ints(prompt_tokens),
        num_samples=2,
        sampling_params=sample_params,
        include_prompt_logprobs=True,
        topk_prompt_logprobs=3,
    ).result()
    _assert_sample(sample, expected_samples=2)
    assert sample.prompt_logprobs is not None
    assert len(sample.prompt_logprobs) == len(prompt_tokens)
    assert sample.topk_prompt_logprobs is not None
    assert len(sample.topk_prompt_logprobs) == len(prompt_tokens)
    computed = sampler.compute_logprobs(types.ModelInput.from_ints(prompt_tokens)).result()
    assert len(computed) == len(prompt_tokens)

    print("7/8 ephemeral sampler snapshot")
    ephemeral = training.save_weights_and_get_sampling_client()
    _assert_sample(
        ephemeral.sample(
            prompt=types.ModelInput.from_ints(prompt_tokens),
            num_samples=1,
            sampling_params=types.SamplingParams(max_tokens=2, seed=13),
        ).result(),
        expected_samples=1,
    )

    print("8/8 base-model sampling session")
    base_sampler = service.create_sampling_client(base_model=args.base_model)
    _assert_sample(
        base_sampler.sample(
            prompt=types.ModelInput.from_ints(prompt_tokens),
            num_samples=1,
            sampling_params=types.SamplingParams(max_tokens=2, seed=17),
        ).result(),
        expected_samples=1,
    )
    print("TINKER SDK SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
