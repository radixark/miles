"""Run Hersh's trainer/stale-sampler/resync logprob gate for every optimizer step."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import statistics
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import tinker
from tinker import types
from transformers import AutoTokenizer

PROMPT = "Explain why the sky appears blue during the day in two sentences."


@dataclass(frozen=True)
class DeltaStats:
    tokens: int
    mean: float
    p50: float
    p90: float
    p99: float
    maximum: float


def _percentile(sorted_values: list[float], percentile: float) -> float:
    return sorted_values[round((len(sorted_values) - 1) * percentile / 100)]


def _delta_stats(left: list[float], right: list[float], weights: list[float]) -> DeltaStats:
    assert len(left) == len(right) == len(weights)
    deltas = [abs(a - b) for a, b, weight in zip(left, right, weights, strict=True) if weight > 0]
    assert deltas and all(math.isfinite(delta) for delta in deltas)
    deltas.sort()
    return DeltaStats(
        tokens=len(deltas),
        mean=statistics.fmean(deltas),
        p50=_percentile(deltas, 50),
        p90=_percentile(deltas, 90),
        p99=_percentile(deltas, 99),
        maximum=deltas[-1],
    )


def _build_datum(tokenizer, completion_tokens: list[int]) -> tuple[list[int], list[float], types.Datum]:
    prompt_tokens = tokenizer.encode(PROMPT)
    tokens = [int(token) for token in prompt_tokens + completion_tokens]
    weights = [float(position >= len(prompt_tokens)) for position in range(1, len(tokens))]
    datum = types.Datum(
        model_input=types.ModelInput.from_ints(tokens[:-1]),
        loss_fn_inputs={"target_tokens": tokens[1:], "weights": weights},
    )
    return tokens, weights, datum


async def _sample_completion(
    sampling, tokenizer, completion_tokens: int, timeout_seconds: float
) -> list[int]:
    result = await asyncio.wait_for(
        sampling.sample_async(
            prompt=types.ModelInput.from_ints(tokenizer.encode(PROMPT)),
            num_samples=1,
            sampling_params=types.SamplingParams(
                max_tokens=completion_tokens, temperature=0.0, seed=17
            ),
        ),
        timeout=timeout_seconds,
    )
    completion_tokens = [int(token) for token in result.sequences[0].tokens]
    assert completion_tokens, "sampler must produce fixed on-policy completion tokens"
    return completion_tokens


async def _result(future, timeout_seconds: float):
    return await asyncio.wait_for(future.result_async(), timeout=timeout_seconds)


async def _trainer_logprobs(training, datum: types.Datum, timeout_seconds: float) -> list[float]:
    result = await _result(await training.forward_async([datum], loss_fn="cross_entropy"), timeout_seconds)
    return [float(value) for value in result.loss_fn_outputs[0]["logprobs"].tolist()]


async def _sampler_logprobs(sampling, tokens: list[int], timeout_seconds: float) -> list[float]:
    result = await asyncio.wait_for(
        sampling.sample_async(
            prompt=types.ModelInput.from_ints(tokens),
            num_samples=1,
            sampling_params=types.SamplingParams(max_tokens=1, temperature=0.0, seed=17),
            include_prompt_logprobs=True,
        ),
        timeout=timeout_seconds,
    )
    prompt_logprobs = result.prompt_logprobs
    assert prompt_logprobs is not None and len(prompt_logprobs) == len(tokens)
    assert prompt_logprobs[0] is None and all(value is not None for value in prompt_logprobs[1:])
    return [float(value) for value in prompt_logprobs[1:]]


async def _publish_sampler(service, training, name: str, timeout_seconds: float):
    saved = await _result(await training.save_weights_for_sampler_async(name=name), timeout_seconds)
    sampler = await asyncio.wait_for(service.create_sampling_client_async(model_path=saved.path), timeout=timeout_seconds)
    return saved.path, sampler


def _print_table(rows: list[tuple[str, str, DeltaStats]]) -> None:
    print("phase | comparison | tokens | mean | p50 | p90 | p99 | max")
    print("--- | --- | ---: | ---: | ---: | ---: | ---: | ---:")
    for phase, comparison, stats in rows:
        print(f"{phase} | {comparison} | {stats.tokens} | {stats.mean:.6f} | {stats.p50:.6f} | {stats.p90:.6f} | {stats.p99:.6f} | {stats.maximum:.6f}")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_path = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(payload, output, indent=2)
            output.write("\n")
        os.replace(temporary_path, path)
    except Exception:
        Path(temporary_path).unlink(missing_ok=True)
        raise


def _phase_failures(args: argparse.Namespace, phases: dict[str, DeltaStats]) -> list[str]:
    failures = []
    initial = phases["pre_step_match"]
    trainer_change = phases["trainer_change"]
    stale_change = phases["stale_sampler_change"]
    stale_mismatch = phases["stale_mismatch"]
    resynced = phases["resynced_match"]
    if initial.p90 > args.match_p90:
        failures.append(f"pre-step p90 {initial.p90:.6f} exceeds {args.match_p90:.6f}")
    if trainer_change.p90 < args.minimum_step_change:
        failures.append(f"optimizer trainer movement {trainer_change.p90:.6f} is below {args.minimum_step_change:.6f}")
    if stale_change.maximum > args.maximum_stale_sampler_change:
        failures.append(f"stale sampler movement {stale_change.maximum:.6f} exceeds {args.maximum_stale_sampler_change:.6f}")
    if stale_mismatch.p90 < initial.p90 + args.minimum_mismatch_growth:
        failures.append(f"stale mismatch grew from {initial.p90:.6f} to only {stale_mismatch.p90:.6f}")
    if resynced.p90 > args.match_p90:
        failures.append(f"resynced p90 {resynced.p90:.6f} exceeds {args.match_p90:.6f}")
    if resynced.p90 > stale_mismatch.p90 * args.maximum_resync_ratio:
        failures.append(f"resync reduced p90 from {stale_mismatch.p90:.6f} to only {resynced.p90:.6f}")
    return failures


async def run(args: argparse.Namespace) -> None:
    service = tinker.ServiceClient(base_url=args.base_url, api_key=args.api_key)
    training = await asyncio.wait_for(
        service.create_lora_training_client_async(
            base_model=args.base_model,
            rank=args.lora_rank,
            train_mlp=True,
            train_attn=False,
            train_unembed=False,
        ),
        timeout=args.timeout_seconds,
    )
    sampler_path, sampler = await _publish_sampler(service, training, "before-step-1", args.timeout_seconds)
    if args.tokenizer_path is None:
        tokenizer = training.get_tokenizer()
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, local_files_only=True)
    completion_tokens = await _sample_completion(
        sampler, tokenizer, args.completion_tokens, args.timeout_seconds
    )
    tokens, weights, datum = _build_datum(tokenizer, completion_tokens)
    trainer_logprobs = await _trainer_logprobs(training, datum, args.timeout_seconds)
    sampler_logprobs = await _sampler_logprobs(sampler, tokens, args.timeout_seconds)

    step_receipts = []
    all_failures = []
    for step in range(1, args.steps + 1):
        pre_step_train = trainer_logprobs
        pre_step_sample = sampler_logprobs
        pre_step_path = sampler_path

        forward_backward = await training.forward_backward_async([datum], loss_fn="cross_entropy")
        forward_backward_result = await _result(forward_backward, args.timeout_seconds)
        optimizer = await training.optim_step_async(types.AdamParams(learning_rate=args.learning_rate, weight_decay=0.0))
        optimizer_result = await _result(optimizer, args.timeout_seconds)
        updated_train = await _trainer_logprobs(training, datum, args.timeout_seconds)
        stale_sample = await _sampler_logprobs(sampler, tokens, args.timeout_seconds)
        updated_path, updated_sampler = await _publish_sampler(service, training, f"after-step-{step}", args.timeout_seconds)
        assert updated_path != pre_step_path, "weight sync must publish a new immutable sampler path"
        updated_sample = await _sampler_logprobs(updated_sampler, tokens, args.timeout_seconds)

        phases = {
            "pre_step_match": _delta_stats(pre_step_train, pre_step_sample, weights),
            "trainer_change": _delta_stats(pre_step_train, updated_train, weights),
            "stale_sampler_change": _delta_stats(pre_step_sample, stale_sample, weights),
            "stale_mismatch": _delta_stats(updated_train, stale_sample, weights),
            "resynced_match": _delta_stats(updated_train, updated_sample, weights),
            "adapter_effect_match": _delta_stats(
                [updated - initial for initial, updated in zip(pre_step_train, updated_train, strict=True)],
                [updated - initial for initial, updated in zip(pre_step_sample, updated_sample, strict=True)],
                weights,
            ),
        }
        failures = _phase_failures(args, phases)
        all_failures.extend(f"step {step}: {failure}" for failure in failures)
        step_receipts.append(
            {
                "step": step,
                "passed": not failures,
                "forward_backward_metrics": forward_backward_result.metrics,
                "optimizer_metrics": optimizer_result.metrics,
                "sampler_paths": {"before_step": pre_step_path, "after_step": updated_path},
                "phases": {name: asdict(stats) for name, stats in phases.items()},
                "logprobs": {
                    "pre_step_trainer": pre_step_train,
                    "pre_step_sampler": pre_step_sample,
                    "updated_trainer": updated_train,
                    "stale_sampler": stale_sample,
                    "updated_sampler": updated_sample,
                },
                "failures": failures,
            }
        )
        _print_table(
            [
                (str(step), "pre-step trainer vs sampler", phases["pre_step_match"]),
                (str(step), "trainer movement after optimizer", phases["trainer_change"]),
                (str(step), "sampler before vs stale sampler", phases["stale_sampler_change"]),
                (str(step), "updated trainer vs stale sampler", phases["stale_mismatch"]),
                (str(step), "updated trainer vs synced sampler", phases["resynced_match"]),
                (str(step), "trainer vs sampler adapter effect", phases["adapter_effect_match"]),
            ]
        )
        print(f"step {step} | immutable weight sync | {pre_step_path} -> {updated_path}")
        if failures:
            break
        trainer_logprobs = updated_train
        sampler_logprobs = updated_sample
        sampler_path = updated_path
        sampler = updated_sampler

    payload = {
        "passed": len(step_receipts) == args.steps and not all_failures,
        "base_url": args.base_url,
        "base_model": args.base_model,
        "lora_rank": args.lora_rank,
        "learning_rate": args.learning_rate,
        "requested_steps": args.steps,
        "completed_steps": len(step_receipts),
        "fixed_completion_tokens": completion_tokens,
        "step_receipts": step_receipts,
        "failures": all_failures,
    }
    if args.output is not None:
        _write_json(args.output, payload)
        print(f"wrote {args.output}")
    if all_failures:
        raise AssertionError("; ".join(all_failures))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--api-key", default="tml-hersh-weight-sync")
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--completion-tokens", type=int, default=256)
    parser.add_argument("--tokenizer-path")
    parser.add_argument("--timeout-seconds", type=float, default=1200)
    parser.add_argument("--match-p90", type=float, default=0.1)
    parser.add_argument("--minimum-step-change", type=float, default=0.02)
    parser.add_argument("--maximum-stale-sampler-change", type=float, default=1e-4)
    parser.add_argument("--minimum-mismatch-growth", type=float, default=0.02)
    parser.add_argument("--maximum-resync-ratio", type=float, default=0.5)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(run(_parse_args()))
