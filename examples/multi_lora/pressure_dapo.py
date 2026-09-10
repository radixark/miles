"""DAPO data preparation and result checks shared by the standalone SDK user."""

import json
import math
import random
import statistics

from miles.rollout.rm_hub.math_dapo_utils import compute_score
from tinker import types

CONTEXT = 8192
CLIP = {"clip_low_threshold": 0.8, "clip_high_threshold": 1.28}


def _advantages(rewards):
    mean, std = statistics.mean(rewards), statistics.stdev(rewards)
    return [(reward - mean) / (std + 1e-6) for reward in rewards]


def _datum(row, advantage, response_tokens):
    tokens = row["prompt"] + row["tokens"]
    assert len(tokens) <= CONTEXT
    prefix = len(row["prompt"]) - 1
    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens[:-1]),
        loss_fn_inputs={
            "target_tokens": tokens[1:],
            "logprobs": [0.0] * prefix + row["logprobs"],
            "advantages": [0.0] * prefix + [advantage / response_tokens] * len(row["tokens"]),
        },
    )


def _write_json(path, record):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _load_dataset(args, tokenizer):
    examples = []
    with args.dataset.open() as stream:
        for line in stream:
            row = json.loads(line)
            messages = row["prompt"]
            if isinstance(messages, str):
                messages = [{"role": "user", "content": messages}]
            tokens = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_dict=False
            )
            if 0 < len(tokens) <= 2048:
                examples.append({"tokens": list(tokens), "label": str(row["label"])})
    random.Random(args.seed).shuffle(examples)
    assert len(examples) >= args.clients * args.max_prompt_groups, "not enough prompt groups per user"
    return examples


async def _sample_group(sampler, example, tokenizer, args):
    budget = CONTEXT - len(example["tokens"])
    result = await sampler.sample_async(
        prompt=types.ModelInput.from_ints(example["tokens"]),
        num_samples=args.samples_per_prompt,
        sampling_params=types.SamplingParams(max_tokens=budget, temperature=1.0, top_p=1.0),
    )
    rows = []
    assert len(result.sequences) == args.samples_per_prompt
    for sequence in result.sequences:
        tokens, logprobs = list(sequence.tokens), list(sequence.logprobs or [])
        assert tokens and len(tokens) == len(logprobs) and len(tokens) <= budget
        assert all(math.isfinite(value) for value in logprobs)
        score = compute_score(tokenizer.decode(tokens, skip_special_tokens=True), example["label"])
        penalty = min(-(len(tokens) - (budget - 1024)) / 1024, 0.0)
        rows.append(
            dict(
                prompt=example["tokens"],
                tokens=tokens,
                logprobs=logprobs,
                accuracy=float(score["acc"]),
                reward=float(score["score"]) + penalty,
                penalty=penalty,
                truncated=sequence.stop_reason == "length",
            )
        )
    return rows


async def _collect_batch(sampler, examples, cursor, tokenizer, args, progress):
    accepted, observed = [], []
    attempts = 0
    while len(accepted) < args.prompt_groups and attempts < args.max_prompt_groups:
        rows = await _sample_group(sampler, examples[cursor % len(examples)], tokenizer, args)
        cursor += 1
        attempts += 1
        # Rejected groups need only scalar metrics. Keeping their token/logprob
        # lists for every client can consume tens of GB.
        observed.extend(
            {
                **{key: row[key] for key in ("accuracy", "reward", "penalty", "truncated")},
                "response_tokens": len(row["tokens"]),
                "total_tokens": len(row["prompt"]) + len(row["tokens"]),
            }
            for row in rows
        )
        if len({row["accuracy"] for row in rows}) > 1:
            accepted.append(rows)
        progress("sampling", attempts=attempts, accepted_groups=len(accepted))
    if len(accepted) != args.prompt_groups:
        raise RuntimeError(f"insufficient nonconstant DAPO groups: {len(accepted)}/{args.prompt_groups}; not an OOM")
    rows = [row for group in accepted for row in group]
    response_tokens = sum(len(row["tokens"]) for row in rows)
    datums = [
        _datum(row, advantage, response_tokens)
        for group in accepted
        for row, advantage in zip(group, _advantages([row["reward"] for row in group]), strict=True)
    ]
    return datums, rows, observed, attempts, cursor


def _metrics(fb, opt, rows, observed, attempts, sampling_seconds, elapsed):
    loss = float(fb.metrics["loss:sum"])
    assert math.isfinite(loss)
    assert not any("skipped" in key or "error" in key for key in opt.metrics), opt.metrics
    for value in opt.metrics.values():
        if isinstance(value, (int, float)):
            assert math.isfinite(value), opt.metrics
    grad_norm = next((float(v) for k, v in opt.metrics.items() if "grad_norm" in k), None)
    assert grad_norm is not None and grad_norm > 0, "optimizer did not report a positive gradient norm"
    trained = sum(len(row["tokens"]) for row in rows)
    delta = 0.0
    for row, output in zip(rows, fb.loss_fn_outputs, strict=True):
        current = output["logprobs"].data[len(row["prompt"]) - 1 :]
        assert all(math.isfinite(value) for value in current), "nonfinite training logprobs"
        delta += sum(abs(a - b) for a, b in zip(current, row["logprobs"], strict=True))
    return {
        "train/loss": loss,
        **({"train/grad_norm": grad_norm} if grad_norm is not None else {}),
        "reward/mean": statistics.mean(row["reward"] for row in observed),
        "reward/accuracy": statistics.mean(row["accuracy"] for row in observed),
        "reward/overlength_penalty": statistics.mean(row["penalty"] for row in observed),
        "rollout/response_tokens": statistics.mean(row["response_tokens"] for row in observed),
        "rollout/max_total_tokens": max(row["total_tokens"] for row in observed),
        "rollout/truncated_fraction": statistics.mean(float(row["truncated"]) for row in observed),
        "rollout/dynamic_sampling_attempts": attempts,
        "throughput/generated_tokens_per_second": sum(row["response_tokens"] for row in observed) / sampling_seconds,
        "time/step_seconds": elapsed,
        "time/sampling_and_publication_seconds": sampling_seconds,
        "train/response_tokens": trained,
        "train/rollout_logprob_absdiff": delta / trained,
    }
