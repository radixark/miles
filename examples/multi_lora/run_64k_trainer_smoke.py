"""Exercise a near-64K LoRA forward/backward and optimizer step."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import time
from pathlib import Path

import tinker
from tinker import types


async def _result(future, timeout_seconds: float):
    return await asyncio.wait_for(future.result_async(), timeout=timeout_seconds)


async def run(args: argparse.Namespace) -> None:
    service = tinker.ServiceClient(base_url=args.base_url, api_key=args.api_key)
    started = time.monotonic()
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
    if args.token_id is None:
        tokenizer = training.get_tokenizer()
        token = int(tokenizer.encode(" test", add_special_tokens=False)[0])
    else:
        token = args.token_id
    tokens = [token] * args.total_tokens
    trainable_tokens = min(args.trainable_tokens, args.total_tokens - 1)
    weights = [0.0] * (args.total_tokens - 1 - trainable_tokens) + [1.0] * trainable_tokens
    datum = types.Datum(
        model_input=types.ModelInput.from_ints(tokens[:-1]),
        loss_fn_inputs={"target_tokens": tokens[1:], "weights": weights},
    )

    forward_backward_started = time.monotonic()
    forward_backward = await _result(
        await training.forward_backward_async([datum], loss_fn="cross_entropy"),
        args.timeout_seconds,
    )
    forward_backward_seconds = time.monotonic() - forward_backward_started
    optimizer_started = time.monotonic()
    optimizer = await _result(
        await training.optim_step_async(types.AdamParams(learning_rate=args.learning_rate, weight_decay=0.0)),
        args.timeout_seconds,
    )
    optimizer_seconds = time.monotonic() - optimizer_started
    verify = await _result(
        await training.forward_async([datum], loss_fn="cross_entropy"),
        args.timeout_seconds,
    )
    logprobs = [float(value) for value in verify.loss_fn_outputs[0]["logprobs"].tolist()]
    trainable_logprobs = logprobs[-trainable_tokens:]
    assert trainable_logprobs and all(math.isfinite(value) for value in trainable_logprobs)

    payload = {
        "passed": True,
        "base_model": args.base_model,
        "lora_rank": args.lora_rank,
        "total_tokens": args.total_tokens,
        "trainable_tokens": trainable_tokens,
        "forward_backward_seconds": forward_backward_seconds,
        "optimizer_seconds": optimizer_seconds,
        "wall_seconds": time.monotonic() - started,
        "forward_backward_metrics": forward_backward.metrics,
        "optimizer_metrics": optimizer.metrics,
        "verify_logprob_min": min(trainable_logprobs),
        "verify_logprob_max": max(trainable_logprobs),
    }
    print(json.dumps(payload, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--api-key", default="tml-64k-trainer-smoke")
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--total-tokens", type=int, default=64 * 1024 - 32)
    parser.add_argument("--trainable-tokens", type=int, default=32)
    parser.add_argument("--token-id", type=int)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--timeout-seconds", type=float, default=3600)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(run(_parse_args()))
