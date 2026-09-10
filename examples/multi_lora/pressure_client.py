"""Concurrent DAPO clients using the async SDK pattern in multi_lora/client.py.

Each client owns a model, optimizer stream, sampling versions and W&B run.
``--steps 0`` trains until interrupted; finite runs require every client to
complete every requested optimizer step. Context includes prompt and response.
"""

import argparse
import asyncio
import contextlib
import json
import math
import os
import random
import shutil
import statistics
import sys
import time
import traceback
from pathlib import Path

from examples.multi_lora.pressure_telemetry import JournalRun, start_uploader
from examples.multi_lora.pressure_timing import StepTimings
from transformers import AutoTokenizer

import tinker
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
    assert len(examples) >= args.clients * args.max_prompt_groups
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
        # lists for every client can consume tens of GB while waiting at a barrier.
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


def _prune_exports(paths, *, checkpoint_root, model_id, kind="sampler_weights"):
    """Only this client's completed, older sampler exports; keep two versions.

    Called after all samples using the previous version have completed. This
    harness never hands its sampler paths to other clients. Saved training
    checkpoints and any pre-existing exports are outside this ownership list.
    """
    if checkpoint_root is None:
        return
    assert kind in ("sampler_weights", "weights")
    root = (checkpoint_root / model_id / kind).resolve()
    for uri in paths[:-2]:
        prefix = f"tinker://{model_id}/{kind}/"
        assert uri.startswith(prefix)
        version = uri.removeprefix(prefix)
        assert (
            version.isdigit() if kind == "sampler_weights" else version.startswith("step-") and version[5:].isdigit()
        )
        path = root / version
        assert not path.is_symlink() and path.resolve().parent == root
        if path.exists():
            shutil.rmtree(path)
    del paths[:-2]


async def _run_client(index, args, examples, tokenizer, barrier):
    tag = f"lora_{index:03d}"
    run = JournalRun(
        args.output_dir,
        tag,
        entity=args.wandb_entity,
        project=args.wandb_project,
        group=args.run_id,
        name=f"{args.phase}-n{args.clients}-{tag}",
        config={
            **vars(args),
            "client_index": index,
            "context_length": CONTEXT,
            "client_pattern": "examples/multi_lora/client.py",
            "loss": "dapo_ppo",
        },
    )
    service = tinker.ServiceClient(base_url=args.base_url, api_key=f"tml-{args.run_id}-{index}")
    shard = examples[index :: args.clients]
    adam = types.AdamParams(
        learning_rate=args.lr, beta1=0.9, beta2=0.98, eps=1e-8, weight_decay=0.1, grad_clip_norm=1.0
    )
    exported = []
    checkpoints = []
    cursor, step = 0, 0

    def progress(stage, **values):
        _write_json(
            args.output_dir / f"{tag}-phase.json",
            {"adapter": tag, "step": step, "stage": stage, "time": time.time(), **values},
        )

    try:
        progress("create_model")
        training = await service.create_lora_training_client_async(
            base_model=args.model, rank=args.lora_rank, train_unembed=False
        )
        run.update_summary({"model_id": training.model_id})
        timing = (
            StepTimings(
                args.output_dir, adapter=tag, model_id=training.model_id, phase=args.phase, clients=args.clients
            )
            if args.phase == "continuous"
            else None
        )
        progress("models_ready_barrier")
        await barrier.wait()
        while args.steps == 0 or step < args.steps:
            # A common boundary keeps all N adapters active in each trial.
            await barrier.wait()
            progress("publication")
            started = time.monotonic()
            save_future = await training.save_weights_for_sampler_async(name=f"step-{step:06d}")
            saved = await save_future
            exported.append(saved.path)
            sampler = await service.create_sampling_client_async(model_path=saved.path)
            _prune_exports(exported, checkpoint_root=args.checkpoint_root, model_id=training.model_id)
            published_at = time.monotonic()
            progress("sampling", attempts=0, accepted_groups=0)
            datums, rows, observed, attempts, cursor = await _collect_batch(
                sampler, shard, cursor, tokenizer, args, progress
            )
            sampled_at = time.monotonic()
            sampling_seconds = sampled_at - started
            progress("samples_ready_barrier", attempts=attempts)
            await barrier.wait()
            training_at = time.monotonic()
            progress("forward_backward_submit")
            fb_future = await training.forward_backward_async(datums, loss_fn="ppo", loss_fn_config=CLIP)
            progress("optimizer_submit")
            optim_future = await training.optim_step_async(adam)
            progress("forward_backward_wait")
            fb = await fb_future
            fb_done_at = time.monotonic()
            progress("optimizer_wait")
            opt = await optim_future
            finished_at = time.monotonic()
            step += 1
            stages = {
                "step_seconds": finished_at - started,
                "publication_seconds": published_at - started,
                "rollout_and_batch_seconds": sampled_at - published_at,
                "client_barrier_seconds": training_at - sampled_at,
                "forward_backward_seconds": fb_done_at - training_at,
                "optimizer_seconds": finished_at - fb_done_at,
            }
            metrics = _metrics(fb, opt, rows, observed, attempts, sampling_seconds, finished_at - started)
            distributions = {}
            if timing is not None:
                summary = timing.add(step, stages)
                metrics.update({f"time/{key}": value for key, value in stages.items()})
                metrics["time/mean_step_seconds"] = summary["mean_seconds"]
                run.update_summary({"timing": summary})
                distributions["time/step_seconds_distribution"] = timing.seconds
            record = {"adapter": tag, "model_id": training.model_id, "step": step, "metrics": metrics}
            print(json.dumps(record, allow_nan=False), flush=True)
            _write_json(args.output_dir / f"{tag}-progress.json", record)
            progress("step_complete")
            run.log(metrics, step=step, histogram=distributions)
            if args.checkpoint_every and step % args.checkpoint_every == 0:
                checkpoint_future = await training.save_state_async(name=f"step-{step:06d}")
                checkpoint = await checkpoint_future
                run.update_summary({"latest_checkpoint": checkpoint.path})
                checkpoints.append(checkpoint.path)
                _prune_exports(
                    checkpoints, checkpoint_root=args.checkpoint_root, model_id=training.model_id, kind="weights"
                )
        run.update_summary({"completed_steps": step})
        run.finish()
        return {"adapter": tag, "model_id": training.model_id, "steps": step}
    except BaseException:
        await _record_failure(args.output_dir, tag, step, barrier)
        with contextlib.suppress(Exception):
            run.update_summary({"completed_steps": step, "status": "failed"})
            run.finish(exit_code=1)
        raise


async def _record_failure(directory, tag, step, barrier):
    error = traceback.format_exc()
    try:
        with contextlib.suppress(Exception):
            _write_json(directory / f"{tag}-error.json", {"adapter": tag, "completed_steps": step, "error": error})
        with contextlib.suppress(Exception):
            print(f"{tag} failed after {step} steps:\n{error}", file=sys.stderr, flush=True)
    finally:
        # Even failed local reporting must release peers. No network SDK cleanup
        # is allowed before propagating the original training exception.
        await barrier.abort()


def _metrics(fb, opt, rows, observed, attempts, sampling_seconds, elapsed):
    loss = float(fb.metrics["loss:sum"])
    assert math.isfinite(loss)
    assert not any("skipped" in key or "error" in key for key in opt.metrics), opt.metrics
    for value in opt.metrics.values():
        if isinstance(value, (int, float)):
            assert math.isfinite(value), opt.metrics
    grad_norm = next((float(v) for k, v in opt.metrics.items() if "grad_norm" in k), None)
    if grad_norm is not None:
        assert grad_norm > 0, "mixed rewards produced zero gradient"
    trained = sum(len(row["tokens"]) for row in rows)
    delta = 0.0
    for row, output in zip(rows, fb.loss_fn_outputs, strict=True):
        current = output["logprobs"].data[len(row["prompt"]) - 1 :]
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


async def _main(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.wandb and not args.telemetry_managed:
        start_uploader(args.output_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    examples = _load_dataset(args, tokenizer)
    barrier = asyncio.Barrier(args.clients)
    try:
        async with asyncio.TaskGroup() as tasks:
            running = [
                tasks.create_task(_run_client(i, args, examples, tokenizer, barrier)) for i in range(args.clients)
            ]
        results = [task.result() for task in running]
        assert len({result["model_id"] for result in results}) == args.clients
        _write_json(args.output_dir / "result.json", {"status": "passed", "clients": results})
    except BaseException:
        error = traceback.format_exc()
        _write_json(args.output_dir / "result.json", {"status": "failed", "error": error})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:10639")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--clients", type=int, required=True)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--prompt-groups", type=int, default=4)
    parser.add_argument("--samples-per-prompt", type=int, default=8)
    parser.add_argument("--max-prompt-groups", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--phase", choices=["search", "continuous"], default="search")
    parser.add_argument("--telemetry-managed", action="store_true", help="Supervisor owns the separate W&B uploader")
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--wandb", action="store_true", help="Upload local metrics to W&B (disabled by default)")
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "miles-pressure-test"))
    parsed = parser.parse_args()
    assert parsed.clients > 0 and parsed.steps >= 0 and parsed.samples_per_prompt >= 2
    asyncio.run(_main(parsed))
