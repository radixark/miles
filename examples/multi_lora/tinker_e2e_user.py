"""One independent Tinker SDK user: publish, DAPO rollout, PPO backward, Adam.

Follows tinker-cookbook/recipes/rl_loop.py's SDK sequence, adapted to DAPO
dynamic sampling and an 8192-token prompt-plus-response budget. No mock backend.

Example (run from the repository root against an already running gateway):
    python -m examples.multi_lora.tinker_e2e_user --model /models/Qwen3-30B-A3B \
        --dataset /datasets/dapo-math-17k.jsonl --output-dir /shared/one-user
"""

import argparse
import asyncio
import contextlib
import os
import time
import traceback
import uuid
from pathlib import Path

from examples.multi_lora.pressure_dapo import (
    CLIP,
    CONTEXT,
    _collect_batch,
    _load_dataset,
    _metrics,
    _sample_group,
    _write_json,
)
from examples.multi_lora.pressure_telemetry import JournalRun, start_uploader
from examples.multi_lora.pressure_timing import StepTimings
from transformers import AutoTokenizer

import tinker
from tinker import types


async def _publish(training, service, step):
    future = await training.save_weights_for_sampler_async(name=f"step-{step:06d}")
    saved = await future
    sampler = await service.create_sampling_client_async(model_path=saved.path)
    return saved.path, sampler


async def _train_step(training, service, adam, *, step, shard, cursor, tokenizer, args, progress):
    started = time.monotonic()
    progress("publication")
    version, sampler = await _publish(training, service, step)
    published_at = time.monotonic()
    progress("sampling", sampler_path=version)
    datums, rows, observed, attempts, cursor = await _collect_batch(sampler, shard, cursor, tokenizer, args, progress)
    sampled_at = time.monotonic()
    # This user trains as soon as its own batch is ready. No cross-user step barrier.
    progress("forward_backward_submit")
    fb_future = await training.forward_backward_async(datums, loss_fn="ppo", loss_fn_config=CLIP)
    progress("optimizer_submit")
    optim_future = await training.optim_step_async(adam)
    submitted_at = time.monotonic()
    progress("forward_backward_wait")
    fb = await fb_future
    fb_done_at = time.monotonic()
    progress("optimizer_wait")
    opt = await optim_future
    finished_at = time.monotonic()
    stages = {
        "step_seconds": finished_at - started,
        "publication_seconds": published_at - started,
        "rollout_and_batch_seconds": sampled_at - published_at,
        "training_submit_seconds": submitted_at - sampled_at,
        "forward_backward_wait_seconds": fb_done_at - submitted_at,
        "optimizer_wait_seconds": finished_at - fb_done_at,
    }
    # SDK timings include server queueing, execution and transport. These are
    # not separately measured GPU kernel times.
    metrics = _metrics(fb, opt, rows, observed, attempts, sampled_at - started, finished_at - started)
    metrics.update({f"time/{key}": value for key, value in stages.items()})
    return cursor, metrics, stages, version


async def run_user(index, args, examples, tokenizer, *, ready=None, finished=None):
    """Same workload for the one-user CLI and concurrent-user launcher."""
    tag = f"lora_{index:03d}"
    run = JournalRun(
        args.output_dir,
        tag,
        entity=args.wandb_entity,
        project=args.wandb_project,
        group=args.run_id,
        name=tag,
        config={**vars(args), "client_index": index, "context_length": CONTEXT},
    )
    service = tinker.ServiceClient(base_url=args.base_url, api_key=f"tml-pressure-{args.run_id}-{index}")
    shard = examples[index :: args.clients]
    adam = types.AdamParams(
        learning_rate=args.lr, beta1=0.9, beta2=0.98, eps=1e-8, weight_decay=0.1, grad_clip_norm=1.0
    )
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
        timing = StepTimings(
            args.output_dir, adapter=tag, model_id=training.model_id, phase="e2e", clients=args.clients
        )
        if ready is not None:
            progress("all_users_created")
            await ready.wait()  # All N distinct models exist before the first rollout.
        while args.steps == 0 or step < args.steps:
            cursor, metrics, stages, version = await _train_step(
                training,
                service,
                adam,
                step=step,
                shard=shard,
                cursor=cursor,
                tokenizer=tokenizer,
                args=args,
                progress=progress,
            )
            step += 1
            summary = timing.add(step, stages)
            record = {
                "adapter": tag,
                "model_id": training.model_id,
                "step": step,
                "sampler_path": version,
                "metrics": metrics,
            }
            _write_json(args.output_dir / f"{tag}-progress.json", record)
            progress("step_complete")
            print(f"{tag}: step {step} complete, {stages['step_seconds']:.2f}s", flush=True)
            run.log(metrics, step=step, histogram={"time/step_seconds_distribution": timing.seconds})
            run.update_summary({"completed_steps": step, "timing": summary})
        # Exercise publication and generation with the final optimizer update too.
        progress("final_publication")
        final_path, sampler = await _publish(training, service, step)
        progress("final_sampling", sampler_path=final_path)
        await _sample_group(sampler, shard[cursor % len(shard)], tokenizer, args)
        result = {"adapter": tag, "model_id": training.model_id, "steps": step, "final_sampler_path": final_path}
        _write_json(args.output_dir / f"{tag}-result.json", {"status": "passed", **result})
        if finished is not None:
            progress("peers_finishing")
            await finished.wait()  # Keep SDK sessions/models alive until every user finishes.
        run.finish()
        progress("complete")
        return result
    except BaseException:
        error = traceback.format_exc()
        with contextlib.suppress(Exception):
            _write_json(args.output_dir / f"{tag}-error.json", {"completed_steps": step, "error": error})
            run.finish(exit_code=1)
        raise


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:10639")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=3, help="Optimizer steps per user; 0 runs until interrupted")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--prompt-groups", type=int, default=4)
    parser.add_argument("--samples-per-prompt", type=int, default=8)
    parser.add_argument("--max-prompt-groups", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", default=uuid.uuid4().hex[:12])
    parser.add_argument(
        "--timeout-seconds", type=float, default=14400, help="Whole trial deadline including startup gates"
    )
    parser.add_argument("--wandb", action="store_true", help="Opt in to a separate telemetry uploader")
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "miles-pressure-test"))
    parser.set_defaults(clients=1)
    return parser


def _prepare(args):
    if min(args.clients, args.lora_rank, args.prompt_groups, args.timeout_seconds) <= 0 or args.steps < 0:
        raise ValueError("counts and timeout must be positive; steps may be zero for continuous training")
    if args.samples_per_prompt < 2 or args.max_prompt_groups < args.prompt_groups:
        raise ValueError("DAPO needs at least two samples/group and enough sampling attempts")
    args.output_dir.mkdir(parents=True, exist_ok=False)  # Never reuse a prior trial's results.
    _write_json(
        args.output_dir / "config.json",
        {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    examples = _load_dataset(args, tokenizer)
    if args.wandb:
        start_uploader(args.output_dir)
    return examples, tokenizer


async def _main(args):
    examples, tokenizer = _prepare(args)
    try:
        async with asyncio.timeout(args.timeout_seconds):
            result = await run_user(0, args, examples, tokenizer)
        _write_json(args.output_dir / "result.json", {"status": "passed", "users": [result]})
    except BaseException:
        _write_json(args.output_dir / "result.json", {"status": "failed", "error": traceback.format_exc()})
        raise


if __name__ == "__main__":
    asyncio.run(_main(_parser().parse_args()))
