"""Train Qwen3.5 MTP through the Tinker SDK using one DAPO batch per update.

Launch the gateway in Bridge mode with a local dense Qwen3.5 checkpoint, one
training GPU, dynamic batching, target modules attn,mlp, and
--enable-mtp-training --mtp-num-layers 1 --mtp-loss-scaling-factor 0.2.
Set --rollout-temperature 0.9 on the gateway so the trainer scores the
policy at the same temperature as this client's sampler.
Point --base-model at the gateway's model name and --tokenizer at the local
checkpoint. --dataset accepts DAPO JSONL rows with prompt and label fields.
--resume takes the prior checkpoint URI and --start-step takes its update count.
--spec-eval-interval records snapshots for eval_mtp_checkpoints.py to evaluate
with NEXTN. Use 1 for every update or 50 for a lighter evaluation cadence.
"""

import argparse
import asyncio
import json
import math
import random
import statistics
import time
from itertools import islice
from pathlib import Path

from transformers import AutoTokenizer

import tinker
from miles.rollout.rm_hub.math_dapo_utils import compute_score
from miles.utils.file_utils import atomic_write_text
from tinker import types

_ROLLOUT_TEMPERATURE = 0.9


async def _rollout(sampler, tokenizer, row, args):
    prompt_text = tokenizer.apply_chat_template(
        row["prompt"], tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    prompt = tokenizer.encode(prompt_text, add_special_tokens=False)
    result = await sampler.sample_async(
        prompt=types.ModelInput.from_ints(prompt),
        num_samples=args.samples_per_prompt,
        sampling_params=types.SamplingParams(
            max_tokens=args.response_length,
            temperature=_ROLLOUT_TEMPERATURE,
            top_p=1.0,
            seed=args.seed if args.seed_sampling else None,
        ),
    )
    return prompt, result.sequences


def _build_batch(rows, rollouts, tokenizer):
    datums, lengths, rewards, truncated = [], [], [], []
    batch_size = sum(len(sequences) for _, sequences in rollouts)
    for row, (prompt, sequences) in zip(rows, rollouts, strict=True):
        scores = [compute_score(tokenizer.decode(seq.tokens), row["label"])["score"] for seq in sequences]
        mean = statistics.mean(scores)
        std = statistics.stdev(scores) if len(scores) > 1 else 0.0
        for sequence, reward in zip(sequences, scores, strict=True):
            response = sequence.tokens
            assert response, "received an empty rollout response"
            advantage = (reward - mean) / (std + 1e-6)
            tokens = prompt + response
            prompt_targets = len(prompt) - 1
            datums.append(
                types.Datum(
                    model_input=types.ModelInput.from_ints(tokens[:-1]),
                    loss_fn_inputs={
                        "target_tokens": tokens[1:],
                        "weights": [0.0] * prompt_targets + [advantage / (len(response) * batch_size)] * len(response),
                        "mtp_mask": types.TensorData(
                            dtype="int64",
                            shape=[len(tokens) - 1],
                            data=[0] * prompt_targets + [1] * len(response),
                        ),
                    },
                )
            )
            lengths.append(len(response))
            rewards.append(reward)
            truncated.append(sequence.stop_reason == "length")
    return datums, {
        "reward": statistics.mean(rewards),
        "response_length": statistics.mean(lengths),
        "truncated_ratio": statistics.mean(truncated),
    }


def _train_rollout_logprob_gap(outputs, rollouts):
    sequences = (sequence for _, group in rollouts for sequence in group)
    per_sample = []
    for output, sequence in zip(outputs, sequences, strict=True):
        train_logprobs = output["logprobs"].data[-len(sequence.tokens) :]
        per_sample.append(
            statistics.fmean(
                abs(train - sampled) for train, sampled in zip(train_logprobs, sequence.logprobs, strict=True)
            )
        )
    return statistics.fmean(per_sample)


async def _export_sampler(service, trainer, name):
    saved = await (await trainer.save_weights_for_sampler_async(name=name))
    return await service.create_sampling_client_async(model_path=saved.path)


async def _save_checkpoint(trainer, step):
    saved = await (await trainer.save_state_async(f"step-{step:04d}"))
    return saved.path


def _shuffled_epochs(rows, seed):
    epoch = 0
    while True:
        shuffled = list(rows)
        random.Random(seed + epoch).shuffle(shuffled)
        yield from shuffled
        epoch += 1


async def _train(args):
    rows = [json.loads(line) for line in args.dataset.read_text().splitlines()]
    assert rows, "expected a nonempty DAPO dataset"
    stop_step = args.start_step + args.steps
    prompts = islice(_shuffled_epochs(rows, args.seed), args.start_step * args.prompts_per_step, None)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    service = tinker.ServiceClient(base_url=args.base_url, api_key=args.api_key)
    if args.resume:
        trainer = await service.create_training_client_from_state_with_optimizer_async(args.resume)
        sampler = await _export_sampler(service, trainer, f"resume-{args.start_step:04d}")
    else:
        trainer = await service.create_lora_training_client_async(
            base_model=args.base_model, rank=args.rank, train_attn=True, train_mlp=True, train_unembed=False
        )
        sampler = await service.create_sampling_client_async(base_model=args.base_model)
    adam = types.AdamParams(
        learning_rate=args.lr, beta1=0.9, beta2=0.98, eps=1e-8, weight_decay=0.1, grad_clip_norm=1.0
    )
    mode = "a" if args.resume else "x"
    checkpoint = None
    with args.metrics.open(mode) as metrics:
        previous_update_at = time.perf_counter()
        for i in range(args.start_step, stop_step):
            step = i + 1
            step_rows = list(islice(prompts, args.prompts_per_step))
            rollout_started_at = time.perf_counter()
            rollouts = await asyncio.gather(*(_rollout(sampler, tokenizer, row, args) for row in step_rows))
            rollout_seconds = time.perf_counter() - rollout_started_at
            datums, rollout_metrics = _build_batch(step_rows, rollouts, tokenizer)
            backward_started_at = time.perf_counter()
            backward = await (await trainer.forward_backward_async(datums, "cross_entropy"))
            backward_seconds = time.perf_counter() - backward_started_at
            scoring_temperature = backward.metrics.get("policy_scoring_temperature:mean")
            if scoring_temperature is not None and not math.isclose(
                scoring_temperature, _ROLLOUT_TEMPERATURE, rel_tol=1e-6
            ):
                raise RuntimeError(
                    f"The trainer scores at temperature {scoring_temperature:g}. "
                    f"Set --rollout-temperature {_ROLLOUT_TEMPERATURE:g} on the gateway to match this recipe."
                )
            mtp_loss = backward.metrics["mtp_loss_microbatch_sum:sum"] / backward.metrics["mtp_microbatches:sum"]
            assert math.isfinite(mtp_loss), f"nonfinite MTP loss at step {step}"
            logprob_gap = _train_rollout_logprob_gap(backward.loss_fn_outputs, rollouts)
            optimizer_started_at = time.perf_counter()
            update = await (await trainer.optim_step_async(adam))
            assert update.metrics["mtp_grad_norm"] > 0, f"no MTP update at step {step}"
            updated_at = time.perf_counter()
            generated_tokens = sum(len(seq.tokens) for _, sequences in rollouts for seq in sequences)
            record = {
                "step": step,
                "mtp_loss": mtp_loss,
                "policy_surrogate_loss": backward.metrics["loss:sum"],
                "train_rollout_logprob_abs_diff": logprob_gap,
                "sampling_temperature": _ROLLOUT_TEMPERATURE,
                "policy_scoring_temperature": scoring_temperature,
                "rollout_seconds": rollout_seconds,
                "generated_response_tokens": generated_tokens,
                "rollout_tokens_per_second": generated_tokens / rollout_seconds,
                "forward_backward_seconds": backward_seconds,
                "optimizer_seconds": updated_at - optimizer_started_at,
                "training_seconds": updated_at - backward_started_at,
                "wall_update_seconds": updated_at - previous_update_at,
                **rollout_metrics,
                **update.metrics,
            }
            previous_update_at = updated_at
            spec_eval = args.spec_eval_interval > 0 and step % args.spec_eval_interval == 0
            if spec_eval or step == stop_step:
                save_started_at = time.perf_counter()
                checkpoint = await _save_checkpoint(trainer, step)
                record["checkpoint_save_seconds"] = time.perf_counter() - save_started_at
                if spec_eval:
                    record["spec_eval_checkpoint"] = checkpoint
            print(json.dumps(record), file=metrics, flush=True)
            print(json.dumps(record), flush=True)
            if step < stop_step:
                sampler = await _export_sampler(service, trainer, f"update-{step:04d}")

    if checkpoint is None:
        checkpoint = await _save_checkpoint(trainer, stop_step)
    manifest = json.dumps({"checkpoint": checkpoint, "steps": stop_step, "rank": args.rank, "lr": args.lr})
    await service.close("success")
    atomic_write_text(args.metrics.with_suffix(".checkpoint.json"), manifest + "\n")
    print(manifest, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:10613")
    parser.add_argument("--api-key", default="tml-mtp-smoke")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument(
        "--seed-sampling",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Set a request seed on each rollout. Turn this off to use the sampler RNG, as in direct Miles.",
    )
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--prompts-per-step", type=int, default=8)
    parser.add_argument("--samples-per-prompt", type=int, default=8)
    parser.add_argument("--response-length", type=int, default=1024)
    parser.add_argument(
        "--spec-eval-interval",
        type=int,
        default=0,
        help="Save an evaluation snapshot every N updates. Use 0 to disable evaluation snapshots.",
    )
    parser.add_argument("--resume")
    parser.add_argument("--start-step", type=int, default=0)
    args = parser.parse_args()
    if args.spec_eval_interval < 0:
        parser.error("--spec-eval-interval must be zero or positive")
    asyncio.run(_train(args))
