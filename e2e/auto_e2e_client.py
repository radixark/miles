"""n tenants run DAPO concurrently on the multi-LoRA Tinker gateway.

Every tenant sends the same requests: the same prompts in the same order, the same sampling
parameters, loss and optimizer settings; only its adapter (and therefore its samples) differs.
Every step of every tenant is a dependent chain:
  rollout (G answers per prompt from the tenant's current adapter version)
    -> forward_backward with DAPO's objective (PPO, asymmetric clipping, group-normalized
       token-level advantages, token-mean loss; dynamic sampling keeps mixed-reward groups)
    -> optim_step
    -> save_weights_for_sampler (publish the new version)
    -> the next step's rollout samples from that version.
"""

import argparse
import asyncio
import json
import statistics
import time
import traceback

import tinker
from miles.rollout.rm_hub.math_dapo_utils import compute_score  # DAPO's rule-based math reward: +1 / -1
from tinker import types

DAPO_CLIP = {"clip_low_threshold": 0.8, "clip_high_threshold": 1.28}


def log(tag: str, message: str) -> None:
    print(f"[auto-e2e {time.strftime('%H:%M:%S')} {tag}] {message}", flush=True)


def load_rows(path: str) -> list[tuple[list[dict], str]]:
    rows = []
    with open(path) as handle:
        for line in handle:
            row = json.loads(line)
            rows.append((row["prompt"], str(row["label"])))
    return rows


async def publish(training, service, name: str):
    path = (await (await training.save_weights_for_sampler_async(name=name))).path
    return path, await service.create_sampling_client_async(model_path=path)


async def rollout_group(sampler, tokenizer, prompt_tokens: list[int], label: str, args):
    max_tokens = min(args.max_new_tokens, args.context_len - len(prompt_tokens))
    response = await sampler.sample_async(
        prompt=types.ModelInput.from_ints(prompt_tokens),
        num_samples=args.samples_per_prompt,
        sampling_params=types.SamplingParams(max_tokens=max_tokens, temperature=1.0, top_p=1.0),
    )
    sequences = list(response.sequences)
    rewards = [
        compute_score(tokenizer.decode(list(sequence.tokens), skip_special_tokens=True), label)
        for sequence in sequences
    ]
    return prompt_tokens, sequences, rewards


def build_datums(groups) -> list[types.Datum]:
    """Group-normalized advantages on every response token, scaled to a token mean over the step."""
    total_response_tokens = sum(len(sequence.tokens) for _, sequences, _ in groups for sequence in sequences)
    datums = []
    for prompt_tokens, sequences, rewards in groups:
        mean, std = statistics.fmean(rewards), statistics.pstdev(rewards)
        for sequence, reward in zip(sequences, rewards, strict=True):
            response = list(sequence.tokens)
            advantage = (reward - mean) / (std + 1e-6) / total_response_tokens
            tokens = prompt_tokens + response
            prefix = len(prompt_tokens) - 1  # prompt positions carry no loss: advantage 0
            datums.append(
                types.Datum(
                    model_input=types.ModelInput.from_ints(tokens[:-1]),
                    loss_fn_inputs={
                        "target_tokens": tokens[1:],
                        "logprobs": [0.0] * prefix + [float(value) for value in sequence.logprobs],
                        "advantages": [0.0] * prefix + [advantage] * len(response),
                    },
                )
            )
    return datums


async def run_tenant(index: int, args, rows_by_step) -> dict:
    tag = f"user-{index:02d}"
    # tml- prefix required by the SDK; the gateway reads the key as the tenant id
    service = tinker.ServiceClient(base_url=args.base_url, api_key=f"tml-auto-e2e-{tag}")
    # the gateway's adapter layout has no unembedding LoRA; the SDK default would be rejected
    training = await service.create_lora_training_client_async(
        base_model=args.base_model, rank=args.lora_rank, train_unembed=False
    )
    tokenizer = training.get_tokenizer()

    def encode(messages):
        return tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, enable_thinking=args.enable_thinking
        )

    started = time.time()
    path, sampler = await publish(training, service, "step0")
    log(tag, f"initial adapter version {path} ({time.time() - started:.1f}s)")
    steps = []
    for step in range(1, args.steps + 1):
        t_step = time.time()
        # rollout: waves of prompts until enough mixed-reward groups (DAPO dynamic sampling)
        candidates = [
            (prompt_tokens, label)
            for messages, label in rows_by_step[step - 1]
            if len(prompt_tokens := encode(messages)) <= args.max_prompt_tokens
        ]
        mixed, uniform = [], []
        for start in range(0, len(candidates), args.prompts_per_step):
            wave = candidates[start : start + args.prompts_per_step]
            results = await asyncio.gather(
                *(rollout_group(sampler, tokenizer, prompt_tokens, label, args) for prompt_tokens, label in wave)
            )
            for group in results:
                (mixed if len(set(group[2])) > 1 else uniform).append(group)
            if len(mixed) >= args.prompts_per_step:
                break
        note = ""
        groups = mixed[: args.prompts_per_step]
        if not groups:
            groups, note = uniform[: args.prompts_per_step], " (no mixed-reward group: zero-advantage step)"
        assert groups, f"{tag} step {step}: rollout produced no groups"
        rewards = [reward for _, _, group_rewards in groups for reward in group_rewards]
        lengths = [len(sequence.tokens) for _, sequences, _ in groups for sequence in sequences]
        truncated = sum(sequence.stop_reason == "length" for _, sequences, _ in groups for sequence in sequences)
        t_rollout = time.time()

        datums = build_datums(groups)
        fb = await (await training.forward_backward_async(datums, loss_fn="ppo", loss_fn_config=DAPO_CLIP))
        t_fb = time.time()
        await (await training.optim_step_async(types.AdamParams(learning_rate=args.lr)))
        t_optim = time.time()
        new_path, sampler = await publish(training, service, f"step{step}")
        assert new_path != path, f"{tag} step {step}: publish returned the previous version {path}"
        path = new_path
        t_publish = time.time()

        record = {
            "step": step,
            "groups": f"{len(groups)}/{len(mixed) + len(uniform)}",
            "samples": len(datums),
            "accuracy": sum(reward > 0 for reward in rewards) / len(rewards),
            "mean_len": statistics.fmean(lengths),
            "truncated": truncated,
            "loss_sum": fb.metrics.get("loss:sum"),
            "rollout_s": t_rollout - t_step,
            "fb_s": t_fb - t_rollout,
            "optim_s": t_optim - t_fb,
            "publish_s": t_publish - t_optim,
        }
        steps.append(record)
        log(
            tag,
            f"step {step}: {record['groups']} mixed groups, {record['samples']} samples, acc {record['accuracy']:.0%}, "
            f"mean {record['mean_len']:.0f} tok, {truncated} truncated | rollout {record['rollout_s']:.0f}s | "
            f"fwd/bwd {record['fb_s']:.0f}s loss:sum {record['loss_sum']} | optim {record['optim_s']:.1f}s | "
            f"published {path} {record['publish_s']:.1f}s{note}",
        )

    # the last published version must serve too
    prompt_tokens, _ = next(
        (encode(messages), label)
        for messages, label in rows_by_step[args.steps]
        if len(encode(messages)) <= args.max_prompt_tokens
    )
    response = await sampler.sample_async(
        prompt=types.ModelInput.from_ints(prompt_tokens),
        num_samples=1,
        sampling_params=types.SamplingParams(max_tokens=32, temperature=1.0),
    )
    assert list(response.sequences[0].tokens), f"{tag}: final rollout returned no tokens"
    log(tag, f"PASS: {args.steps} dependent steps, final version {path} serves ({time.time() - started:.0f}s total)")
    return {"tag": tag, "steps": steps, "total_s": time.time() - started}


async def main(args) -> int:
    rows = load_rows(args.dataset)
    per_step = args.prompts_per_step * args.sampling_rounds
    rows_by_step = [rows[step * per_step : (step + 1) * per_step] for step in range(args.steps + 1)]
    log(
        "main",
        f"{args.n_users} tenants x {args.steps} steps; {per_step} candidate prompts per step from {args.dataset}",
    )
    started = time.time()
    results = await asyncio.gather(
        *(run_tenant(index, args, rows_by_step) for index in range(args.n_users)), return_exceptions=True
    )
    failed = 0
    for index, result in enumerate(results):
        if isinstance(result, BaseException):
            failed += 1
            log(f"user-{index:02d}", "FAILED: " + "".join(traceback.format_exception(result)).strip())
    passed = args.n_users - failed
    if failed:
        log("main", f"FAIL: {failed}/{args.n_users} tenants failed ({time.time() - started:.0f}s)")
        return 1
    accuracy = statistics.fmean(step["accuracy"] for result in results for step in result["steps"])
    fb_s = statistics.fmean(step["fb_s"] for result in results for step in result["steps"])
    rollout_s = statistics.fmean(step["rollout_s"] for result in results for step in result["steps"])
    log(
        "main",
        f"PASS: {passed}/{args.n_users} tenants x {args.steps} dependent steps in {time.time() - started:.0f}s; "
        f"mean acc {accuracy:.0%}, mean rollout {rollout_s:.0f}s, mean fwd/bwd {fb_s:.0f}s per step",
    )
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:9646")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--dataset", required=True, help="dapo-math-17k jsonl: prompt (chat messages) + label")
    parser.add_argument("--n-users", type=int, required=True)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--prompts-per-step", type=int, default=2, help="accepted mixed-reward prompt groups per step")
    parser.add_argument("--samples-per-prompt", type=int, default=8)
    parser.add_argument("--sampling-rounds", type=int, default=3, help="waves of prompts to try per step")
    parser.add_argument("--max-prompt-tokens", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=6144)
    parser.add_argument("--context-len", type=int, default=8192)
    parser.add_argument("--enable-thinking", action="store_true", help="Qwen3 thinking mode (long rollouts)")
    raise SystemExit(asyncio.run(main(parser.parse_args())))
