"""One client of the multi-LoRA Tinker gateway, written the way a Tinker user writes it:
DAPO on GSM8K (or dapo-math-17k), every step a dependent chain on this client's LoRA.

  1. forward_backward: DAPO's objective (PPO, asymmetric clipping, group-normalized
     token-level advantages, token-mean loss) on the rollout of the current version
  2. optim_step: update this LoRA
  3. save_weights_for_sampler under a new name: publish this step's version
  4. sample with the weights just published; that rollout is the next step's training data

auto_e2e_test.sh runs N copies of this file at once through e2e/run_clients.py, one tenant
(one api_key, one slot) each; every copy sends the same prompts in the same order.
"""

import argparse
import json
import statistics
import time

import tinker

# miles' rule-based math reward (rm_type "math"): the last \\boxed{} answer, normalized, against the label
from miles.rollout.rm_hub.math_utils import grade_answer_verl
from miles.utils.processing_utils import load_tokenizer
from tinker import types

DAPO_CLIP = {"clip_low_threshold": 0.8, "clip_high_threshold": 1.28}


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def main(args) -> None:
    service = tinker.ServiceClient(base_url=args.base_url, api_key=args.api_key)
    training = service.create_lora_training_client(
        base_model=args.base_model,  # the checkpoint the gateway serves
        rank=args.lora_rank,
        train_unembed=False,  # the gateway's adapter layout has no unembedding LoRA
    )
    tokenizer = load_tokenizer(args.base_model, trust_remote_code=True)
    per_step = args.prompts_per_step * args.sampling_rounds
    prompts = load_prompts(args.dataset, tokenizer, args, count=(args.steps + 1) * per_step)
    started = time.time()

    # a fresh adapter is B = 0, the base model; the multi-LoRA router refuses a request without
    # an adapter, so publish it before the first rollout
    saved = training.save_weights_for_sampler(name="step-0").result()
    sampler = service.create_sampling_client(model_path=saved.path)
    groups = rollout(sampler, tokenizer, prompts[:per_step], args)
    log(f"step=0 published={saved.path} acc={accuracy(groups):.0%} ({time.time() - started:.0f}s)")

    for step in range(1, args.steps + 1):
        t_start = time.time()
        # 1. gradients of DAPO's objective on the rollout of the version just sampled
        fb = training.forward_backward(dapo_datums(groups), loss_fn="ppo", loss_fn_config=DAPO_CLIP).result()
        t_fb = time.time()
        # 2. update this LoRA
        training.optim_step(types.AdamParams(learning_rate=args.lr)).result()
        t_optim = time.time()
        # 3. publish this step's weights under a new name
        saved = training.save_weights_for_sampler(name=f"step-{step}").result()
        sampler = service.create_sampling_client(model_path=saved.path)
        t_publish = time.time()
        # 4. sample with the weights just published; the next step trains on it
        groups = rollout(sampler, tokenizer, prompts[step * per_step : (step + 1) * per_step], args)
        t_rollout = time.time()

        lengths = [len(sequence.tokens) for _, sequences, _ in groups for sequence in sequences]
        prompt_lengths = [len(prompt_tokens) for prompt_tokens, _, _ in groups]
        completion = tokenizer.decode(list(groups[0][1][0].tokens), skip_special_tokens=True)
        log(
            f"step={step} loss={fb.metrics['loss:sum']:.4f} acc={accuracy(groups):.0%} "
            f"mean_len={statistics.fmean(lengths):.0f} max_len={max(lengths)} prompt_len={statistics.fmean(prompt_lengths):.0f} "
            f"fwd_bwd={t_fb - t_start:.1f}s optim={t_optim - t_fb:.1f}s publish={t_publish - t_optim:.1f}s "
            f"rollout={t_rollout - t_publish:.1f}s published={saved.path} completion={completion[-120:]!r}"
        )
    log(f"PASS steps={args.steps} final={saved.path} total={time.time() - started:.0f}s")


def load_prompts(path: str, tokenizer, args, count: int) -> list[tuple[list[int], str]]:
    """``(prompt_tokens, label)`` rows: chat messages rendered with the checkpoint's template.
    jsonl rows carry ``prompt`` (messages) and ``label`` (dapo-math-17k); parquet rows carry
    ``messages`` and ``label`` (GSM8K as prepared for miles)."""
    if path.endswith(".parquet"):
        import pyarrow.parquet as pq  # only the parquet datasets need it

        rows = pq.read_table(path, columns=["messages", "label"]).to_pylist()
    else:
        with open(path) as handle:
            rows = [json.loads(line) for line in handle]
    prompts = []
    for row in rows:
        messages = row.get("messages") or row["prompt"]
        # render, then encode: apply_chat_template(tokenize=True) returns two tokens on this transformers build
        text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False, enable_thinking=args.enable_thinking
        )
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) <= args.max_prompt_tokens:
            prompts.append((tokens, str(row["label"])))
        if len(prompts) == count:
            return prompts
    raise ValueError(f"{path} has {len(prompts)} usable prompts, {count} needed")


def rollout(sampler, tokenizer, rows: list[tuple[list[int], str]], args) -> list[tuple[list[int], list, list[float]]]:
    """``(prompt_tokens, sequences, rewards)`` per prompt. DAPO's dynamic sampling: waves of
    prompts until enough groups have mixed rewards (a uniform group carries no signal)."""
    mixed, uniform = [], []
    for start in range(0, len(rows), args.prompts_per_step):
        wave = rows[start : start + args.prompts_per_step]
        futures = [sample_group(sampler, prompt_tokens, args) for prompt_tokens, _ in wave]  # in flight together
        for (prompt_tokens, label), future in zip(wave, futures, strict=True):
            sequences = list(future.result().sequences)
            rewards = [
                (
                    1.0
                    if grade_answer_verl(tokenizer.decode(list(sequence.tokens), skip_special_tokens=True), label)
                    else -1.0
                )
                for sequence in sequences
            ]
            (mixed if len(set(rewards)) > 1 else uniform).append((prompt_tokens, sequences, rewards))
        if len(mixed) >= args.prompts_per_step:
            break
    groups = mixed[: args.prompts_per_step] or uniform[: args.prompts_per_step]  # no mixed group: zero advantages
    assert groups, "rollout produced no groups"
    return groups


def sample_group(sampler, prompt_tokens: list[int], args):
    """G samples of one prompt; returns the SDK future so a wave of prompts samples concurrently."""
    return sampler.sample(
        prompt=types.ModelInput.from_ints(prompt_tokens),
        num_samples=args.samples_per_prompt,
        sampling_params=types.SamplingParams(
            max_tokens=min(args.max_new_tokens, args.context_len - len(prompt_tokens)), temperature=1.0, top_p=1.0
        ),
    )


def dapo_datums(groups) -> list[types.Datum]:
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


def accuracy(groups) -> float:
    rewards = [reward for _, _, group_rewards in groups for reward in group_rewards]
    return sum(reward > 0 for reward in rewards) / len(rewards)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:9646")
    parser.add_argument("--api-key", default="tml-e2e-user-00", help="tml- prefix required; the gateway's tenant id")
    parser.add_argument("--base-model", required=True, help="the checkpoint the gateway serves")
    parser.add_argument("--dataset", required=True, help="GSM8K parquet (messages, label) or dapo-math-17k jsonl")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--prompts-per-step", type=int, default=2, help="accepted mixed-reward prompt groups per step")
    parser.add_argument("--samples-per-prompt", type=int, default=8)
    parser.add_argument("--sampling-rounds", type=int, default=3, help="waves of prompts to try per step")
    parser.add_argument("--max-prompt-tokens", type=int, default=2048)
    parser.add_argument(
        "--max-new-tokens", type=int, default=8192, help="clamped to the context left after the prompt"
    )
    parser.add_argument("--context-len", type=int, default=8192)
    parser.add_argument("--enable-thinking", action="store_true", help="Qwen3 thinking mode (long rollouts)")
    main(parser.parse_args())
