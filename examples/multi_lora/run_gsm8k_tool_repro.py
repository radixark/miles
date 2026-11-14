"""Run paired 100-update GSM8K tool-use clients through Miles and hosted Tinker.

Uses the calculator/answer schema and prompt from the original Trajectory blog
(5ae114f75d), with the official Qwen3.5 renderer and exact sampled logprobs.
Install examples/multi_lora/requirements.txt. Pass original GSM8K JSONL files.

    python -m examples.multi_lora.run_gsm8k_tool_repro --train train.jsonl \
        --test test.jsonl --output /tmp/gsm8k-repro --miles-url http://localhost:10613
"""

import argparse
import asyncio
import hashlib
import json
import math
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from examples.multi_lora.custom_gspo import make_gspo_loss
from examples.multi_lora.custom_maxpo import compute_mean_maxpo_advantages
from tinker_cookbook import renderers
from transformers import AutoTokenizer

import tinker
from tinker import types

SYSTEM = """You are an expert math problem solver specializing in grade school math word problems. Your task is to solve math word problems step-by-step.

You will be given:
1. A grade school level math word problem
2. The problem will involve basic arithmetic operations (+, -, *, /)

Your goal is to:
1. Read the problem carefully and identify what is being asked
2. Break down the problem into smaller steps
3. Perform calculations step-by-step using the calculator tool
4. Show your work clearly
5. Provide the final numerical answer using the answer tool

Example:
First, I need to find how many clips Natalia sold in May. She sold half as many as in April, so 48/2
[calls calculator tool for 48/2, response: 24]
That's 24 clips in May. Then I add the clips from both months:
[calls calculator tool for 48 + 24, response: 72]
72 clips total.
[calls answer tool for 72]

Be precise with your calculations and only provide the final number as your answer."""
TOOLS = [
    dict(
        name="calculator",
        description="Performs basic arithmetic operations",
        parameters={
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["add", "subtract", "multiply", "divide"],
                    "description": "The operation to perform",
                },
                "a": {"type": "number", "description": "First operand"},
                "b": {"type": "number", "description": "Second operand"},
            },
            "required": ["operation", "a", "b"],
        },
    ),
    dict(
        name="answer",
        description="Submits the final numerical answer to the problem",
        parameters={
            "type": "object",
            "properties": {"answer": {"type": "number", "description": "The final numerical answer to the problem"}},
            "required": ["answer"],
        },
    ),
]


def _write(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=lambda value: value.model_dump(mode="json")))
    temporary.replace(path)


def _tool_result(name, arguments):
    try:
        args = json.loads(arguments)
        if name == "answer":
            return str(args["answer"]), float(args["answer"])
        if name != "calculator":
            return "Unknown tool", None
        a, b = float(args["a"]), float(args["b"])
        op = args["operation"]
        if op == "add":
            value = a + b
        elif op == "subtract":
            value = a - b
        elif op == "multiply":
            value = a * b
        elif op == "divide":
            value = a / b
        else:
            return "Invalid operation", None
        return str(value), None
    except (ValueError, TypeError, KeyError, ZeroDivisionError) as error:
        return "Tool error: " + str(error), None


async def _episode(sampler, renderer, row, args, semaphore, temperature, seed):
    messages = renderer.create_conversation_prefix_with_tools(TOOLS, system_prompt=SYSTEM)
    question = row["question"] if args.raw_question else f"Problem: {row['question']}\n\nPlease solve this step-by-step."
    messages.append(dict(role="user", content=question))
    # Neil's saved ST template differs from the Cookbook's HF-style tool prompt.
    history = None
    if args.neil_chat_template:
        schema = "\n".join(json.dumps({**tool, "parameters": json.dumps(tool["parameters"], ensure_ascii=False)}, ensure_ascii=False) for tool in TOOLS)
        tool_prompt = f"""\nYou are provided with function signatures within <tools></tools> XML tags:
<tools>
{schema}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>
"""
        text = f"<|im_start|>system\n{(SYSTEM + chr(10) * 2 + tool_prompt).strip()}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n"
        history = renderer.tokenizer.encode(text, add_special_tokens=False)
    calls, submitted, failure = [], None, None
    for turn in range(args.max_turns):
        prompt = renderer.build_generation_prompt(messages)
        if history is not None:
            prompt = types.ModelInput.from_ints(history + renderer.tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False))
        prompt_ids = prompt.to_ints()
        remaining = args.total_response_tokens - sum(len(c["tokens"]) for c in calls) if args.total_response_tokens else args.max_tokens
        budget = min(args.max_tokens, remaining, args.context_tokens - len(prompt_ids))
        if budget <= 0:
            failure = "token_limit" if remaining <= 0 else "context_limit"
            break
        async with semaphore:
            response = await sampler.sample_async(
                prompt=prompt,
                num_samples=1,
                sampling_params=types.SamplingParams(
                    max_tokens=budget,
                    temperature=temperature,
                    top_p=1.0,
                    seed=seed + turn,
                    stop=renderer.get_stop_sequences(),
                ),
            )
        sample = response.sequences[0]
        if not sample.tokens or sample.logprobs is None or len(sample.logprobs) != len(sample.tokens):
            raise RuntimeError("Missing exact sampled tokens/logprobs")
        message, termination = renderer.parse_response(sample.tokens)
        if history is not None:
            history = prompt.to_ints() + sample.tokens
        calls.append(
            dict(
                prompt=prompt_ids,
                tokens=sample.tokens,
                logprobs=sample.logprobs,
                message=message,
                termination=str(termination),
                stop_reason=sample.stop_reason,
            )
        )
        messages.append(message)
        tool_calls = message.get("tool_calls", [])
        if not tool_calls:
            failure = "no_answer_tool" if sample.stop_reason != "length" else "token_limit"
            break
        answered = False
        for call in tool_calls:
            function = call.function
            result, answer = _tool_result(function.name, function.arguments)
            messages.append(dict(role="tool", content=result, name=function.name, tool_call_id=call.id))
            if history is not None:
                try:
                    payload = json.dumps({"result": float(result)})
                except ValueError:
                    payload = result
                history += renderer.tokenizer.encode(f"<|im_start|>user\n<tool_response>\n{payload}\n</tool_response><|im_end|>\n", add_special_tokens=False)
            if function.name == "answer":
                submitted = answer
                answered = True
                break
        if answered:
            break
    else:
        failure = "turn_limit"
    expected = float(row["answer"].split("####")[-1].strip().replace(",", ""))
    reward = float(submitted is not None and math.isfinite(submitted) and submitted == expected)
    return dict(
        question=row["question"],
        reward=reward,
        submitted=submitted,
        expected=expected,
        failure=failure,
        calls=calls,
        generated_tokens=sum(len(c["tokens"]) for c in calls),
    )


def _training_data(episodes, advantages, objective):
    datums, old, masks, episode_indices = [], [], [], []
    for episode in episodes:
        indices = []
        for call in episode["calls"]:
            prefix, response = call["prompt"], call["tokens"]
            tokens = prefix + response
            indices.append(len(datums))
            datums.append(types.Datum(model_input=types.ModelInput.from_ints(tokens[:-1]), loss_fn_inputs={"target_tokens": tokens[1:]}))
            old.append(torch.tensor([0.0] * (len(prefix) - 1) + call["logprobs"]))
            masks.append(torch.tensor([False] * (len(prefix) - 1) + [True] * len(response)))
        if not indices:
            raise RuntimeError("Episode has no trainable model call")
        episode_indices.append(indices)
    previous = [torch.cat([old[i][masks[i]] for i in group]) for group in episode_indices]
    sequence_loss = make_gspo_loss(
        previous,
        [torch.ones_like(p, dtype=torch.bool) for p in previous],
        torch.tensor(advantages, dtype=torch.float32),
        clip_low=0.02,
        clip_high=0.03,
    )

    def loss_fn(data, logprobs):
        if objective == "gspo":
            sequences = [torch.cat([logprobs[i][masks[i]] for i in group]) for group in episode_indices]
            loss, metrics = sequence_loss([None] * len(sequences), sequences)
            deltas = torch.cat([(old[i] - lp)[masks[i]] for i, lp in enumerate(logprobs)])
            metrics["sampler_learner_mismatch_k3"] = (deltas.exp() - deltas - 1).mean().detach().item()
            return loss, metrics
        terms = []
        for advantage, group in zip(advantages, episode_indices, strict=True):
            delta = torch.cat([(logprobs[i] - old[i])[masks[i]] for i in group])
            ratio = delta.exp()
            terms.append(-torch.minimum(ratio * advantage, ratio.clamp(0.8, 1.2) * advantage).mean())
        return torch.stack(terms).mean(), {}

    return datums, loss_fn


async def _collect_training_groups(sampler, renderer, rows, args, semaphore, seed, update):
    rng = np.random.default_rng(seed + update * 1000)
    groups, selected, requested, rejected, all_rewards = [], [], 0, 0, []
    all_failures = Counter()
    while len(groups) < args.batch_size:
        needed = args.batch_size - len(groups)
        indices = rng.choice(len(rows), needed, replace=False)
        batch = [rows[i] for i in indices for _ in range(args.group_size)]
        episodes = await asyncio.gather(*[_episode(sampler, renderer, row, args, semaphore, 1.0, seed * 1000000 + update * 10000 + (requested + index) * args.max_turns) for index, row in enumerate(batch)])
        requested += len(episodes)
        all_rewards.extend(e["reward"] for e in episodes)
        all_failures.update(e["failure"] or "answer_submitted" for e in episodes)
        for index, task in enumerate(indices):
            group = episodes[index * args.group_size : (index + 1) * args.group_size]
            if args.filter_groups:
                group = [e for e in group if e["failure"] is None]
                if len(group) < math.ceil(args.group_size / 2) or len({e["reward"] for e in group}) < 2:
                    rejected += 1
                    continue
            groups.append(group)
            selected.append(int(task))
        print(json.dumps(dict(event="collecting_groups", update=update + 1, accepted_groups=len(groups), requested_samples=requested, rejected_groups=rejected)), flush=True)
        if requested > args.batch_size * args.group_size * 100:
            raise RuntimeError("Could not collect a full batch of valid reward-contrast groups")
    return groups, selected, requested, rejected, float(np.mean(all_rewards)), dict(all_failures)


async def _run_client(provider, slot, args, rows, tests, renderer, semaphore, ready, services):
    output = Path(args.output) / f"{provider}-{slot}"
    output.mkdir(parents=True, exist_ok=True)
    key = os.environ["TINKER_API_KEY"] if provider == "tinker" else f"tml-gsm8k-{slot}"
    service = tinker.ServiceClient(api_key=key, **({"base_url": args.miles_url} if provider == "miles" else {}))
    services.append(service)
    training = await service.create_lora_training_client_async(base_model=args.model, rank=32, train_unembed=False)
    if args.initial_checkpoint is not None:
        if provider != "miles":
            raise ValueError("Initial gateway checkpoints require --providers miles")
        imported = await training.load_state_async(args.initial_checkpoint.format(client=slot))
        await imported
    model_id = training.model_id
    _write(
        output / "status.json",
        dict(
            provider=provider,
            slot=slot,
            model_id=model_id,
            completed_updates=0,
            status="waiting for matched start",
            objective=args.objective,
        ),
    )
    print(json.dumps(dict(event="created", provider=provider, slot=slot, model_id=model_id)), flush=True)
    ready[provider, slot] = True
    while len(ready) != len(args.providers) * args.clients:
        await asyncio.sleep(0.5)
    started = time.time()
    seed = args.seed + slot
    metrics_path = output / "metrics.jsonl"
    for update in range(args.updates + 1):
        start = time.time()
        save = await training.save_weights_for_sampler_async(name=f"update-{update:03d}")
        checkpoint = (await save).path
        sampler = await service.create_sampling_client_async(model_path=checkpoint)
        if update % args.eval_every == 0 or update == args.updates:
            evaluations = await asyncio.gather(*[_episode(sampler, renderer, row, args, semaphore, 0.0, seed * 1000000 + index) for index, row in enumerate(tests[: args.eval_tasks])])
            _write(output / f"eval-{update:03d}.json", dict(update=update, checkpoint=checkpoint, samples=evaluations))
            if update == 0 and args.require_baseline_signal and not any(e["reward"] for e in evaluations):
                raise RuntimeError("No baseline reward signal; inspect saved tool-call traces before training")
        if update == args.updates:
            final = await training.save_state_async(name=f"final-update-{update}")
            state_path = (await final).path
            _write(
                output / "status.json",
                dict(
                    provider=provider,
                    slot=slot,
                    model_id=model_id,
                    completed_updates=update,
                    status="complete",
                    checkpoint=state_path,
                    elapsed_seconds=time.time() - started,
                ),
            )
            return
        episode_groups, selected, requested, rejected, unfiltered_reward, rollout_failures = await _collect_training_groups(sampler, renderer, rows, args, semaphore, seed, update)
        episodes = [e for group in episode_groups for e in group]
        rewards = [np.array([e["reward"] for e in group]) for group in episode_groups]
        maxpo_metrics = {}
        if args.objective == "maxpo":
            groups = [compute_mean_maxpo_advantages(group, args.maxpo_k, args.maxpo_mean_weight) for group in rewards]
            advantages = np.concatenate([group[0] for group in groups])
            maxpo_metrics = {key: float(np.mean([group[1][key] for group in groups])) for key in groups[0][1]}
        else:
            advantages = np.concatenate([(group - group.mean()) / (group.std() + 1e-8) for group in rewards])
        rollout_done = time.time()
        datums, loss_fn = _training_data(episodes, advantages, args.objective)
        fb = await training.forward_backward_custom_async(datums, loss_fn)
        result = await fb
        optim = await training.optim_step_async(types.AdamParams(learning_rate=args.lr * min(1.0, (update + 1) / max(1, args.warmup_steps)), beta1=0.9, beta2=0.95, eps=1e-8, weight_decay=0.01, grad_clip_norm=args.grad_clip_norm))
        await optim
        end = time.time()
        lengths = [e["generated_tokens"] for e in episodes]
        row = dict(
            update=update + 1,
            train_reward=float(np.mean([e["reward"] for e in episodes])),
            elapsed_seconds=end - started,
            step_seconds=end - start,
            rollout_and_eval_seconds=rollout_done - start,
            training_seconds=end - rollout_done,
            admitted_samples=len(episodes),
            requested_samples=requested,
            rejected_groups=rejected,
            unfiltered_train_reward=unfiltered_reward,
            rollout_failures=rollout_failures,
            learning_rate=args.lr * min(1.0, (update + 1) / max(1, args.warmup_steps)),
            zero_advantage_groups=float(np.mean([group.std() == 0 for group in rewards])),
            length_mean=float(np.mean(lengths)),
            length_p95=float(np.percentile(lengths, 95)),
            sampled_token_nll=-sum(sum(c["logprobs"]) for e in episodes for c in e["calls"]) / sum(lengths),
            failures=dict(Counter(e["failure"] or "answer_submitted" for e in episodes)),
            task_fingerprint=hashlib.sha256(json.dumps(selected).encode()).hexdigest(),
            metrics={**result.metrics, **maxpo_metrics},
            checkpoint=checkpoint,
        )
        with metrics_path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        _write(output / f"train-traces-{update + 1:03d}.json", episodes[:3])
        _write(
            output / "status.json",
            dict(
                provider=provider,
                slot=slot,
                model_id=model_id,
                completed_updates=update + 1,
                status="running",
                latest=row,
            ),
        )
        print(json.dumps(dict(provider=provider, slot=slot, **row)), flush=True)


async def _main(args):
    rows = [json.loads(line) for line in Path(args.train).read_text().splitlines()]
    tests = [json.loads(line) for line in Path(args.test).read_text().splitlines()]
    renderer = renderers.get_renderer(args.renderer, AutoTokenizer.from_pretrained(args.tokenizer))
    semaphores = {p: asyncio.Semaphore(args.parallel_samples) for p in args.providers}
    ready = {}
    services, status = [], "errored"
    try:
        async with asyncio.TaskGroup() as tasks:
            for provider in args.providers:
                for slot in range(args.clients):
                    tasks.create_task(_run_client(provider, slot, args, rows, tests, renderer, semaphores[provider], ready, services))
        status = "success"
    finally:
        await asyncio.gather(*(service.close(status) for service in services))
        _write(Path(args.output) / "finished.json", dict(status=status, time=time.time()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--miles-url", default="http://127.0.0.1:10613")
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--tokenizer", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--renderer", default="qwen3_5_disable_thinking")
    parser.add_argument("--raw-question", action="store_true")
    parser.add_argument("--neil-chat-template", action="store_true")
    parser.add_argument("--filter-groups", action="store_true")
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--total-response-tokens", type=int, default=0)
    parser.add_argument("--require-baseline-signal", action="store_true")
    parser.add_argument("--initial-checkpoint", help="Initial Miles weights checkpoint; {client} selects the tenant.")
    parser.add_argument("--providers", nargs="+", choices=["miles", "tinker"], default=["miles", "tinker"])
    parser.add_argument("--clients", type=int, default=8)
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--objective", choices=["gspo", "grpo", "maxpo"], default="gspo")
    parser.add_argument("--maxpo-k", type=int, default=64)
    parser.add_argument("--maxpo-mean-weight", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-turns", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--context-tokens", type=int, default=16377)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--eval-tasks", type=int, default=128)
    parser.add_argument("--parallel-samples", type=int, default=128)
    args = parser.parse_args()
    if args.objective == "maxpo" and not 1 <= args.maxpo_k < args.group_size:
        raise ValueError("MaxPO requires 1 <= --maxpo-k < --group-size")
    if args.objective == "maxpo" and args.filter_groups:
        raise ValueError("MaxPO requires fixed groups; do not enable variable-size group filtering")
    Path(args.output).mkdir(parents=True, exist_ok=True)
    _write(Path(args.output) / "config.json", vars(args))
    asyncio.run(_main(args))


if __name__ == "__main__":
    main()
