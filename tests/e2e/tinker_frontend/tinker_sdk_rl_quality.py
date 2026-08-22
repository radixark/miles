#!/usr/bin/env python3
"""4-adapter RL training-quality acceptance, driven END TO END by the
UNMODIFIED official ``tinker==0.24.1`` SDK against the miles tinker frontend.

The SDK port of tests/e2e/multi_lora_operations/multi_lora_rl_quality.py: four adapters
run concurrent, fully independent GRPO loops on disjoint GSM8K shards
(different ranks/learning rates), 50 optimizer steps each, Qwen3 thinking
mode with a tight max_tokens budget (the learnable regime). Per step and per
adapter, everything goes over /api/v1:

  SamplingClient.sample (num_samples per prompt, temp 1.0, logprobs back)
    -> client-side math grading (reward 1/0)
    -> grouped advantages (per-prompt mean baseline, std-normalized,
       sample-mean token scaling)
    -> TrainingClient.forward_backward(loss_fn="importance_sampling",
       per-token advantages + the sampler's logprobs)
    -> TrainingClient.optim_step(AdamParams(lr, grad_clip_norm=1.0))
    -> save_weights_and_get_sampling_client (publish barrier: the loop stays
       on-policy, and the frontend fails stale samplers loudly by design)

Serving version / step clock come from the operator /adapter_runs routes
(same uvicorn, X-API-Key). One CSV per adapter + summary.json, the same
schema as the raw-op acceptance run.

Run on the head node from a venv with ``tinker==0.24.1`` installed
(PYTHONPATH must include the miles tree for the math grader):
  python tests/e2e/tinker_frontend/tinker_sdk_rl_quality.py --out-dir <dir>
"""

import argparse
import csv
import json
import os
import statistics
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

import tinker
from tinker import types

DEFAULT_SPECS = [
    # name, lora rank, learning rate, gsm8k shard (disjoint quarter of train)
    dict(name="rl_a", rank=8, lr=1e-5, shard=0),
    dict(name="rl_b", rank=16, lr=2e-5, shard=1),
    dict(name="rl_c", rank=16, lr=4e-5, shard=2),
    dict(name="rl_d", rank=32, lr=1e-5, shard=3),
]


@dataclass
class StepRecord:
    step: int
    t_start: float
    dt_s: float
    n_prompts: int
    n_samples: int
    reward_mean: float
    reward_std: float
    mean_resp_len: float
    frac_stop: float
    frac_zero_adv: float
    loss_sum: float | None
    grad_norm: float | None
    logprob_absdiff_mean: float | None
    serving_version: int | None
    note: str = ""


@dataclass
class AdapterRun:
    spec: dict
    model_id: str = ""
    adapter_name: str = ""
    registration_id: str = ""
    records: list[StepRecord] = field(default_factory=list)
    error: str | None = None
    final_step_clock: int | None = None
    final_serving_version: int | None = None


class OperatorApi:
    """The registration control plane (same uvicorn as /api/v1); used only to
    READ acceptance evidence: adapter name, step clock, serving version."""

    def __init__(self, base: str, api_key: str) -> None:
        self.base = base.rstrip("/")
        self.api_key = api_key

    def get(self, path: str) -> dict:
        req = urllib.request.Request(self.base + path, headers={"X-API-Key": self.api_key})
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read())

    def find_adapter(self, model_id: str) -> dict:
        session_id, seq = model_id.rsplit(":train:", 1)
        for status in self.get("/adapter_runs")["adapters"]:
            metadata = status.get("metadata") or {}
            if metadata.get("session_id") == session_id and str(metadata.get("model_seq_id")) == seq:
                return status
        raise RuntimeError(f"no registration found for model '{model_id}'")

    def status_of(self, name: str) -> dict:
        return self.get(f"/adapter_runs/{name}")


def group_advantages(rewards: list[float], group_size: int) -> list[float]:
    """GRPO-style per-prompt advantages: mean baseline, std-normalized."""
    advantages = []
    for start in range(0, len(rewards), group_size):
        group = rewards[start : start + group_size]
        mean = sum(group) / len(group)
        std = statistics.pstdev(group)
        advantages.extend([(r - mean) / (std + 1e-6) if std > 0 else 0.0 for r in group])
    return advantages


def rl_datum(prompt_ids: list[int], resp_tokens: list[int], resp_logprobs: list[float], per_token_adv: float):
    """Importance-sampling datum over the full sequence: zero advantage (and
    zero rollout logprob) on the prompt span, the sampler's logprobs and the
    scaled advantage on the response span. Next-token alignment holds by
    construction, which is exactly what the frontend validates."""
    full = prompt_ids + resp_tokens
    n_prompt = len(prompt_ids)
    return types.Datum(
        model_input=types.ModelInput.from_ints(full[:-1]),
        loss_fn_inputs={
            "target_tokens": full[1:],
            "logprobs": [0.0] * (n_prompt - 1) + resp_logprobs,
            "advantages": [0.0] * (n_prompt - 1) + [per_token_adv] * len(resp_tokens),
        },
    )


def adapter_loop(run: AdapterRun, base_url, api_key, operator, dataset, tokenizer, grade, args, log):
    spec = run.spec
    name = spec["name"]

    # One ServiceClient (= one SDK session) per adapter: fully independent.
    service = tinker.ServiceClient(base_url=base_url, api_key=api_key)
    base_model = service.get_server_capabilities().supported_models[0].model_name
    client = service.create_lora_training_client(base_model=base_model, rank=spec["rank"])
    run.model_id = str(client.model_id)
    status = operator.find_adapter(run.model_id)
    run.adapter_name = status["name"]
    run.registration_id = status["registration_id"]
    log(
        f"({name}) model {run.model_id} -> registration '{run.adapter_name}' "
        f"slot={status.get('slot')} rank={spec['rank']} lr={spec['lr']} rid={run.registration_id[:8]}"
    )

    # Publish the fresh (identity) adapter before the first sampling round.
    sampling = client.save_weights_and_get_sampling_client()

    params = types.SamplingParams(max_tokens=args.max_new_tokens, temperature=1.0, top_p=1.0, top_k=-1)
    cursor = 0
    step = 0
    while step < args.steps:
        t0 = time.time()
        note = ""

        prompts, labels = [], []
        while len(prompts) < args.prompts_per_step:
            row = dataset[cursor % len(dataset)]
            cursor += 1
            ids = row["input_ids"]
            if 0 < len(ids) <= args.max_prompt_tokens:
                prompts.append(ids)
                labels.append(row["label"])

        try:
            futures = [
                sampling.sample(
                    prompt=types.ModelInput.from_ints(ids),
                    num_samples=args.samples_per_prompt,
                    sampling_params=params,
                )
                for ids in prompts
            ]
            responses = [future.result() for future in futures]
        except Exception as e:  # noqa: BLE001 - a failed round is retried, not a crash
            log(f"({name}) step {step + 1}: sampling failed ({type(e).__name__}: {str(e)[:200]}); retrying")
            time.sleep(5)
            continue

        datums, rewards, resp_lens, stops, n_seqs = [], [], [], 0, 0
        sample_rows = []  # (prompt_index, resp_tokens, resp_logprobs)
        for prompt_index, response in enumerate(responses):
            label = labels[prompt_index]
            for seq in response.sequences:
                n_seqs += 1
                resp_tokens = list(seq.tokens)
                resp_logprobs = list(seq.logprobs or [])
                reward = 1.0 if resp_tokens and grade(tokenizer.decode(resp_tokens), label) else 0.0
                stops += seq.stop_reason == "stop"
                rewards.append(reward)
                resp_lens.append(len(resp_tokens))
                sample_rows.append((prompt_index, resp_tokens, resp_logprobs))

        advantages = group_advantages(rewards, args.samples_per_prompt)
        usable = [i for i, (_, toks, _) in enumerate(sample_rows) if len(toks) > 0]
        n_usable = len(usable)
        for i in usable:
            prompt_index, resp_tokens, resp_logprobs = sample_rows[i]
            per_token = advantages[i] / (len(resp_tokens) * n_usable)
            datums.append(rl_datum(prompts[prompt_index], resp_tokens, resp_logprobs, per_token))

        reward_mean = sum(rewards) / len(rewards)
        reward_std = statistics.pstdev(rewards)
        frac_zero_adv = sum(1 for i in usable if advantages[i] == 0.0) / max(n_usable, 1)

        loss_sum = grad_norm = absdiff = version = None
        optim_ok = False
        try:
            fb_future = client.forward_backward(datums, "importance_sampling")
            optim_future = client.optim_step(types.AdamParams(learning_rate=spec["lr"], grad_clip_norm=1.0))
            fb = fb_future.result()
            optim = optim_future.result()
            optim_ok = True
            loss_sum = fb.metrics.get("loss:sum")
            grad_norm = optim.metrics.get("grad_norm")

            diffs = []
            for row_index, i in enumerate(usable):
                prompt_index, resp_tokens, resp_logprobs = sample_rows[i]
                train_row = fb.loss_fn_outputs[row_index]["logprobs"].tolist()
                train_tail = train_row[len(prompts[prompt_index]) - 1 :]
                diffs.extend(abs(tr - ro) for tr, ro in zip(train_tail, resp_logprobs, strict=True))
            absdiff = sum(diffs) / len(diffs) if diffs else None

            sampling = client.save_weights_and_get_sampling_client()
            version = operator.status_of(run.adapter_name).get("version")
        except Exception as e:  # noqa: BLE001 - an op failure is a per-step finding
            note = f"{type(e).__name__}: {str(e)[:300]}"
            log(f"({name}) step {step + 1}: {note}")
            if not optim_ok:
                time.sleep(2)
                continue

        step += 1
        rec = StepRecord(
            step=step,
            t_start=t0,
            dt_s=time.time() - t0,
            n_prompts=len(prompts),
            n_samples=n_usable,
            reward_mean=reward_mean,
            reward_std=reward_std,
            mean_resp_len=sum(resp_lens) / max(len(resp_lens), 1),
            frac_stop=stops / max(n_seqs, 1),
            frac_zero_adv=frac_zero_adv,
            loss_sum=loss_sum,
            grad_norm=grad_norm,
            logprob_absdiff_mean=absdiff,
            serving_version=version,
            note=note,
        )
        run.records.append(rec)
        log(
            f"({name}) step {step}/{args.steps}: reward={reward_mean:.3f} grad_norm={grad_norm} "
            f"absdiff={absdiff if absdiff is None else round(absdiff, 4)} version={version} dt={rec.dt_s:.1f}s"
        )

    final = operator.status_of(run.adapter_name)
    run.final_step_clock = final.get("step")
    run.final_serving_version = final.get("version")


def least_squares_slope(ys: list[float]) -> float:
    n = len(ys)
    if n < 2:
        return 0.0
    xs = range(1, n + 1)
    mean_x, mean_y = (n + 1) / 2, sum(ys) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    den = sum((x - mean_x) ** 2 for x in xs)
    return num / den


def write_csv(run: AdapterRun, out_dir: str) -> str:
    path = os.path.join(out_dir, f"{run.spec['name']}.csv")
    fields = [f for f in StepRecord.__dataclass_fields__]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for rec in run.records:
            writer.writerow({k: getattr(rec, k) for k in fields})
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8068")
    parser.add_argument("--api-key", default=os.environ.get("MILES_TINKER_API_KEY", "tml-miles-gpu-acceptance"))
    parser.add_argument("--data", default="/root/datasets/gsm8k/train.parquet")
    parser.add_argument("--tokenizer", default="/root/models/Qwen3-4B")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--prompts-per-step", type=int, default=8)
    parser.add_argument("--samples-per-prompt", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-prompt-tokens", type=int, default=1024)
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Qwen3 thinking mode: with a tight max_tokens budget the base policy mostly truncates "
        "(low initial reward), which is the headroom the reward-growth check needs.",
    )
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    import pandas as pd  # noqa: PLC0415
    from transformers import AutoTokenizer  # noqa: PLC0415

    from miles.rollout.rm_hub.math_utils import grade_answer_verl  # noqa: PLC0415

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    df = pd.read_parquet(args.data)

    specs = DEFAULT_SPECS
    shards: dict[int, list[dict]] = {}
    for spec in specs:
        rows = df.iloc[spec["shard"] :: len(specs)]
        shard = []
        for _, row in rows.iterrows():
            messages = [dict(m) for m in row["messages"]]
            encoded = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, enable_thinking=args.enable_thinking
            )
            input_ids = encoded["input_ids"] if not isinstance(encoded, list) else encoded
            if input_ids and isinstance(input_ids[0], list):
                input_ids = input_ids[0]
            shard.append(dict(input_ids=[int(t) for t in input_ids], label=str(row["label"])))
        shards[spec["shard"]] = shard
        print(f"shard {spec['shard']}: {len(shard)} prompts", flush=True)

    operator = OperatorApi(args.base_url, args.api_key)
    log_lock = threading.Lock()

    def log(msg: str) -> None:
        with log_lock:
            print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    runs = [AdapterRun(spec=spec) for spec in specs]
    threads = []
    for run in runs:
        thread = threading.Thread(
            target=_thread_main,
            args=(
                run,
                args.base_url,
                args.api_key,
                operator,
                shards[run.spec["shard"]],
                tokenizer,
                grade_answer_verl,
                args,
                log,
            ),
            name=run.spec["name"],
            daemon=True,
        )
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()

    summary = {}
    for run in runs:
        rewards = [rec.reward_mean for rec in run.records]
        first10 = rewards[:10]
        last10 = rewards[-10:]
        summary[run.spec["name"]] = dict(
            spec={k: v for k, v in run.spec.items()},
            model_id=run.model_id,
            registration=run.adapter_name,
            steps_recorded=len(run.records),
            step_clock=run.final_step_clock,
            serving_version=run.final_serving_version,
            reward_first10_mean=sum(first10) / len(first10) if first10 else None,
            reward_last10_mean=sum(last10) / len(last10) if last10 else None,
            reward_slope_per_step=least_squares_slope(rewards),
            logprob_absdiff_mean=(
                sum(r.logprob_absdiff_mean for r in run.records if r.logprob_absdiff_mean is not None)
                / max(sum(1 for r in run.records if r.logprob_absdiff_mean is not None), 1)
            ),
            mean_step_dt_s=sum(r.dt_s for r in run.records) / max(len(run.records), 1),
            failures=[f"step {r.step}: {r.note}" for r in run.records if r.note],
            error=run.error,
            csv=write_csv(run, args.out_dir),
        )
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)

    grew = sum(
        1
        for s in summary.values()
        if s["reward_first10_mean"] is not None and s["reward_last10_mean"] > s["reward_first10_mean"]
    )
    print(f"\n=== RL QUALITY (SDK): reward grew (last10 > first10) on {grew}/{len(runs)} adapters ===", flush=True)


def _thread_main(run, base_url, api_key, operator, dataset, tokenizer, grade, args, log) -> None:
    try:
        adapter_loop(run, base_url, api_key, operator, dataset, tokenizer, grade, args, log)
    except Exception as e:  # noqa: BLE001 - a dead loop is a finding, not a harness crash
        run.error = f"{type(e).__name__}: {e}"
        log(f"({run.spec['name']}) LOOP ABORTED: {run.error}")


if __name__ == "__main__":
    main()
