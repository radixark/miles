#!/usr/bin/env python3
"""Run four concurrent client-driven GRPO loops against the Multi-LoRA backend."""

import argparse
import csv
import json
import os
import statistics
import threading
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field

import ray

API = "http://127.0.0.1:8068"

DEFAULT_SPECS = [
    # Each adapter uses a disjoint quarter of the GSM8K training split.
    dict(name="rl_a", rank=8, lr=1e-5, shard=0),
    dict(name="rl_b", rank=16, lr=2e-5, shard=1),
    dict(name="rl_c", rank=16, lr=4e-5, shard=2),
    dict(name="rl_d", rank=32, lr=1e-5, shard=3),
]


def http(method: str, path: str, body: dict | None = None, base: str = API, timeout: float = 900) -> dict:
    req = urllib.request.Request(
        base + path,
        method=method,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = e.read().decode()[:500]
        except Exception:  # noqa: BLE001,S110 - the status code alone still identifies the failure
            pass
        raise RuntimeError(f"HTTP {e.code} on {method} {path}: {detail}") from e


def discover_router(explicit: str | None, candidates=(20080, 30080)) -> str:
    """The sglang router binds the node IP; the port may differ from the
    requested one when something (e.g. a stray nginx) squats it. Probe the
    worker-listing endpoints to find the live router."""
    if explicit:
        return explicit.rstrip("/")
    from miles.utils.misc import get_current_node_ip  # noqa: PLC0415

    ip = get_current_node_ip()
    for port in candidates:
        base = f"http://{ip}:{port}"
        for endpoint in ("/list_workers", "/workers"):
            try:
                body = http("GET", endpoint, base=base, timeout=5)
                if isinstance(body, dict) and ("urls" in body or "workers" in body):
                    return base
            except Exception:  # noqa: BLE001,S112 - probe failures just move to the next candidate
                continue
    raise RuntimeError(f"no router found on ports {candidates} at {ip}")


class Ops:
    """Operation plane over the controller Ray actor (one shared handle; each
    adapter thread only touches its own name's ordinal counter)."""

    def __init__(self):
        self.controller = ray.get_actor("miles_tinker_controller", namespace="miles")
        self.ordinals: dict[str, int] = {}

    def enqueue(self, name: str, kind: str, payload: dict | None = None) -> str:
        ordinal = self.ordinals.get(name, 0) + 1
        self.ordinals[name] = ordinal
        op_id = f"op-{name}-{ordinal}-{kind}-{uuid.uuid4().hex[:8]}"
        view = ray.get(self.controller.enqueue_operation.remote(name, op_id, ordinal, kind, payload))
        assert view["state"] == "QUEUED", view
        return op_id

    def wait(self, op_id: str, timeout_s: float = 1800) -> dict:
        deadline = time.monotonic() + timeout_s
        view = None
        while time.monotonic() < deadline:
            view = ray.get(self.controller.get_operation.remote(op_id))
            if view is not None and view["state"] in ("SUCCEEDED", "FAILED", "CANCELLED"):
                return view
            time.sleep(1)
        raise TimeoutError(f"operation {op_id} not terminal within {timeout_s}s: {view}")

    def run(self, name: str, kind: str, payload: dict | None = None, timeout_s: float = 1800) -> dict:
        op_id = self.enqueue(name, kind, payload)
        view = self.wait(op_id, timeout_s)
        ray.get(self.controller.ack_operation.remote(op_id))
        return view

    def step_of(self, name: str) -> int:
        return ray.get(self.controller.adapter_step.remote(name))


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
    registration_id: str = ""
    serving_name: str = ""
    records: list[StepRecord] = field(default_factory=list)
    error: str | None = None
    final_step_clock: int | None = None
    final_serving_version: int | None = None


def group_advantages(rewards: list[float], group_size: int) -> list[float]:
    """GRPO-style per-prompt advantages: mean baseline, std-normalized."""
    advantages = []
    for start in range(0, len(rewards), group_size):
        group = rewards[start : start + group_size]
        mean = sum(group) / len(group)
        std = statistics.pstdev(group)
        advantages.extend([(r - mean) / (std + 1e-6) if std > 0 else 0.0 for r in group])
    return advantages


def sample_batch(router: str, serving_name: str, prompts: list[list[int]], args) -> list[dict]:
    """One batched /generate: each prompt replicated n times, temperature 1.0
    so the returned logprobs are the sampling distribution's."""
    input_ids = [ids for ids in prompts for _ in range(args.samples_per_prompt)]
    body = dict(
        input_ids=input_ids,
        sampling_params=dict(
            temperature=1.0,
            top_p=1.0,
            top_k=-1,
            max_new_tokens=args.max_new_tokens,
        ),
        lora_path=serving_name,
        return_logprob=True,
    )
    outputs = http("POST", "/generate", body, base=router, timeout=args.sample_timeout_s)
    assert isinstance(outputs, list) and len(outputs) == len(input_ids), f"batch size mismatch: {len(outputs)}"
    return outputs


def adapter_loop(run: AdapterRun, ops: Ops, router: str, dataset: list[dict], grade, args, log) -> None:
    spec = run.spec
    name = spec["name"]

    reg = http("POST", "/adapter_runs", {"name": name, "config": {"rank": spec["rank"]}})
    deadline = time.monotonic() + 900
    while time.monotonic() < deadline:
        if http("GET", f"/adapter_runs/state?names={name}")["states"].get(name) == "READY":
            break
        time.sleep(2)
    else:
        raise TimeoutError(f"adapter '{name}' never became READY")
    info = http("GET", f"/adapter_runs/{name}")
    run.registration_id = info["registration_id"]
    from miles.ray.multi_lora.identity import serving_lora_name  # noqa: PLC0415

    run.serving_name = serving_lora_name(name, run.registration_id)
    log(
        f"({name}) registered: slot={reg.get('slot')} rank={spec['rank']} lr={spec['lr']} rid={run.registration_id[:8]}"
    )

    # Publish the fresh (identity) adapter before the first sampling round so
    # the serving name exists on the engines.
    view = ops.run(name, "save_weights_for_sampler", {})
    assert view["state"] == "SUCCEEDED", f"({name}) initial publish failed: {view.get('error')}"

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
            outputs = sample_batch(router, run.serving_name, prompts, args)
        except (urllib.error.URLError, RuntimeError, TimeoutError, AssertionError) as e:
            log(f"({name}) step {step + 1}: sampling failed ({e}); retrying next round")
            time.sleep(5)
            continue

        samples, rewards, resp_lens, stops = [], [], [], 0
        for i, out in enumerate(outputs):
            prompt_ids = prompts[i // args.samples_per_prompt]
            label = labels[i // args.samples_per_prompt]
            token_logprobs = (out.get("meta_info") or {}).get("output_token_logprobs") or []
            resp_tokens = [int(t[1]) for t in token_logprobs]
            resp_logprobs = [float(t[0]) for t in token_logprobs]
            reward = 1.0 if resp_tokens and grade(out.get("text") or "", label) else 0.0
            finish = ((out.get("meta_info") or {}).get("finish_reason") or {}).get("type")
            stops += finish == "stop"
            rewards.append(reward)
            resp_lens.append(len(resp_tokens))
            samples.append(
                dict(
                    tokens=prompt_ids + resp_tokens,
                    response_length=len(resp_tokens),
                    loss_mask=[1] * len(resp_tokens),
                    rollout_log_probs=resp_logprobs,
                )
            )

        # Grouped advantages; sample-mean scaling folds GRPO's normalization
        # into the per-token channel (the backend's loss is a plain token sum).
        advantages = group_advantages(rewards, args.samples_per_prompt)
        usable = [i for i, s in enumerate(samples) if s["response_length"] > 0]
        n_usable = len(usable)
        for i in usable:
            r_len = samples[i]["response_length"]
            per_token = advantages[i] / (r_len * n_usable)
            samples[i]["advantages"] = [per_token] * r_len

        reward_mean = sum(rewards) / len(rewards)
        reward_std = statistics.pstdev(rewards)
        frac_zero_adv = sum(1 for i in usable if advantages[i] == 0.0) / max(n_usable, 1)

        loss_sum = grad_norm = absdiff = version = None
        optim_ok = False
        try:
            fb = ops.run(
                name,
                "forward_backward",
                dict(samples=[samples[i] for i in usable], loss=dict(loss_fn="importance_sampling")),
                timeout_s=args.op_timeout_s,
            )
            if fb["state"] != "SUCCEEDED":
                raise RuntimeError(f"forward_backward FAILED: {fb.get('error')}")
            metrics = (fb.get("result") or {}).get("metrics") or {}
            loss_sum = metrics.get("loss:sum")
            train_logprobs = (fb.get("result") or {}).get("logprobs") or []
            diffs = [
                abs(tr - ro)
                for lp_row, i in zip(train_logprobs, usable, strict=True)
                for tr, ro in zip(lp_row, samples[i]["rollout_log_probs"], strict=True)
            ]
            absdiff = sum(diffs) / len(diffs) if diffs else None

            optim = ops.run(
                name,
                "optim_step",
                dict(adam_params=dict(learning_rate=spec["lr"], grad_clip_norm=1.0)),
                timeout_s=args.op_timeout_s,
            )
            if optim["state"] != "SUCCEEDED":
                raise RuntimeError(f"optim_step FAILED: {optim.get('error')}")
            optim_ok = True
            grad_norm = (optim.get("result") or {}).get("grad_norm")

            publish = ops.run(name, "save_weights_for_sampler", {}, timeout_s=args.op_timeout_s)
            if publish["state"] != "SUCCEEDED":
                note = f"publish FAILED (sampling stays on previous version): {publish.get('error')}"
            else:
                version = (publish.get("result") or {}).get("serving_version")
        except Exception as e:  # noqa: BLE001 - an op failure is a per-step finding; the loop continues
            note = f"{type(e).__name__}: {str(e)[:300]}"
            log(f"({name}) step {step + 1}: {note}")
            if not optim_ok:
                # No optimizer step landed: this round is not a step.
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
            frac_stop=stops / len(outputs),
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

    run.final_step_clock = ops.step_of(name)
    run.final_serving_version = http("GET", f"/adapter_runs/{name}").get("version")
    if args.deregister:
        http("DELETE", f"/adapter_runs/{name}")


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
    global API
    parser = argparse.ArgumentParser()
    parser.add_argument("--ray-address", default="auto")
    parser.add_argument("--api", default=API)
    parser.add_argument("--router", default=None, help="router base URL; discovered from 20080/30080 when omitted")
    parser.add_argument("--data", default="/root/gsm8k/train.parquet")
    parser.add_argument("--tokenizer", default="/root/models/Qwen3-4B")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--prompts-per-step", type=int, default=8)
    parser.add_argument("--samples-per-prompt", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-prompt-tokens", type=int, default=1024)
    parser.add_argument("--sample-timeout-s", type=float, default=900)
    parser.add_argument("--op-timeout-s", type=float, default=1800)
    parser.add_argument("--deregister", action="store_true", help="deregister adapters after the run")
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Qwen3 thinking mode. With a tight max_new_tokens budget the base policy mostly truncates "
        "(low initial reward), which is exactly the headroom the reward-growth check needs; non-thinking "
        "GSM8K starts near 0.9 and has almost no group variance left to learn from.",
    )
    args = parser.parse_args()

    API = args.api
    os.makedirs(args.out_dir, exist_ok=True)

    ray.init(address=args.ray_address, namespace="miles", ignore_reinit_error=True, log_to_driver=False)
    router = discover_router(args.router)
    print(f"router: {router}", flush=True)

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
            # transformers >= 5 returns a BatchEncoding; earlier versions a flat list.
            input_ids = encoded["input_ids"] if not isinstance(encoded, list) else encoded
            if input_ids and isinstance(input_ids[0], list):
                input_ids = input_ids[0]
            shard.append(dict(input_ids=[int(t) for t in input_ids], label=str(row["label"])))
        shards[spec["shard"]] = shard
        print(f"shard {spec['shard']}: {len(shard)} prompts", flush=True)

    ops = Ops()
    log_lock = threading.Lock()

    def log(msg: str) -> None:
        with log_lock:
            print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    runs = [AdapterRun(spec=spec) for spec in specs]
    threads = []
    for run in runs:
        thread = threading.Thread(
            target=_thread_main,
            args=(run, ops, router, shards[run.spec["shard"]], grade_answer_verl, args, log),
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
    print(f"\n=== RL QUALITY: reward grew (last10 > first10) on {grew}/{len(runs)} adapters ===", flush=True)


def _thread_main(run: AdapterRun, ops: Ops, router: str, dataset, grade, args, log) -> None:
    try:
        adapter_loop(run, ops, router, dataset, grade, args, log)
    except Exception as e:  # noqa: BLE001 - a dead loop is a finding, not a crash of the harness
        run.error = f"{type(e).__name__}: {e}"
        log(f"({run.spec['name']}) LOOP ABORTED: {run.error}")


if __name__ == "__main__":
    main()
