"""Standalone Terminal-Bench-2 eval: an OpenAI-compatible *API* as the policy,
per-task Daytona sandboxes as the env. No GPU, no miles training pipeline.

This reuses the exact agent-env loop miles runs during training
(``openenv_agent_function._multi_turn``: reset -> {policy emits a shell command
-> exec -> feed output back} -> canonical tests/test.sh -> binary reward) but
swaps miles' session-server policy for a plain API client. The machine running
this only orchestrates; the policy runs in the cloud (e.g. DeepSeek) and each
episode runs in its own per-task Daytona sandbox (the task's OFFICIAL image +
env server layer — recipe in the sibling ``tb2_sandbox_recipe`` module,
materialized by ``tb2_sandbox_daytona``), created
before the episode and deleted after.

Why a separate script and not ``run-openenv-tbench2.py``: those launchers always
bring up Megatron+sglang via Ray (the policy must be miles' own engine so the
session server can record on-policy tokens for the backward pass). Pure eval
needs none of that. The top-level ``openenv_agent_function.run`` also hardcodes
``api_key="EMPTY"`` (self-hosted engines don't check it), so we call the
underlying ``_multi_turn`` with our own authenticated client instead.

Env vars:
  DEEPSEEK_API_KEY / POLICY_API_KEY   policy API key (required)
  POLICY_BASE_URL   OpenAI-compatible root (default https://api.deepseek.com)
  POLICY_MODEL      default deepseek-v4-flash
  OPENENV_TB2_TASKS_DIR  per-task sandbox mode (TB2 checkout path); with
                    DAYTONA_API_KEY or a key file at ~/.config/daytona/api_key.
                    Otherwise OPENENV_ENV_URL is used.
  OPENENV_MAX_TURNS, OPENENV_MAX_ROLLOUT_TIME_SECONDS, ...   as in the adapter.

Usage:
  # DAYTONA_API_KEY may be omitted if ~/.config/daytona/api_key is provisioned
  DEEPSEEK_API_KEY=sk-... DAYTONA_API_KEY=dtn_... \
  OPENENV_TB2_TASKS_DIR=/path/to/terminal-bench-2 \
  python eval_tbench2_via_api.py --tasks chess-best-move --concurrency 2
  # or from make_tbench2_data.py output:
  python eval_tbench2_via_api.py --data /root/tbench2_train.jsonl
"""

import argparse
import asyncio
import json
import os
import sys

from openai import AsyncOpenAI

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import openenv_agent_function as oaf  # noqa: E402


def _load_rows(args: argparse.Namespace) -> list[dict]:
    """--tasks builds rows with the adapter's own agent-contract prompt, so it
    reproduces exactly what make_tbench2_data.py emits for training."""
    if args.tasks:
        ids = [t.strip() for t in args.tasks.split(",") if t.strip()]
        return [
            {"prompt": [{"role": "system", "content": oaf.TB2_AGENT_SYSTEM_PROMPT}], "metadata": {"task_id": t}}
            for t in ids
        ]
    with open(args.data) as f:
        return [json.loads(line) for line in f if line.strip()]


async def _eval_one(
    policy: AsyncOpenAI, model: str, row: dict, request_kwargs: dict
) -> tuple[str, float | None, dict]:
    classes = oaf._load_tbench2()
    task_id = row.get("metadata", {}).get("task_id", "?")
    cap = float(os.getenv("OPENENV_MAX_ROLLOUT_TIME_SECONDS", "3600"))
    try:
        # Per-task wall-clock cap (openenv_agent_function.run has this; _multi_turn
        # alone does not). Bounds a task that loops on slow generations so one
        # straggler can't stall the whole sweep.
        reward, metrics = await asyncio.wait_for(
            oaf._multi_turn(
                classes,
                policy,
                model,
                row.get("prompt", []),
                request_kwargs,
                row.get("metadata", {}),
            ),
            timeout=cap,
        )
        return task_id, reward, metrics
    except asyncio.TimeoutError:
        return task_id, None, {"error": f"timeout>{cap:.0f}s"}
    except Exception as e:  # noqa: BLE001 - one task failing must not sink the sweep
        return task_id, None, {"error": f"{type(e).__name__}: {e}"}


async def main() -> None:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--data", help="jsonl from make_tbench2_data.py (one row per task)")
    src.add_argument("--tasks", help="comma-separated task_ids (system prompt inlined)")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--out", default="", help="optional path to write per-task jsonl results")
    args = ap.parse_args()

    api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("POLICY_API_KEY")
    if not api_key:
        sys.exit("set DEEPSEEK_API_KEY (or POLICY_API_KEY)")
    base_url = os.getenv("POLICY_BASE_URL", "https://api.deepseek.com")
    model = os.getenv("POLICY_MODEL", "deepseek-v4-flash")
    rows = _load_rows(args)
    tasks_dir = os.getenv("OPENENV_TB2_TASKS_DIR", "").strip()
    if tasks_dir:
        env_desc = f"per-task daytona sandboxes (tasks_dir={tasks_dir})"
    else:
        env_desc = os.getenv("OPENENV_ENV_URL", oaf._DEFAULT_ENV_URL)
    print(f"policy={model} @ {base_url} | env={env_desc} | " f"{len(rows)} tasks | concurrency={args.concurrency}")

    policy = AsyncOpenAI(base_url=base_url, api_key=api_key)
    request_kwargs = {"temperature": args.temperature}
    sem = asyncio.Semaphore(args.concurrency)

    async def _run(row: dict) -> tuple[str, float | None, dict]:
        async with sem:
            tid, reward, metrics = await _eval_one(policy, model, row, request_kwargs)
            tag = "ERR" if reward is None else f"{reward:.0f}"
            extra = metrics.get("error") or f"turns={metrics.get('turns')}"
            print(f"  [{tag}] {tid:40s} {extra}", flush=True)
            return tid, reward, metrics

    results = await asyncio.gather(*(_run(r) for r in rows))

    scored = [(t, r) for t, r, _ in results if r is not None]
    solved = sum(1 for _, r in scored if r >= 1.0)
    errs = sum(1 for _, r, _ in results if r is None)
    print(f"\n=== pass {solved}/{len(scored)} scored ({errs} errored) ===")
    if scored:
        print("solved:", sorted(t for t, r in scored if r >= 1.0))
    if args.out:
        with open(args.out, "w") as f:
            for t, r, m in results:
                f.write(json.dumps({"task_id": t, "reward": r, "metrics": m}) + "\n")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
