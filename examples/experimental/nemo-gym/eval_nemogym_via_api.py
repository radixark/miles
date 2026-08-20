#!/usr/bin/env python3
"""Validate the NeMo Gym leg without a GPU trainer (golden / API-policy scan).

Drives a running mini_swe_agent_2 server through the same ``/run`` contract
the miles agent function uses, so a pass here validates everything except the
session server and training itself:

  Golden scan (no model at all — sandbox + image + SWE-bench harness):
    # start the server with the golden override:
    #   gym env start ... '++mini_swe_agent_2.responses_api_agents.mini_swe_agent_2.run_golden=true'
    python eval_nemogym_via_api.py --input swe_verified.jsonl --golden --limit 5

  API-policy scan (a real model drives episodes via the policy_base_url
  override — the exact field the trainer relies on):
    export DEEPSEEK_API_KEY=$(cat ~/.config/deepseek/api_key)
    python eval_nemogym_via_api.py --input swe_verified.jsonl --limit 2 \
        --policy-base-url https://api.deepseek.com/v1

The server sends its own configured model name on every policy request, so
start it with ``policy_model_name`` (env.yaml) set to the name the policy
endpoint expects (e.g. ``deepseek-chat``).

Input rows are miles prompt data (``{"prompt": ..., "metadata": {instance}}``,
the output of download_and_process_data.py) or raw SWE-bench instances.
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nemogym_agent_function import build_responses_create_params, post_json  # noqa: E402


def _load_instances(path: str, limit: int | None) -> list[dict]:
    instances = []
    with open(path) as f:
        for line in f:
            if limit is not None and len(instances) >= limit:
                break
            row = json.loads(line)
            instance = row.get("metadata", row)
            instance.setdefault("subset", "gym")
            instance.setdefault("split", "train")
            instances.append(instance)
    return instances


async def _run_one(args, instance: dict) -> dict:
    request = {
        **instance,
        "responses_create_params": build_responses_create_params(
            {"temperature": args.temperature, "top_p": args.top_p, "max_tokens": args.max_tokens}
        ),
    }
    if not args.golden:
        request["policy_base_url"] = args.policy_base_url
        if args.policy_api_key:
            request["policy_api_key"] = args.policy_api_key

    instance_id = instance.get("instance_id", "?")
    t0 = time.monotonic()
    try:
        response = await asyncio.wait_for(post_json(f"{args.nemo_gym_url}/run", request), timeout=args.timeout)
        reward = float(response.get("reward", 0.0))
        eval_report = response.get("metadata", {}) or {}
        error = eval_report.get("error")
    except Exception as e:  # noqa: BLE001 - a scan reports failures, it doesn't crash
        reward, eval_report, error = 0.0, {}, str(e)
    elapsed = time.monotonic() - t0
    status = "OK " if (error is None and (not args.golden or reward == 1.0)) else "FAIL"
    suffix = f"  error={error}" if error else ""
    print(f"[{status}] {instance_id}  reward={reward}  {elapsed:.0f}s{suffix}", flush=True)
    return {"instance_id": instance_id, "reward": reward, "error": error, "eval_report": eval_report}


async def _main(args) -> int:
    instances = _load_instances(args.input, args.limit)
    print(f"{'golden' if args.golden else 'api-policy'} scan: {len(instances)} instance(s) via {args.nemo_gym_url}")

    sem = asyncio.Semaphore(args.concurrency)

    async def bounded(instance):
        async with sem:
            return await _run_one(args, instance)

    results = await asyncio.gather(*(bounded(i) for i in instances))

    if args.output:
        with open(args.output, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")

    rewards = [r["reward"] for r in results]
    failures = [r for r in results if r["error"] or (args.golden and r["reward"] != 1.0)]
    print(f"\nmean reward {sum(rewards) / max(len(rewards), 1):.2f} over {len(rewards)}; {len(failures)} failure(s)")
    if args.golden and failures:
        print("golden scan FAILED: every gold patch must score 1.0")
        return 1
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="miles prompt data or raw SWE-bench jsonl")
    parser.add_argument("--nemo-gym-url", default=os.getenv("NEMO_GYM_URL", "http://localhost:12000"))
    parser.add_argument(
        "--golden", action="store_true", help="server must run with run_golden=true; expect reward 1.0"
    )
    parser.add_argument("--policy-base-url", default=None, help="OpenAI-compatible policy endpoint for the episode")
    parser.add_argument("--policy-api-key", default=os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--timeout", type=float, default=float(os.getenv("NEMO_GYM_RUN_TIMEOUT", "3600")))
    parser.add_argument("--output", default=None, help="write per-instance results jsonl here")
    args = parser.parse_args()

    if not args.golden and not args.policy_base_url:
        parser.error("either --golden or --policy-base-url is required")

    sys.exit(asyncio.run(_main(args)))


if __name__ == "__main__":
    main()
