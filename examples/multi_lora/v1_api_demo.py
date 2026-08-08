"""End-to-end demo of the /v1 fine-tuning API against a running service.

Registers a dataset and evaluator, submits N LoRA post-training jobs, watches
their optimizer steps and token usage, cancels some mid-run, and waits for
the rest to complete. Everything goes through the public /v1 surface — this
is also the reference client for the API.

Start the service first:
    python examples/multi_lora/run_multi_lora.py serve      # API on :8068

Then:
    python examples/multi_lora/v1_api_demo.py --api-url http://127.0.0.1:8068 \\
        --data /root/datasets/gsm8k/train.parquet --jobs 4 --max-steps 5 --cancel 2
"""

import argparse
import json
import sys
import time

import httpx

POLL_INTERVAL_S = 15.0


class V1Client:
    def __init__(self, api_url: str):
        self.api_url = api_url.rstrip("/")
        self.http = httpx.Client(timeout=30.0)

    def call(self, method: str, path: str, **kwargs) -> tuple[int, dict]:
        response = self.http.request(method, f"{self.api_url}{path}", **kwargs)
        return response.status_code, response.json()

    def must(self, method: str, path: str, **kwargs) -> dict:
        status, body = self.call(method, path, **kwargs)
        if status != 200:
            raise RuntimeError(f"{method} {path} -> {status}: {json.dumps(body)}")
        return body


def job_row(job: dict) -> str:
    progress = job.get("jobProgress") or {}
    totals = (job.get("usage") or {}).get("totals") or {}
    return (
        f"  {job['name'].split('/')[-1]:8s} state={job['state']:10s} stop={str(job['stopReason']):18s} "
        f"step={progress.get('completedSteps')}/{progress.get('maxSteps')} "
        f"rolloutTok={totals.get('rolloutTokens', 0):>10,} trainTok={totals.get('trainingTokens', 0):>10,}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default="http://127.0.0.1:8068")
    parser.add_argument("--data", default="/root/datasets/gsm8k/train.parquet")
    parser.add_argument("--input-key", default="messages")
    parser.add_argument("--label-key", default="label")
    parser.add_argument("--rm-type", default="math")
    parser.add_argument("--jobs", type=int, default=4, help="LoRA jobs to submit")
    parser.add_argument("--cancel", type=int, default=2, help="jobs to cancel once past --cancel-at-step")
    parser.add_argument("--cancel-at-step", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--batch-size-prompts", type=int, default=4)
    parser.add_argument("--rollouts-per-prompt", type=int, default=4)
    parser.add_argument("--timeout-s", type=float, default=3600.0)
    args = parser.parse_args()

    client = V1Client(args.api_url)
    info = client.must("GET", "/v1/info")
    print(f"[demo] base={info['baseModel']} slots={info['slots']}")
    if args.jobs > info["slots"]["free"]:
        print(f"[demo] only {info['slots']['free']} free slots for {args.jobs} jobs; extra creates would 429")

    # Resources are idempotent-ish for reruns: tolerate ALREADY_EXISTS.
    for path, payload in (
        ("/v1/datasets", {
            "datasetId": "gsm8k",
            "source": {"clusterPath": args.data},
            "schema": {"inputKey": args.input_key, "labelKey": args.label_key},
        }),
        ("/v1/evaluators", {"evaluatorId": "math", "kind": "BUILTIN", "builtin": {"rmType": args.rm_type}}),
    ):
        status, body = client.call("POST", path, json=payload)
        if status not in (200, 409):
            raise RuntimeError(f"POST {path} -> {status}: {json.dumps(body)}")
        print(f"[demo] {path} -> {'created' if status == 200 else 'already there'}")

    job_names = [f"lora-{n}" for n in range(1, args.jobs + 1)]
    for name in job_names:
        job = client.must("POST", "/v1/postTrainingJobs", json={
            "jobId": name,
            "dataset": "datasets/gsm8k",
            "evaluator": "evaluators/math",
            "trainingConfig": {
                "loraRank": args.lora_rank,
                "batchSizePrompts": args.batch_size_prompts,
                "rolloutsPerPrompt": args.rollouts_per_prompt,
                "maxSteps": args.max_steps,
            },
        })
        print(f"[demo] created {job['name']} uid={job['uid'][:8]}… state={job['state']}")

    to_cancel = job_names[-args.cancel:] if args.cancel else []
    cancelled: set[str] = set()
    start = time.time()
    while True:
        time.sleep(POLL_INTERVAL_S)
        listing = client.must("GET", "/v1/postTrainingJobs")
        jobs = {j["name"].split("/")[-1]: j for j in listing["postTrainingJobs"]}
        print(f"\n[t+{time.time() - start:5.0f}s]")
        for name in job_names:
            if name in jobs:
                print(job_row(jobs[name]))

        for name in to_cancel:
            if name in cancelled or name not in jobs or jobs[name]["state"] != "RUNNING":
                continue
            step = (jobs[name].get("jobProgress") or {}).get("completedSteps") or 0
            if step >= args.cancel_at_step:
                body = client.must("POST", f"/v1/postTrainingJobs/{name}:cancel")
                print(f"  >>> CANCEL {name} at step {step} -> {body['state']} ({body['stopReason']})")
                cancelled.add(name)

        states = {name: jobs[name]["state"] for name in job_names if name in jobs}
        if len(states) == len(job_names) and all(s in ("COMPLETED", "CANCELLED") for s in states.values()):
            break
        if time.time() - start > args.timeout_s:
            print("[demo] TIMEOUT waiting for terminal states")
            return 1

    print("\n[demo] all jobs terminal; final ledger:")
    for entry in client.must("GET", "/v1/usage")["entries"]:
        totals = entry["usage"]["totals"]
        print(
            f"  {entry['name']:8s} uid={entry['uid'][:8]}… finalized={str(entry['finalized']):5s} "
            f"rollout={totals['rolloutTokens']:>10,} training={totals['trainingTokens']:>10,} "
            f"computed={totals['computedTokens']:>10,}"
        )
    model = client.must("GET", f"/v1/models/{job_names[0]}")
    print(f"\n[demo] model {model['name']}: state={model['state']} checkpoints={[c['checkpointId'] for c in model['checkpoints']]}")
    print("[demo] DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
