"""Measure warm scoring latency, sparse transport, and routed endpoint scaling."""

import argparse
import asyncio
import json
import time
from pathlib import Path

import httpx
import numpy as np
from transformers import AutoTokenizer


async def _make_payloads(client, args):
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    row = json.loads(args.data.open().readline())
    prompt = tokenizer.apply_chat_template(
        row["prompt"], tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=False
    )
    response = await client.post(
        args.urls[0] + "/generate",
        json=dict(
            input_ids=prompt,
            sampling_params=dict(temperature=0, max_new_tokens=128),
            return_logprob=True,
            top_logprobs_num=16,
        ),
    )
    response.raise_for_status()
    meta = response.json()["meta_info"]
    response_ids = [entry[1] for entry in meta["output_token_logprobs"]]
    supports = [[entry[1] for entry in entries] for entries in meta["output_top_logprobs"]]
    if not response_ids or len(response_ids) != len(supports):
        raise ValueError("The workload requires a nonempty response with aligned top-k scores")
    workloads = []
    for length in [len(response_ids), 512]:
        tokens = prompt + [response_ids[i % len(response_ids)] for i in range(length)]
        rows = [supports[i % len(supports)] for i in range(length)]
        union = sorted({token for candidates in rows for token in candidates})
        for start in [0, len(prompt) - 1]:
            base = dict(
                input_ids=tokens,
                sampling_params=dict(temperature=0, max_new_tokens=0),
                return_logprob=True,
                logprob_start_len=start,
            )
            workloads.append(
                (
                    dict(
                        response_length=length,
                        prompt_length=len(prompt),
                        logprob_start_len=start,
                        unique_ids=len(union),
                    ),
                    rows,
                    {
                        "dense": {**base, "token_ids_logprob": union},
                        "sparse": {**base, "token_ids_logprob_positions": [[] for _ in prompt] + rows},
                    },
                )
            )
    return workloads


async def _request(client, url, payload, rows):
    start = time.monotonic()
    response = await client.post(url + "/generate", json=payload)
    response.raise_for_status()
    scores = response.json()["meta_info"]["input_token_ids_logprobs"][-len(rows) :]
    elapsed = time.monotonic() - start
    if len(scores) != len(rows):
        raise AssertionError("Teacher response positions drifted")
    selected = []
    sparse = "token_ids_logprob_positions" in payload
    for candidates, entries in zip(rows, scores, strict=True):
        if sparse and [entry[1] for entry in entries] != candidates:
            raise AssertionError("Sparse candidate order drifted")
        values = {entry[1]: entry[0] for entry in entries}
        selected.append([values[token] for token in candidates])
    return elapsed, len(response.content), np.asarray(selected)


async def _measure(client, urls, payload, rows, concurrency, requests, oracle):
    semaphore = asyncio.Semaphore(concurrency)

    async def one(index):
        async with semaphore:
            url = urls[index % len(urls)]
            latency, size, scores = await _request(client, url, payload, rows)
            return latency, size, float(np.abs(scores - oracle[url]).max())

    start = time.monotonic()
    results = await asyncio.gather(*(one(i) for i in range(requests)))
    elapsed = time.monotonic() - start
    return dict(
        wall_seconds=elapsed,
        requests_per_second=requests / elapsed,
        median_latency_seconds=float(np.median([r[0] for r in results])),
        p95_latency_seconds=float(np.percentile([r[0] for r in results], 95)),
        mean_response_bytes=float(np.mean([r[1] for r in results])),
        max_score_error=max(r[2] for r in results),
    )


async def _run(args):
    report = []
    async with httpx.AsyncClient(timeout=120) as client:
        workloads = await _make_payloads(client, args)
        for shape, rows, payloads in workloads:
            oracle = {}
            for url in args.urls:
                _, _, oracle[url] = await _request(client, url, payloads["dense"], rows)
                _, _, sparse = await _request(client, url, payloads["sparse"], rows)
                error = float(np.abs(sparse - oracle[url]).max())
                if error > 2e-3:
                    raise AssertionError(f"Sparse scoring oracle failed: shape={shape}, url={url}, max_error={error}")
            if args.require_replica_parity:
                error = max(float(np.abs(value - oracle[args.urls[0]]).max()) for value in oracle.values())
                if error > 2e-3:
                    raise AssertionError(f"Identical model replicas disagree: shape={shape}, max_error={error}")
            for endpoint_count in sorted({1, len(args.urls)}):
                for concurrency in [1, 8]:
                    for repeat in range(args.repeats):
                        for mode in ["dense", "sparse"] if repeat % 2 == 0 else ["sparse", "dense"]:
                            result = await _measure(
                                client,
                                args.urls[:endpoint_count],
                                payloads[mode],
                                rows,
                                concurrency,
                                args.requests,
                                oracle,
                            )
                            report.append(
                                {
                                    **shape,
                                    "mode": mode,
                                    "endpoints": endpoint_count,
                                    "concurrency": concurrency,
                                    "repeat": repeat,
                                    "requests": args.requests,
                                    **result,
                                }
                            )
                            args.output.write_text(json.dumps(report, indent=2))
                            print(json.dumps(report[-1]), flush=True)
    # Batched kernels can differ from an isolated forward; report the measured
    # error and reject a large discrepancy instead of treating it as performance.
    if max(row["max_score_error"] for row in report) > 0.02:
        raise AssertionError("Concurrent scoring exceeded the 0.02 log-probability error gate")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--urls", nargs="+", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--requests", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--require-replica-parity", action="store_true")
    args = parser.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
