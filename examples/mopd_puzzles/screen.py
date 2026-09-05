"""Measure accuracy, response length, and throughput against an SGLang server."""

import argparse
import asyncio
import json
import time
from collections import defaultdict
from pathlib import Path

import httpx
import numpy as np
from examples.mopd_puzzles.tasks import score
from transformers import AutoTokenizer


async def _run(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    rows = [json.loads(line) for path in args.data for line in path.read_text().splitlines()]
    semaphore = asyncio.Semaphore(args.concurrency)
    results = []
    started = time.monotonic()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    async with httpx.AsyncClient(timeout=600, limits=httpx.Limits(max_connections=args.concurrency)) as client:

        async def request(row):
            messages = row["prompt"]
            if args.answer_only:
                messages = [
                    dict(
                        role="system",
                        content="Solve the puzzle. Output only one <answer>...</answer> block. Do not include reasoning or explanation.",
                    ),
                    *messages[1:],
                ]
            tokens = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=False
            )
            async with semaphore:
                start = time.monotonic()
                response = await client.post(
                    args.url + "/generate",
                    json=dict(
                        input_ids=tokens,
                        sampling_params=dict(
                            temperature=args.temperature,
                            top_p=1.0,
                            top_k=-1,
                            max_new_tokens=args.max_tokens,
                            skip_special_tokens=True,
                            no_stop_trim=True,
                            **({"stop": ["</answer>"]} if args.stop_at_answer else {}),
                        ),
                        return_logprob=False,
                    ),
                )
                response.raise_for_status()
                value = response.json()
                meta = value["meta_info"]
                result = dict(
                    config=row["metadata"]["config"],
                    puzzle_id=row["metadata"]["puzzle_id"],
                    correct=score(value["text"], row["label"]),
                    text=value["text"],
                    label=row["label"],
                    completion_tokens=meta["completion_tokens"],
                    finish_reason=meta["finish_reason"],
                    latency_s=time.monotonic() - start,
                )
                results.append(result)
                with args.output.open("a") as output:
                    output.write(json.dumps(result) + "\n")
                if len(results) % 64 == 0:
                    print("Completed", len(results), "/", len(rows), flush=True)

        await asyncio.gather(*(request(row) for row in rows))
    elapsed = time.monotonic() - started
    groups = defaultdict(list)
    for row in results:
        groups[row["config"]].append(row)
    report = dict(
        elapsed_s=elapsed,
        response_tokens_per_s=sum(r["completion_tokens"] for r in results) / elapsed,
        model=args.model,
        url=args.url,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        answer_only=args.answer_only,
        stop_at_answer=args.stop_at_answer,
        configs={},
    )
    for config, group in sorted(groups.items()):
        lengths = [r["completion_tokens"] for r in group]
        report["configs"][config] = dict(
            n=len(group),
            accuracy=np.mean([r["correct"] for r in group]),
            median_tokens=float(np.median(lengths)),
            p95_tokens=float(np.percentile(lengths, 95)),
            truncated=sum(r["finish_reason"]["type"] == "length" for r in group) / len(group),
        )
    args.output.with_suffix(".summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--answer-only", action="store_true")
    parser.add_argument("--stop-at-answer", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
