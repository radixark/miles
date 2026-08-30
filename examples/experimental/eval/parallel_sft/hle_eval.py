"""Run a small, directly scored HLE evaluation against an OpenAI endpoint.

The production HLE benchmark uses an LLM judge for free-form answers. This
driver also supports that dataset shape, but its built-in score is deliberately
limited to multiple-choice rows so infrastructure smoke tests have a
deterministic, judge-free correctness signal.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from pathlib import Path
from typing import Any

import aiohttp
from tap import Tap


class Args(Tap):
    input: str
    base_url: str
    model: str
    output_jsonl: str
    summary_json: str
    api_key: str = "dummy"
    concurrency: int = 8
    max_tasks: int | None = None
    max_tokens: int = 8192
    temperature: float = 0.0
    request_timeout_sec: int = 3600
    multiple_choice_only: bool = False


def extract_choice(text: str) -> str | None:
    """Extract a final A-Z multiple-choice answer without scanning reasoning."""
    patterns = (
        r"\\boxed\s*\{\s*([A-Z])\s*\}",
        r"(?im)^\s*final\s+answer\s*:\s*(?:\*\*)?([A-Z])\b",
        r"(?im)^\s*answer\s*:\s*(?:\*\*)?([A-Z])\b",
        r"(?m)^\s*([A-Z])\s*[.)]?\s*$",
    )
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].upper()
    return None


def load_rows(path: Path, *, multiple_choice_only: bool, max_tasks: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if multiple_choice_only and row.get("answer_type") != "multipleChoice":
                continue
            rows.append(row)
            if max_tasks is not None and len(rows) >= max_tasks:
                break
    return rows


async def evaluate_one(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    args: Args,
    row: dict[str, Any],
) -> dict[str, Any]:
    prompt = (
        f"{row['question']}\n\n"
        "Solve the problem carefully. End your response on a new line with "
        "`Final answer: X`, where X is the answer-choice letter."
    )
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
    }
    headers = {"Authorization": f"Bearer {args.api_key}"}
    started = time.monotonic()
    result: dict[str, Any] = {
        "id": row["id"],
        "answer": row["answer"],
        "answer_type": row["answer_type"],
    }
    try:
        async with semaphore:
            async with session.post(
                f"{args.base_url.rstrip('/')}/chat/completions",
                json=payload,
                headers=headers,
            ) as response:
                body = await response.text()
                result["status_code"] = response.status
                if response.status != 200:
                    result["error"] = body[:2000]
                    return result
                completion = json.loads(body)
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["status_code"] = 0
        return result
    finally:
        result["elapsed_sec"] = round(time.monotonic() - started, 3)

    choice = completion["choices"][0]["message"]
    content = choice.get("content") or ""
    predicted = extract_choice(content)
    result.update(
        {
            "content": content,
            "reasoning_content": choice.get("reasoning_content"),
            "predicted_answer": predicted,
            "correct": float(predicted == str(row["answer"]).strip().upper()),
            "completion_tokens": completion.get("usage", {}).get("completion_tokens", 0),
        }
    )
    return result


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [result for result in results if result.get("status_code") == 200]
    graded = [result for result in completed if result.get("predicted_answer") is not None]
    correct = sum(result.get("correct", 0.0) for result in graded)
    completion_tokens = sum(result.get("completion_tokens", 0) for result in completed)
    return {
        "metrics": {
            "tasks_total": len(results),
            "completed": len(completed),
            "errors": len(results) - len(completed),
            "request_success_rate": len(completed) / len(results) if results else 0.0,
            "graded": len(graded),
            "correct": correct,
            "accuracy": correct / len(graded) if graded else 0.0,
            "completion_tokens": completion_tokens,
        },
        "rewards": [result.get("correct") for result in results],
        "per_task": {result["id"]: result for result in results},
    }


async def main_async(args: Args) -> None:
    rows = load_rows(
        Path(args.input),
        multiple_choice_only=args.multiple_choice_only,
        max_tasks=args.max_tasks,
    )
    if not rows:
        raise ValueError("No HLE rows matched the requested filters")

    timeout = aiohttp.ClientTimeout(total=args.request_timeout_sec)
    connector = aiohttp.TCPConnector(limit=args.concurrency)
    semaphore = asyncio.Semaphore(args.concurrency)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        results = await asyncio.gather(
            *(evaluate_one(session, semaphore, args, row) for row in rows)
        )

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(f"{json.dumps(result)}\n" for result in results))
    summary = summarize(results)
    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["metrics"], sort_keys=True), flush=True)


def main() -> None:
    args = Args().parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
