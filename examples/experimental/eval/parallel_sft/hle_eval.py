"""Evaluate HLE generations with an optional external OpenAI-compatible judge."""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import aiohttp
from tap import Tap


JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "extracted_final_answer": {"type": "string"},
        "reasoning": {"type": "string"},
        "correct": {"type": "string", "enum": ["yes", "no"]},
        "confidence": {"type": "integer", "minimum": 0, "maximum": 100},
    },
    "required": [
        "extracted_final_answer",
        "reasoning",
        "correct",
        "confidence",
    ],
    "additionalProperties": False,
}

JUDGE_PROMPT = """Judge whether the response to the question is correct based only on the
reference answer. Extract the response's final answer; do not solve the problem
yourself or propose an alternative answer. Small numerical rounding differences
are acceptable, but ambiguity or a missing final answer is incorrect.

Question:
{question}

Response:
{response}

Reference answer:
{answer}

Return a JSON object with exactly these fields:
- extracted_final_answer: string; use "None" if no final answer is present
- reasoning: a brief comparison of the extracted and reference answers
- correct: "yes" or "no"
- confidence: the 0-100 confidence stated in the response, or 100 if absent
"""


class Args(Tap):
    input: str
    base_url: str
    model: str
    output_jsonl: str
    summary_json: str
    api_key: str = "dummy"
    concurrency: int = 8
    max_tasks: int | None = None
    n_trials: int = 1
    max_tokens: int = 131072
    temperature: float = 0.0
    request_timeout_sec: int = 3600
    multiple_choice_only: bool = False

    # ``judge_base_url`` is an OpenAI-compatible base URL, including ``/v1``.
    # This works with an independently hosted SGLang server or router.
    judge_base_url: str | None = None
    judge_model: str | None = None
    judge_api_key: str = "dummy"
    judge_api_key_env: str = "HLE_JUDGE_API_KEY"
    judge_concurrency: int = 16
    judge_max_tokens: int = 16384
    judge_temperature: float = 0.0
    judge_request_timeout_sec: int = 3600
    judge_max_retries: int = 3


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


def generation_prompt(row: dict[str, Any]) -> str:
    """Build the HLE response prompt for either answer type."""
    if row.get("answer_type") == "multipleChoice":
        final_answer_instruction = "the answer-choice letter"
    else:
        final_answer_instruction = "the exact final answer"
    return f"{row['question']}\n\nSolve the problem carefully. End your response with two lines in this format:\nFinal answer: <{final_answer_instruction}>\nConfidence: <an integer from 0 to 100>"


def judge_prompt(row: dict[str, Any], response: str) -> str:
    """Build the reference-based HLE grading prompt."""
    return JUDGE_PROMPT.format(
        question=row["question"],
        response=response,
        answer=row["answer"],
    )


def judge_payload(args: Args, row: dict[str, Any], response: str) -> dict[str, Any]:
    """Build an SGLang/OpenAI-compatible schema-constrained grading request."""
    if args.judge_model is None:
        raise ValueError("--judge_model is required when --judge_base_url is set")
    return {
        "model": args.judge_model,
        "messages": [{"role": "user", "content": judge_prompt(row, response)}],
        "max_tokens": args.judge_max_tokens,
        "temperature": args.judge_temperature,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "hle_judgment", "schema": JUDGE_SCHEMA},
        },
    }


def parse_judgment(text: str) -> dict[str, Any]:
    """Parse and validate a grader JSON response."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, count=1)
        stripped = re.sub(r"\s*```$", "", stripped, count=1)
    if not stripped.startswith("{"):
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end < start:
            raise ValueError("Judge response does not contain a JSON object")
        stripped = stripped[start : end + 1]

    judgment = json.loads(stripped)
    missing = set(JUDGE_SCHEMA["required"]) - judgment.keys()
    if missing:
        raise ValueError(f"Judge response is missing fields: {sorted(missing)}")
    if judgment["correct"] not in {"yes", "no"}:
        raise ValueError("Judge field 'correct' must be 'yes' or 'no'")
    confidence = judgment["confidence"]
    if isinstance(confidence, bool) or not isinstance(confidence, int):
        raise ValueError("Judge field 'confidence' must be an integer")
    if not 0 <= confidence <= 100:
        raise ValueError("Judge field 'confidence' must be between 0 and 100")
    return judgment


def resolve_judge_api_key(args: Args) -> str:
    """Resolve grader authentication without requiring a secret in command argv."""
    return os.environ.get(args.judge_api_key_env, args.judge_api_key)


async def evaluate_one(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    args: Args,
    row: dict[str, Any],
    trial_index: int,
) -> dict[str, Any]:
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": generation_prompt(row)}],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
    }
    headers = {"Authorization": f"Bearer {args.api_key}"}
    started = time.monotonic()
    result: dict[str, Any] = {
        "id": row["id"],
        "trial_index": trial_index,
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
    result.update(
        {
            "content": content,
            "reasoning_content": choice.get("reasoning_content"),
            "completion_tokens": completion.get("usage", {}).get("completion_tokens", 0),
        }
    )
    if row.get("answer_type") == "multipleChoice":
        predicted = extract_choice(content)
        result["direct_predicted_answer"] = predicted
        if predicted is not None:
            result["direct_correct"] = float(predicted == str(row["answer"]).strip().upper())
            if args.judge_base_url is None:
                result["predicted_answer"] = predicted
                result["correct"] = result["direct_correct"]
    return result


async def judge_one(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    args: Args,
    row: dict[str, Any],
    result: dict[str, Any],
) -> None:
    """Grade one successful generation through the external judge endpoint."""
    if result.get("status_code") != 200 or args.judge_base_url is None:
        return

    result["judge_requested"] = True
    payload = judge_payload(args, row, result.get("content", ""))
    headers = {"Authorization": f"Bearer {resolve_judge_api_key(args)}"}
    started = time.monotonic()
    last_error = ""
    for attempt in range(1, args.judge_max_retries + 1):
        try:
            async with semaphore:
                async with session.post(
                    f"{args.judge_base_url.rstrip('/')}/chat/completions",
                    json=payload,
                    headers=headers,
                ) as response:
                    body = await response.text()
                    result["judge_status_code"] = response.status
                    if response.status != 200:
                        raise RuntimeError(f"Judge HTTP {response.status}: {body[:2000]}")
                    completion = json.loads(body)
            message = completion["choices"][0]["message"]
            content = message.get("content") or message.get("reasoning_content") or ""
            judgment = parse_judgment(content)
            result.update(
                {
                    "judgment": judgment,
                    "judge_content": content,
                    "judge_reasoning_content": message.get("reasoning_content"),
                    "judge_completion_tokens": completion.get("usage", {}).get("completion_tokens", 0),
                    "predicted_answer": judgment["extracted_final_answer"],
                    "correct": float(judgment["correct"] == "yes"),
                    "judge_confidence": judgment["confidence"],
                    "judge_attempts": attempt,
                    "judge_elapsed_sec": round(time.monotonic() - started, 3),
                }
            )
            return
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < args.judge_max_retries:
                await asyncio.sleep(min(2 ** (attempt - 1), 8))
    result["judge_error"] = last_error
    result["judge_status_code"] = result.get("judge_status_code", 0)
    result["judge_attempts"] = args.judge_max_retries
    result["judge_elapsed_sec"] = round(time.monotonic() - started, 3)


async def evaluate_and_judge_one(
    generation_session: aiohttp.ClientSession,
    generation_semaphore: asyncio.Semaphore,
    judge_session: aiohttp.ClientSession,
    judge_semaphore: asyncio.Semaphore,
    args: Args,
    row: dict[str, Any],
    trial_index: int,
) -> dict[str, Any]:
    """Generate one response and send it to the external grader immediately."""
    result = await evaluate_one(
        generation_session,
        generation_semaphore,
        args,
        row,
        trial_index,
    )
    await judge_one(judge_session, judge_semaphore, args, row, result)
    return result


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [result for result in results if result.get("status_code") == 200]
    graded = [result for result in completed if result.get("correct") is not None]
    correct = sum(result.get("correct", 0.0) for result in graded)
    completion_tokens = sum(result.get("completion_tokens", 0) for result in completed)
    metrics: dict[str, int | float] = {
        "tasks_total": len(results),
        "problems_total": len({result["id"] for result in results}),
        "completed": len(completed),
        "errors": len(results) - len(completed),
        "request_success_rate": len(completed) / len(results) if results else 0.0,
        "graded": len(graded),
        "correct": correct,
        "accuracy": correct / len(graded) if graded else 0.0,
        "completion_tokens": completion_tokens,
    }

    judge_requested = [result for result in completed if result.get("judge_requested")]
    if judge_requested:
        judge_completed = [result for result in judge_requested if result.get("judge_status_code") == 200 and result.get("judgment") is not None]
        judge_confidences = [result["judge_confidence"] for result in judge_completed]
        metrics.update(
            {
                "judge_requested": len(judge_requested),
                "judge_completed": len(judge_completed),
                "judge_errors": len(judge_requested) - len(judge_completed),
                "judge_success_rate": len(judge_completed) / len(judge_requested),
                "judge_completion_tokens": sum(result.get("judge_completion_tokens", 0) for result in judge_completed),
                "mean_judge_confidence": sum(judge_confidences) / len(judge_confidences) if judge_confidences else 0.0,
            }
        )

    per_problem: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        per_problem[result["id"]].append(result)
    graded_problems = [trials for trials in per_problem.values() if any(trial.get("correct") is not None for trial in trials)]
    if graded_problems:
        problems_any_correct = sum(any(trial.get("correct") == 1.0 for trial in trials) for trials in graded_problems)
        metrics.update(
            {
                "problems_graded": len(graded_problems),
                "problems_any_correct": problems_any_correct,
                "problem_any_correct_rate": problems_any_correct / len(graded_problems),
            }
        )

    return {
        "metrics": metrics,
        "rewards": [result.get("correct") for result in results],
        "per_task": {f"{result['id']}:trial_{result.get('trial_index', 0)}": result for result in results},
    }


async def main_async(args: Args) -> None:
    if args.n_trials <= 0:
        raise ValueError("--n_trials must be positive")
    if args.concurrency <= 0 or args.judge_concurrency <= 0:
        raise ValueError("Concurrency values must be positive")
    if args.judge_max_retries <= 0:
        raise ValueError("--judge_max_retries must be positive")
    if args.judge_base_url is not None and not args.judge_model:
        raise ValueError("--judge_model is required when --judge_base_url is set")

    rows = load_rows(
        Path(args.input),
        multiple_choice_only=args.multiple_choice_only,
        max_tasks=args.max_tasks,
    )
    if not rows:
        raise ValueError("No HLE rows matched the requested filters")

    work_items = [(row, trial_index) for row in rows for trial_index in range(args.n_trials)]
    timeout = aiohttp.ClientTimeout(total=args.request_timeout_sec)
    connector = aiohttp.TCPConnector(limit=args.concurrency)
    semaphore = asyncio.Semaphore(args.concurrency)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        if args.judge_base_url is None:
            results = await asyncio.gather(*(evaluate_one(session, semaphore, args, row, trial_index) for row, trial_index in work_items))
        else:
            judge_timeout = aiohttp.ClientTimeout(total=args.judge_request_timeout_sec)
            judge_connector = aiohttp.TCPConnector(limit=args.judge_concurrency)
            judge_semaphore = asyncio.Semaphore(args.judge_concurrency)
            async with aiohttp.ClientSession(timeout=judge_timeout, connector=judge_connector) as judge_session:
                results = await asyncio.gather(
                    *(
                        evaluate_and_judge_one(
                            session,
                            semaphore,
                            judge_session,
                            judge_semaphore,
                            args,
                            row,
                            trial_index,
                        )
                        for row, trial_index in work_items
                    )
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
