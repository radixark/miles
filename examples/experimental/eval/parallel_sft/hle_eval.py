"""Evaluate HLE generations with an optional external OpenAI-compatible judge."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
import re
import time
from collections import defaultdict
from collections.abc import Awaitable
from pathlib import Path
from typing import Any

import aiohttp
from tap import Tap
from transformers import AutoTokenizer

JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "correct": {"type": "string", "enum": ["yes", "no"]},
    },
    "required": [
        "reasoning",
        "correct",
    ],
    "additionalProperties": False,
}

JUDGE_PROMPT = """Compare the candidate final answer with the reference answer.
Do not solve the original problem or infer an answer that the candidate did not
state. Small numerical rounding differences and mathematically equivalent forms
are acceptable.

Candidate final answer:
{candidate_answer}

Reference answer:
{reference_answer}

Return a JSON object with exactly these fields:
- reasoning: a brief comparison of the candidate and reference answers
- correct: "yes" or "no"
"""

# Ordinary English words with no bearing on any answer. Rendered in a random
# order they read as nonsense, which is the point: the filler must occupy context
# without giving the model anything to reason about.
FILLER_WORDS: tuple[str, ...] = (
    "amber",
    "anchor",
    "apron",
    "arbor",
    "attic",
    "badger",
    "bagel",
    "bamboo",
    "barrel",
    "basket",
    "beacon",
    "beetle",
    "bellow",
    "birch",
    "biscuit",
    "blanket",
    "blossom",
    "boulder",
    "bramble",
    "brass",
    "breeze",
    "bridle",
    "brook",
    "bucket",
    "bundle",
    "burrow",
    "butter",
    "cabin",
    "candle",
    "canvas",
    "canyon",
    "carpet",
    "cattle",
    "cedar",
    "cellar",
    "chalk",
    "chestnut",
    "chimney",
    "cider",
    "cinder",
    "clover",
    "cobble",
    "compass",
    "copper",
    "cotton",
    "cradle",
    "crimson",
    "crumble",
    "cushion",
    "dapple",
    "dawn",
    "dimple",
    "drift",
    "drizzle",
    "dusk",
    "ember",
    "fable",
    "falcon",
    "feather",
    "fennel",
    "ferry",
    "fiddle",
    "flannel",
    "flicker",
    "fossil",
    "furrow",
    "gable",
    "garland",
    "gentle",
    "ginger",
    "glimmer",
    "gravel",
    "grove",
    "hammock",
    "harbor",
    "harvest",
    "hazel",
    "heather",
    "hollow",
    "honey",
    "hush",
    "ivory",
    "jasmine",
    "jingle",
    "juniper",
    "kettle",
    "kindle",
    "lantern",
    "lattice",
    "lavender",
    "ledger",
    "lemon",
    "linen",
    "lumber",
    "mantle",
    "maple",
    "marble",
    "meadow",
    "mellow",
    "mingle",
    "mitten",
    "mossy",
    "murmur",
    "nectar",
    "nestle",
    "nimble",
    "nutmeg",
    "oaken",
    "orchard",
    "otter",
    "paddle",
    "pantry",
    "parsley",
    "pebble",
    "pepper",
    "pewter",
    "pickle",
    "pillow",
    "plaster",
    "plume",
    "pond",
    "porch",
    "pottery",
    "prairie",
    "puddle",
    "quilt",
    "quiver",
    "rafter",
    "ramble",
    "raven",
    "ribbon",
    "ripple",
    "river",
    "rubble",
    "rustic",
    "saddle",
    "saffron",
    "sandal",
    "satchel",
    "scarlet",
    "shingle",
    "shutter",
    "sizzle",
    "slumber",
    "smolder",
    "sorrel",
    "sparrow",
    "spindle",
    "spruce",
    "stable",
    "steeple",
    "stitch",
    "sturdy",
    "sundial",
    "supper",
    "swallow",
    "tangle",
    "tassel",
    "teapot",
    "thimble",
    "thistle",
    "thunder",
    "timber",
    "tinder",
    "toffee",
    "trellis",
    "trestle",
    "trickle",
    "trough",
    "tumble",
    "tundra",
    "twilight",
    "umber",
    "velvet",
    "wagon",
    "walnut",
    "wander",
    "warble",
    "whisker",
    "whistle",
    "wicker",
    "willow",
    "winnow",
    "wobble",
    "woolen",
    "yarrow",
    "yonder",
    "zephyr",
)

DEFAULT_FILLER_TRAILER = "The text above, after the question, is meaningless filler. Answer the question before it."


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
    max_context_length: int | None = None
    tokenizer_path: str | None = None
    incremental: bool = False
    generations_jsonl: str | None = None
    temperature: float = 0.0
    request_timeout_sec: int = 3600
    disable_thinking: bool = False
    multiple_choice_only: bool = False

    # Context-position experiments: pad every templated prompt to this many
    # tokens with nonsense words placed after the question, so the model must
    # answer from deep inside its context window. Requires --max_context_length.
    filler_target_prompt_tokens: int | None = None
    filler_seed: str = "hle-filler-v1"
    filler_trailer: str = DEFAULT_FILLER_TRAILER

    # ``judge_base_url`` is an OpenAI-compatible base URL, including ``/v1``.
    # This works with an independently hosted SGLang server or router.
    judge_base_url: str | None = None
    judge_model: str | None = None
    judge_api_key: str = "dummy"
    judge_api_key_env: str = "HLE_JUDGE_API_KEY"
    judge_concurrency: int = 16
    judge_max_qps: float = 0.0
    judge_max_tokens: int = 16384
    judge_temperature: float = 0.0
    judge_max_tokens_param: str = "max_tokens"
    judge_reasoning_effort: str | None = None
    judge_omit_temperature: bool = False
    judge_request_timeout_sec: int = 3600
    judge_max_retries: int = 3


class RequestStartRateLimiter:
    """Enforce a process-wide minimum interval between request starts."""

    def __init__(self, max_qps: float) -> None:
        self._minimum_interval = 1.0 / max_qps if max_qps > 0 else 0.0
        self._next_start = 0.0
        self._lock = asyncio.Lock()

    async def wait(self) -> None:
        if self._minimum_interval == 0:
            return
        async with self._lock:
            delay = self._next_start - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
            self._next_start = time.monotonic() + self._minimum_interval


def extract_final_answer(text: str) -> str | None:
    """Extract an answer only when the final non-empty line follows the contract."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return None
    match = re.fullmatch(r"(?:\*\*)?final\s+answer(?:\*\*)?\s*:\s*(.+)", lines[-1], re.IGNORECASE)
    if match is None:
        return None
    answer = match.group(1).strip()
    if answer.startswith("**") and answer.endswith("**") and len(answer) > 4:
        answer = answer[2:-2].strip()
    return answer or None


def extract_choice(text: str) -> str | None:
    """Extract a final A-Z multiple-choice answer without scanning reasoning."""
    answer = extract_final_answer(text)
    if answer is None:
        return None
    match = re.fullmatch(r"(?:\\boxed\s*\{\s*)?([A-Z])(?:\s*\})?[.)]?", answer, re.IGNORECASE)
    return match.group(1).upper() if match is not None else None


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
        final_answer_instruction = "LETTER"
    else:
        final_answer_instruction = "ANSWER"
    return (
        f"{row['question']}\n\n"
        "Solve the problem carefully. Your final non-empty line must use exactly "
        f"this format:\nFinal answer: {final_answer_instruction}\n"
        "Do not write anything after the final-answer line."
    )


def judge_prompt(row: dict[str, Any], candidate_answer: str) -> str:
    """Build a reference-based prompt that excludes the model's reasoning trace."""
    return JUDGE_PROMPT.format(
        candidate_answer=candidate_answer,
        reference_answer=row["answer"],
    )


def judge_payload(args: Args, row: dict[str, Any], candidate_answer: str) -> dict[str, Any]:
    """Build an SGLang/OpenAI-compatible schema-constrained grading request."""
    if args.judge_model is None:
        raise ValueError("--judge_model is required when --judge_base_url is set")
    payload = {
        "model": args.judge_model,
        "messages": [{"role": "user", "content": judge_prompt(row, candidate_answer)}],
        args.judge_max_tokens_param: args.judge_max_tokens,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "hle_judgment", "schema": JUDGE_SCHEMA, "strict": True},
        },
    }
    if not args.judge_omit_temperature:
        payload["temperature"] = args.judge_temperature
    if args.judge_reasoning_effort is not None:
        payload["reasoning_effort"] = args.judge_reasoning_effort
    return payload


def render_filler(rng: random.Random, word_count: int) -> str:
    """Render ``word_count`` random dictionary words as sentence-shaped nonsense."""
    words = [rng.choice(FILLER_WORDS) for _ in range(word_count)]
    sentences = []
    index = 0
    while index < len(words):
        chunk = words[index : index + rng.randint(6, 14)]
        sentences.append(chunk[0].capitalize() + " " + " ".join(chunk[1:]) + ".")
        index += len(chunk)
    paragraphs = [" ".join(sentences[start : start + 8]) for start in range(0, len(sentences), 8)]
    return "\n\n".join(paragraphs)


def compose_filler_message(base_prompt: str, filler: str, trailer: str) -> str:
    """Place the filler after the question so the task sits at the start of the context."""
    parts = [base_prompt, filler]
    if trailer:
        parts.append(trailer)
    return "\n\n".join(parts)


def templated_prompt_tokens(tokenizer: Any, args: Args, content: str) -> int:
    return len(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=not args.disable_thinking,
            return_dict=False,
        )
    )


def build_filler_message(tokenizer: Any, args: Args, row: dict[str, Any], base_prompt: str) -> tuple[str, int]:
    """Pad the prompt with filler until the templated length reaches the target.

    Returns the message and its templated token count. The count lands on the
    target exactly when a word boundary allows it and otherwise a few tokens
    below it; it never exceeds the target.
    """
    target = args.filler_target_prompt_tokens
    assert target is not None
    rng = random.Random(f"{args.filler_seed}:{row['id']}")
    overhead = templated_prompt_tokens(tokenizer, args, compose_filler_message(base_prompt, "x", args.filler_trailer))
    need = target - overhead
    if need <= 0:
        raise ValueError(f"HLE row {row['id']} already exceeds the filler target: {overhead} prompt tokens")
    # Every word costs at least one token, so this stream is long enough.
    text = render_filler(rng, need + 64)
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = encoded["offset_mapping"]
    best: tuple[str, int] | None = None
    seen: set[int] = set()
    token_count = min(need, len(offsets))
    while token_count not in seen:
        seen.add(token_count)
        end = offsets[max(token_count, 1) - 1][1]
        cut = text.rfind(" ", 0, end)
        message = compose_filler_message(base_prompt, text[: cut if cut > 0 else end].rstrip(), args.filler_trailer)
        count = templated_prompt_tokens(tokenizer, args, message)
        if count == target:
            return message, count
        if count < target and (best is None or count > best[1]):
            best = (message, count)
        token_count = min(max(token_count + target - count, 1), len(offsets))
    if best is None:
        raise ValueError(f"HLE row {row['id']}: could not fit filler under {target} prompt tokens")
    return best


def prepare_context_budgets(args: Args, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Reserve the exact templated prompt length before allocating output tokens."""
    if args.max_context_length is None:
        return rows
    if args.max_context_length <= 0 or not args.tokenizer_path:
        raise ValueError("A positive --max_context_length requires --tokenizer_path")
    if (
        args.filler_target_prompt_tokens is not None
        and not 0 < args.filler_target_prompt_tokens < args.max_context_length
    ):
        raise ValueError("--filler_target_prompt_tokens must be positive and below --max_context_length")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, local_files_only=True)
    prepared = []
    for row in rows:
        message = generation_prompt(row)
        base_prompt_tokens = templated_prompt_tokens(tokenizer, args, message)
        prompt_tokens = base_prompt_tokens
        if args.filler_target_prompt_tokens is not None:
            message, prompt_tokens = build_filler_message(tokenizer, args, row, message)
        output_tokens = min(args.max_tokens, args.max_context_length - prompt_tokens)
        if output_tokens <= 0:
            raise ValueError(f"HLE row {row['id']} leaves no output budget: {prompt_tokens} prompt tokens")
        prepared.append(
            {
                **row,
                "_message": message,
                "_prompt_tokens": prompt_tokens,
                "_filler_tokens": prompt_tokens - base_prompt_tokens,
                "_max_tokens": output_tokens,
            }
        )
    return prepared


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
    message = row.get("_message") or generation_prompt(row)
    payload: dict[str, Any] = {
        "model": args.model,
        "messages": [{"role": "user", "content": message}],
        "max_tokens": row.get("_max_tokens", args.max_tokens),
        "temperature": args.temperature,
    }
    if args.disable_thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    headers = {"Authorization": f"Bearer {args.api_key}"}
    started = time.monotonic()
    result: dict[str, Any] = {
        "id": row["id"],
        "trial_index": trial_index,
        "answer": row["answer"],
        "answer_type": row["answer_type"],
        "requested_max_tokens": payload["max_tokens"],
        "expected_prompt_tokens": row.get("_prompt_tokens"),
        "filler_tokens": row.get("_filler_tokens", 0),
        "message_sha256": hashlib.sha256(message.encode()).hexdigest(),
        "max_context_length": args.max_context_length,
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
                choices = completion.get("choices")
                if not choices or not isinstance(choices[0].get("message"), dict):
                    raise ValueError("Generation response has no chat message")
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
            "prompt_tokens": completion.get("usage", {}).get("prompt_tokens"),
            "usage": completion.get("usage", {}),
            "finish_reason": completion["choices"][0].get("finish_reason"),
            "response_model": completion.get("model"),
        }
    )
    if args.max_context_length is not None:
        actual_prompt = result["prompt_tokens"]
        if actual_prompt is None or actual_prompt != row["_prompt_tokens"]:
            result["context_budget_error"] = "Endpoint prompt-token count differs from the supplied tokenizer"
        elif actual_prompt + result["completion_tokens"] > args.max_context_length:
            result["context_budget_error"] = "Endpoint exceeded the requested total context budget"
        if "context_budget_error" in result:
            result["status_code"] = 0
            result["error"] = result["context_budget_error"]
            return result
    predicted = extract_final_answer(content)
    result["predicted_answer"] = predicted
    result["final_answer_format_valid"] = predicted is not None
    if predicted is None:
        result["correct"] = 0.0
        result["final_answer_error"] = "The final non-empty line did not match 'Final answer: ANSWER'"
        return result

    if row.get("answer_type") == "multipleChoice":
        predicted_choice = extract_choice(content)
        result["predicted_answer"] = predicted_choice
        result["direct_predicted_answer"] = predicted_choice
        result["direct_correct"] = float(predicted_choice == str(row["answer"]).strip().upper())
        result["correct"] = result["direct_correct"]
    return result


async def judge_one(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    rate_limiter: RequestStartRateLimiter,
    args: Args,
    row: dict[str, Any],
    result: dict[str, Any],
) -> None:
    """Grade one successful generation through the external judge endpoint."""
    if result.get("status_code") != 200 or args.judge_base_url is None:
        return
    if row.get("answer_type") == "multipleChoice" or result.get("predicted_answer") is None:
        return

    result["judge_requested"] = True
    payload = judge_payload(args, row, result["predicted_answer"])
    headers = {"Authorization": f"Bearer {resolve_judge_api_key(args)}"}
    started = time.monotonic()
    last_error = ""
    for attempt in range(1, args.judge_max_retries + 1):
        try:
            async with semaphore:
                await rate_limiter.wait()
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
                    "judge_usage": completion.get("usage", {}),
                    "judge_response_model": completion.get("model"),
                    "correct": float(judgment["correct"] == "yes"),
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
    judge_rate_limiter: RequestStartRateLimiter,
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
    if args.generations_jsonl is not None:
        with Path(args.generations_jsonl).open("a") as stream:
            stream.write(json.dumps(result) + "\n")
    await judge_one(judge_session, judge_semaphore, judge_rate_limiter, args, row, result)
    return result


async def collect_results(work: list[Awaitable[dict[str, Any]]], args: Args) -> list[dict[str, Any]]:
    """Persist completed trials as they arrive, without repeatedly serializing traces."""
    if not args.incremental:
        return await asyncio.gather(*work)
    output_path = Path(args.output_jsonl)
    progress_path = Path(args.summary_json).with_suffix(".progress.json")
    results = []
    with output_path.open("x") as stream:
        print(json.dumps({"event": "started", "trials": len(work)}), flush=True)
        for pending in asyncio.as_completed(work):
            result = await pending
            results.append(result)
            stream.write(json.dumps(result) + "\n")
            stream.flush()
            metrics = summarize(results)["metrics"]
            progress = {"planned_trials": len(work), "finished_trials": len(results), "metrics": metrics}
            temporary = progress_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(progress, indent=2))
            temporary.replace(progress_path)
            print(
                json.dumps(
                    {"event": "trial_finished", "id": result["id"], "trial_index": result["trial_index"], **progress}
                ),
                flush=True,
            )
    return results


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
        judge_completed = [
            result
            for result in judge_requested
            if result.get("judge_status_code") == 200 and result.get("judgment") is not None
        ]
        metrics.update(
            {
                "judge_requested": len(judge_requested),
                "judge_completed": len(judge_completed),
                "judge_errors": len(judge_requested) - len(judge_completed),
                "judge_success_rate": len(judge_completed) / len(judge_requested),
                "judge_completion_tokens": sum(result.get("judge_completion_tokens", 0) for result in judge_completed),
            }
        )

    per_problem: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        per_problem[result["id"]].append(result)
    graded_problems = [
        trials for trials in per_problem.values() if any(trial.get("correct") is not None for trial in trials)
    ]
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
    if args.judge_max_qps < 0:
        raise ValueError("--judge_max_qps must be non-negative")
    if args.judge_max_retries <= 0:
        raise ValueError("--judge_max_retries must be positive")
    if args.judge_base_url is not None and not args.judge_model:
        raise ValueError("--judge_model is required when --judge_base_url is set")
    if args.judge_max_tokens_param not in {"max_tokens", "max_completion_tokens"}:
        raise ValueError("--judge_max_tokens_param must be max_tokens or max_completion_tokens")
    if args.max_tokens <= 0 or args.judge_max_tokens <= 0:
        raise ValueError("Output token limits must be positive")
    if args.filler_target_prompt_tokens is not None and args.max_context_length is None:
        raise ValueError("--filler_target_prompt_tokens requires --max_context_length and --tokenizer_path")
    if args.incremental and Path(args.output_jsonl).exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_jsonl}; use a new output path")
    if args.generations_jsonl is not None:
        generation_path = Path(args.generations_jsonl)
        generation_path.parent.mkdir(parents=True, exist_ok=True)
        generation_path.touch(exist_ok=False)
    Path(args.output_jsonl).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)

    rows = load_rows(
        Path(args.input),
        multiple_choice_only=args.multiple_choice_only,
        max_tasks=args.max_tasks,
    )
    if not rows:
        raise ValueError("No HLE rows matched the requested filters")
    rows = prepare_context_budgets(args, rows)

    work_items = [(row, trial_index) for row in rows for trial_index in range(args.n_trials)]
    timeout = aiohttp.ClientTimeout(total=args.request_timeout_sec)
    connector = aiohttp.TCPConnector(limit=args.concurrency)
    semaphore = asyncio.Semaphore(args.concurrency)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        if args.judge_base_url is None:
            results = await collect_results(
                [evaluate_one(session, semaphore, args, row, trial_index) for row, trial_index in work_items], args
            )
        else:
            judge_timeout = aiohttp.ClientTimeout(total=args.judge_request_timeout_sec)
            judge_connector = aiohttp.TCPConnector(limit=args.judge_concurrency)
            judge_semaphore = asyncio.Semaphore(args.judge_concurrency)
            judge_rate_limiter = RequestStartRateLimiter(args.judge_max_qps)
            async with aiohttp.ClientSession(timeout=judge_timeout, connector=judge_connector) as judge_session:
                results = await collect_results(
                    [
                        evaluate_and_judge_one(
                            session,
                            semaphore,
                            judge_session,
                            judge_semaphore,
                            judge_rate_limiter,
                            args,
                            row,
                            trial_index,
                        )
                        for row, trial_index in work_items
                    ],
                    args,
                )

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not args.incremental:
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
