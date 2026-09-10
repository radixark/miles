"""Grade explicit final Sokoban answers with NeMo Gym's real /verify endpoint."""

import asyncio
import math
import os
import re
from copy import deepcopy
from enum import Enum
from typing import Any, Protocol

import httpx

_GRADING_KEYS = frozenset(
    {"sokoban_extracted_answer", "sokoban_reward", "sokoban_grading_status", "sokoban_grader_version"}
)


class _SokobanSample(Protocol):
    """The reward hook's input contract, without importing GPU dependencies."""

    response: str
    metadata: dict[str, Any]
    label: str | None
    status: Enum | str


class InvalidSokobanAnswer(ValueError):
    """A model completion that cannot provide an unambiguous final move list."""


def extract_final_answer(response: str) -> str:
    """Extract one answer after the reasoning boundary; never search reasoning.

    This hook expects a think-delimited reasoning model. The opening <think>
    may be in the prompt, but exactly one closing </think> must be generated.
    Only uppercase movement letters and whitespace are permitted in the answer.
    """
    if not isinstance(response, str):
        raise TypeError("Sokoban response must be a string")
    if response.count("</think>") != 1:
        raise InvalidSokobanAnswer("missing_or_ambiguous_reasoning_end")

    final = response.split("</think>", 1)[1].strip()
    final = final.removesuffix("<|im_end|>").strip()
    if "<think>" in final or "<|im_end|>" in final:
        raise InvalidSokobanAnswer("invalid_final_boundary")
    tags = re.findall(r"</?answer\b[^>]*>", final, flags=re.IGNORECASE)
    if tags != ["<answer>", "</answer>"]:
        raise InvalidSokobanAnswer("missing_or_multiple_answers")

    match = re.search(r"<answer>([^<>]*)</answer>", final)
    if match is None:
        raise InvalidSokobanAnswer("malformed_answer_tags")
    answer = match.group(1)
    if re.fullmatch(r"[UDLR \t\r\n\f\v]+", answer) is None:
        raise InvalidSokobanAnswer("invalid_move_characters")
    moves = "".join(answer.split())
    if not moves:
        raise InvalidSokobanAnswer("empty_answer")
    return moves


def build_verify_request(sample: _SokobanSample) -> dict[str, Any]:
    """Send only canonical moves; keep the original completion untouched."""
    if not isinstance(sample.metadata, dict) or sample.metadata.get("source_dataset") != "sokoban":
        raise ValueError("Sokoban grading requires metadata.source_dataset='sokoban'")
    for key in ("sokoban_question", "gamestr"):
        if not isinstance(sample.metadata.get(key), str) or not sample.metadata[key].strip():
            raise ValueError(f"Sokoban grading requires nonempty metadata.{key}")
    if not isinstance(sample.label, str) or not sample.label.strip():
        raise ValueError("Sokoban grading requires a reference answer")

    status = sample.status.value if isinstance(sample.status, Enum) else sample.status
    if status == "truncated":
        raise InvalidSokobanAnswer("truncated")
    if status != "completed":
        raise ValueError(f"Unexpected Sokoban sample status: {status!r}")

    moves = extract_final_answer(sample.response)
    question = sample.metadata["sokoban_question"]
    return {
        "responses_create_params": {"input": [{"role": "user", "content": question}]},
        "question": question,
        "answer": sample.label,
        "metadata": deepcopy({key: value for key, value in sample.metadata.items() if key not in _GRADING_KEYS}),
        "response": {
            "id": "sokoban-response",
            "created_at": 0,
            "model": "policy",
            "object": "response",
            "status": "completed",
            "parallel_tool_calls": False,
            "tool_choice": "none",
            "tools": [],
            "output": [
                {
                    "id": "sokoban-message",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": f"<answer>{moves}</answer>", "annotations": []}],
                }
            ],
        },
    }


def _binary_reward(value: object) -> float:
    if type(value) not in (int, float) or value not in (0, 1) or not math.isfinite(value):
        raise ValueError("NeMo Gym must return a finite numeric binary reward, not a boolean or string")
    return float(value)


def _validate_grade(result: object, expected_answer: str) -> float:
    if not isinstance(result, dict) or result.get("task_name") != "sokoban":
        raise ValueError("NeMo Gym returned an invalid Sokoban response")
    masked = result.get("mask_sample")
    failure = result.get("failure_reason")
    if (masked is not None and masked is not False) or (failure is not None and failure != ""):
        raise RuntimeError("NeMo Gym reported a masked or failed verification")
    reward = _binary_reward(result.get("reward"))
    if _binary_reward(result.get("score")) != reward:
        raise ValueError("NeMo Gym score and reward disagree")
    if result.get("extracted_answer") != expected_answer:
        raise ValueError("NeMo Gym extracted a different answer from the submitted moves")
    return reward


def _record_grade(sample: _SokobanSample, *, reward: float, answer: str, status: str) -> None:
    sample.metadata.update(
        sokoban_extracted_answer=answer,
        sokoban_reward=reward,
        sokoban_grading_status=status,
        sokoban_grader_version="final-answer-v1",
    )


async def _score(client: httpx.AsyncClient, url: str, sample: _SokobanSample) -> float:
    try:
        payload = build_verify_request(sample)
    except InvalidSokobanAnswer as error:
        _record_grade(sample, reward=0.0, answer="", status=str(error))
        return 0.0

    submitted = payload["response"]["output"][0]["content"][0]["text"]
    moves = submitted.removeprefix("<answer>").removesuffix("</answer>")
    response = await client.post(f"{url.rstrip('/')}/verify", json=payload)
    response.raise_for_status()
    reward = _validate_grade(response.json(), moves)
    _record_grade(sample, reward=reward, answer=moves, status="verified")
    return reward


async def reward_func(args: Any, samples: _SokobanSample | list[_SokobanSample], **kwargs: Any) -> float | list[float]:
    """Support individual and batched rewards; infrastructure errors fail loudly."""
    batch = samples if isinstance(samples, list) else [samples]
    if not batch:
        return []
    url = os.environ["NEMO_GYM_SOKOBAN_URL"]
    async with httpx.AsyncClient(timeout=30.0, trust_env=False) as client:
        rewards = await asyncio.gather(*(_score(client, url, sample) for sample in batch))
    return rewards if isinstance(samples, list) else rewards[0]
