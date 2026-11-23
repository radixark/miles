"""Rewards of the math recipes: Miles' boxed grade (``--rm-type math``) with one shaping term each.

overlong_score (DeepSeek-V4-Flash): DAPO's soft overlong penalty on top of the grade. The penalty is 0 up to
``--rollout-max-response-len`` minus OVERLONG_BUFFER tokens, then falls linearly to -OVERLONG_FACTOR at the cap.

finish_or_fail_score (GLM-5.2): a response cut at the length cap scores 0. GLM-5.2 writes boxed answers while it is
still thinking, so the plain grade would credit a guess boxed early in a response that runs into a short cap.

Both return the training reward under ``score`` and the plain grade under ``acc``; launch with
``--reward-key score --eval-reward-key acc``.
"""

import os

from miles.rollout.rm_hub.math_utils import grade_answer_verl
from miles.utils.types import Sample

OVERLONG_BUFFER = int(os.environ.get("OVERLONG_BUFFER", "4096"))
OVERLONG_FACTOR = float(os.environ.get("OVERLONG_FACTOR", "1.0"))


def _grade(sample: Sample) -> float:
    return 1.0 if grade_answer_verl(sample.response, sample.label) else 0.0


def _overlong(args, sample: Sample) -> dict:
    acc = _grade(sample)
    start = args.rollout_max_response_len - OVERLONG_BUFFER
    penalty = min(-(sample.response_length - start) / OVERLONG_BUFFER * OVERLONG_FACTOR, 0.0)
    return {"score": acc + penalty, "acc": acc, "overlong_penalty": penalty}


def _finish_or_fail(args, sample: Sample) -> dict:
    truncated = sample.status == Sample.Status.TRUNCATED
    acc = 0.0 if truncated else _grade(sample)
    return {"score": acc, "acc": acc, "truncated": float(truncated)}


async def overlong_score(args, sample_or_samples, **kwargs):
    if isinstance(sample_or_samples, list):
        return [_overlong(args, s) for s in sample_or_samples]
    return _overlong(args, sample_or_samples)


async def finish_or_fail_score(args, sample_or_samples, **kwargs):
    if isinstance(sample_or_samples, list):
        return [_finish_or_fail(args, s) for s in sample_or_samples]
    return _finish_or_fail(args, sample_or_samples)
