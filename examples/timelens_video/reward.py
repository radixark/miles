"""TimeLens temporal-grounding reward: IoU of the first predicted span with the ground-truth span.

Parsing and IoU follow TimeLens ``training/utils/parser.py`` and its ``tiou`` reward
(``training/train/reward_funcs.py``): an unparsable answer or one with start >= end scores 0.
"""

import json
import re

from miles.ray.rollout.train_data_conversion import _post_process_rewards
from miles.utils.types import Sample

_TIME_COLON = re.compile(r"\b(\d{1,2}:\d{2}:\d{2}(?:\.\d+)?|\d{1,2}:\d{2}(?:\.\d+)?)\b")
_SPAN_PATTERNS = (r"(\d+\.?\d*)\s*-\s*(\d+\.?\d*)", r"(\d+\.?\d*)\s+to\s+(\d+\.?\d*)")
_NUMBER = re.compile(r"\b(\d+\.\d+|\d+)\b")


def extract_answer(content: str) -> str:
    match = re.match(r"<think>.*?</think>\s*<answer>(.*?)</answer>", content, re.DOTALL)
    return match.group(1) if match else content


def extract_time(paragraph: str) -> list[tuple[float, float]]:
    paragraph = paragraph.lower().replace("to", "-")
    matches = _TIME_COLON.findall(paragraph)
    matches = matches[: len(matches) // 2 * 2]
    if matches:
        seconds = []
        for t in matches:
            parts = t.split(":")
            if len(parts) == 3:
                seconds.append(int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2]))
            else:
                seconds.append(int(parts[0]) * 60 + float(parts[1]))
        return [(seconds[i], seconds[i + 1]) for i in range(0, len(seconds), 2)]
    for pattern in _SPAN_PATTERNS:
        spans = re.findall(pattern, paragraph)
        if spans:
            return [(float(s), float(e)) for s, e in spans]
    numbers = _NUMBER.findall(paragraph)
    numbers = numbers[: len(numbers) // 2 * 2]
    return [(float(numbers[i]), float(numbers[i + 1])) for i in range(0, len(numbers), 2)]


def iou(a, b) -> float:
    denom = max(a[1], b[1]) - min(a[0], b[0])
    if denom <= 0:
        return 0.0
    return max(min(a[1], b[1]) - max(a[0], b[0]), 0) / denom


def tiou(response: str, span) -> float:
    spans = extract_time(extract_answer(response))
    if not spans or spans[0][0] >= spans[0][1]:
        return 0.0
    return iou(span, spans[0])


async def timelens_tiou(args, sample, **kwargs) -> float:
    label = json.loads(sample.label) if isinstance(sample.label, str) else sample.label
    return tiou(sample.response, label["span"])


def zero_truncated_rewards(args, samples: list[Sample]):
    """Training-only reward post-process: a response cut at the length limit scores 0, then GRPO normalizes.

    Only the first predicted span is scored, so text after it is free and the policy drifts into explanations
    that run to the length limit. Eval keeps the TimeLens scoring.
    """
    for sample in samples:
        if sample.status == Sample.Status.TRUNCATED:
            sample.reward = 0.0
    return _post_process_rewards(args, samples, None)
