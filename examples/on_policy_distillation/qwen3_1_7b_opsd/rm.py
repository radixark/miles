"""Teacher scoring for privileged-context OPSD, plus accuracy for held-out rows.

The student generates from the problem alone; the teacher scores that same response on a
prompt that also contains the reference solution. Both prompts are rendered by
prepare_data.py, so the teacher prompt arrives on the sample as metadata.

The divergence is a top-k reverse KL, clipped per vocabulary entry. The teacher's top-k
comes from the scoring call below; the student's comes free with generation, because
--opd-log-prob-top-k puts top_logprobs_num on the rollout request. Neither side needs a
second scoring call, which is what keeps this expressible as a reward function. It is
handed to miles as sample.opd_reverse_kl, the per-token divergence --use-opd subtracts
from the advantage. Held-out rows are scored for accuracy here because --custom-rm-path
is consulted unconditionally, so a per-sample rm_type would never be reached.
"""

import math
from typing import Any

import torch
from math_verify import parse, verify

from miles.utils.http_utils import post
from miles.utils.processing_utils import load_tokenizer
from miles.utils.types import Sample

# The paper's jsd_token_clip for Qwen3-1.7B. Token-level divergence is heavy-tailed, so
# without a ceiling a few stylistic tokens carry the update instead of the content ones.
TAU = 0.05


def _extract_boxed(text: str) -> str | None:
    start = text.rfind("\\boxed{")
    if start < 0:
        return None
    depth = 0
    for i in range(start + 6, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 7 : i].strip()
    return None


def _is_correct(response: str, label: str) -> float:
    predicted = _extract_boxed(response)
    if predicted is None:
        return 0.0
    # parsing_timeout=None is required: the default uses signal.alarm(), which only works
    # on the main thread, and reward functions run on a worker thread.
    gold = parse(f"${label}$", fallback_mode="no_fallback", parsing_timeout=None)
    guess = parse(f"${predicted}$", fallback_mode="no_fallback", parsing_timeout=None)
    return 1.0 if verify(gold, guess, timeout_seconds=None) else 0.0


def _per_position_maps(entries: list, response_length: int) -> list[dict[int, float]]:
    trimmed = entries[-response_length:] if response_length > 0 else []
    return [{int(e[1]): float(e[0]) for e in (position or [])} for position in trimmed]


async def _score_teacher(args: Any, sample: Sample) -> None:
    tokenizer = load_tokenizer(args.hf_checkpoint, chat_template_path=args.chat_template_path)
    prompt_ids = tokenizer.encode(sample.metadata["teacher_prompt"], add_special_tokens=False)
    response_ids = sample.tokens[len(sample.tokens) - sample.response_length :]

    top_k = args.opd_log_prob_top_k
    # No lora_path, so the teacher is the base weights, which LoRA keeps frozen.
    response = await post(
        args.rm_url,
        {
            "input_ids": list(prompt_ids) + list(response_ids),
            "sampling_params": {"temperature": 0, "max_new_tokens": 0, "skip_special_tokens": False},
            "return_logprob": True,
            "top_logprobs_num": top_k,
            "logprob_start_len": len(prompt_ids) - 1,
        },
    )

    scored = response["meta_info"]["input_token_logprobs"][-sample.response_length :]
    # The teacher must have scored the student's own tokens; a mismatch means the
    # privileged prompt shifted the response and every log-prob would be off by position.
    assert [int(entry[1]) for entry in scored] == list(response_ids), "teacher/student token mismatch"

    teacher_top = _per_position_maps(response["meta_info"]["input_top_logprobs"], sample.response_length)
    student_top = _per_position_maps(sample.metadata["opd_student_top_logprobs"], sample.response_length)

    # Reverse KL over the ids both sides ranked, weighted by the student's own probability,
    # and clipped per entry before the sum so no single entry can dominate the position.
    divergence = []
    for teacher_pos, student_pos in zip(teacher_top, student_top, strict=True):
        total = 0.0
        for token_id, student_logp in student_pos.items():
            teacher_logp = teacher_pos.get(token_id)
            if teacher_logp is None:
                continue
            total += min(math.exp(student_logp) * (student_logp - teacher_logp), TAU)
        divergence.append(total)
    sample.opd_reverse_kl = torch.tensor(divergence, dtype=torch.float32)


async def reward_func(args: Any, sample: Sample, **kwargs: Any) -> float:
    if sample.metadata.get("opsd_eval"):
        return _is_correct(sample.response or "", str(sample.label))
    await _score_teacher(args, sample)
    return 0.0
