"""Teacher scoring for privileged-context OPSD, plus accuracy for held-out rows.

The student generates from the problem alone; the teacher scores that same response on a
prompt that also contains the reference solution. Both prompts are rendered by
prepare_data.py, so the teacher prompt arrives on the sample as metadata.

The divergence is forward KL over the teacher's top-k support, clipped per vocabulary
entry, which is the objective the paper adopts. It is computed here and handed to miles
as sample.opd_reverse_kl, the per-token divergence that --use-opd subtracts from the
advantage. Held-out rows are scored for accuracy here because --custom-rm-path is
consulted unconditionally, so a per-sample rm_type would never be reached.
"""

import math
from typing import Any

import torch
from math_verify import parse, verify

from miles.utils.http_utils import post
from miles.utils.lora import LORA_ADAPTER_NAME
from miles.utils.processing_utils import load_tokenizer
from miles.utils.types import Sample

# Teacher support width and the per-entry clip from the paper's Qwen3-1.7B configuration.
TOP_K = 16
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


def _per_position_maps(response: dict, field: str, response_length: int) -> list[dict[int, float]]:
    entries = response["meta_info"][field][-response_length:]
    return [{int(e[1]): float(e[0]) for e in (position or [])} for position in entries]


async def _score_teacher(args: Any, sample: Sample) -> None:
    tokenizer = load_tokenizer(args.hf_checkpoint, chat_template_path=args.chat_template_path)
    teacher_ids = tokenizer.encode(sample.metadata["teacher_prompt"], add_special_tokens=False)
    response_length = sample.response_length

    # No lora_path, so the teacher is the base weights, which LoRA keeps frozen.
    teacher = await post(
        args.rm_url,
        {
            "input_ids": list(teacher_ids) + list(sample.tokens[-response_length:]),
            "sampling_params": {"temperature": 0, "max_new_tokens": 0, "skip_special_tokens": False},
            "return_logprob": True,
            "top_logprobs_num": TOP_K,
            "logprob_start_len": len(teacher_ids) - 1,
        },
    )
    teacher_top = _per_position_maps(teacher, "input_top_logprobs", response_length)

    # The student's log-probs are needed at the teacher's ids, which only a scoring call
    # against the rollout engine can provide. lora_path selects the current policy.
    support = sorted({token_id for position in teacher_top for token_id in position})
    student = await post(
        f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate",
        {
            "input_ids": list(sample.tokens),
            "sampling_params": {"temperature": 0, "max_new_tokens": 0, "skip_special_tokens": False},
            "return_logprob": True,
            "token_ids_logprob": support,
            "logprob_start_len": len(sample.tokens) - response_length - 1,
            "lora_path": LORA_ADAPTER_NAME,
        },
    )
    student_top = _per_position_maps(student, "input_token_ids_logprobs", response_length)

    # Forward KL over the teacher's support, clipped per entry before the sum so that a few
    # stylistic tokens cannot dominate the update.
    divergence = []
    for teacher_pos, student_pos in zip(teacher_top, student_top, strict=True):
        total = 0.0
        for token_id, teacher_logp in teacher_pos.items():
            student_logp = student_pos.get(token_id)
            if student_logp is None:
                continue
            total += min(math.exp(teacher_logp) * (teacher_logp - student_logp), TAU)
        divergence.append(total)
    sample.opd_reverse_kl = torch.tensor(divergence, dtype=torch.float32)


async def reward_func(args: Any, sample: Sample, **kwargs: Any) -> float:
    if sample.metadata.get("opsd_eval"):
        return _is_correct(sample.response or "", str(sample.label))
    await _score_teacher(args, sample)
    return 0.0
