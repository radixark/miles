"""Teacher scoring for privileged-context OPSD, plus accuracy for held-out rows.

The student generates from the problem alone; the teacher scores that same response on a
prompt that also contains the reference solution. Both prompts are rendered by
prepare_data.py, so the teacher prompt arrives on the sample as metadata.

This variant hands miles the teacher's top-k support rather than a divergence, so the
core forward-KL loss can minimise KL(teacher || student) against the student's training
logits. Per-entry clipping lives there too, under --opd-kl-clip. Only the teacher is
scored here; the student side never leaves the training step. Held-out rows are scored for accuracy here because --custom-rm-path
is consulted unconditionally, so a per-sample rm_type would never be reached.
"""

from typing import Any

from math_verify import parse, verify

from miles.utils.http_utils import post
from miles.utils.processing_utils import load_tokenizer
from miles.utils.types import Sample

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

    # Forward KL needs the teacher's distribution, not a divergence: the student's
    # log-probs at these ids come from the training logits, where they are differentiable.
    # Flattened to response_length * top_k so the loss can reshape it to [R, k]; short
    # positions pad with id 0 at -inf, which carries zero teacher mass.
    teacher_top = _per_position_maps(response["meta_info"]["input_top_logprobs"], sample.response_length)
    ids: list[int] = []
    logps: list[float] = []
    for position in teacher_top:
        entries = sorted(position.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        for token_id, logp in entries:
            ids.append(int(token_id))
            logps.append(float(logp))
        for _ in range(top_k - len(entries)):
            ids.append(0)
            logps.append(float("-inf"))
    sample.teacher_top_ids = ids
    sample.teacher_top_logprobs = logps


async def reward_func(args: Any, sample: Sample, **kwargs: Any) -> float:
    if sample.metadata.get("opsd_eval"):
        return _is_correct(sample.response or "", str(sample.label))
    await _score_teacher(args, sample)
    return 0.0
