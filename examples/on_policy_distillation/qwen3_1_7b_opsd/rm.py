"""Teacher scoring for privileged-context OPSD, plus accuracy for held-out rows.

The student generates from the problem alone; the teacher scores that same response on a
prompt that also contains the reference solution. Both prompts are rendered by
prepare_data.py, so the teacher prompt arrives on the sample as metadata.

The teacher's log-probs at the student's own sampled tokens are handed to miles as
sample.teacher_log_probs, which --use-opd turns into the per-token reverse KL
(log p_S - log p_T) it subtracts from the advantage. Held-out rows are scored for
accuracy here because --custom-rm-path is consulted unconditionally, so a per-sample
rm_type would never be reached.
"""

from typing import Any

import torch
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


async def _score_teacher(args: Any, sample: Sample) -> None:
    tokenizer = load_tokenizer(args.hf_checkpoint, chat_template_path=args.chat_template_path)
    prompt_ids = tokenizer.encode(sample.metadata["teacher_prompt"], add_special_tokens=False)
    response_ids = sample.tokens[len(sample.tokens) - sample.response_length :]

    # No lora_path, so the teacher is the base weights, which LoRA keeps frozen.
    response = await post(
        args.rm_url,
        {
            "input_ids": list(prompt_ids) + list(response_ids),
            "sampling_params": {"temperature": 0, "max_new_tokens": 0, "skip_special_tokens": False},
            "return_logprob": True,
            "logprob_start_len": len(prompt_ids) - 1,
        },
    )

    scored = response["meta_info"]["input_token_logprobs"][-sample.response_length :]
    # The teacher must have scored the student's own tokens; a mismatch means the
    # privileged prompt shifted the response and every log-prob would be off by position.
    assert [int(entry[1]) for entry in scored] == list(response_ids), "teacher/student token mismatch"
    sample.teacher_log_probs = torch.tensor([entry[0] for entry in scored], dtype=torch.float32)


async def reward_func(args: Any, sample: Sample, **kwargs: Any) -> float:
    if sample.metadata.get("opsd_eval"):
        return _is_correct(sample.response or "", str(sample.label))
    await _score_teacher(args, sample)
    return 0.0
