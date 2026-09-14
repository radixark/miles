"""Score privileged teacher prompts during training and boxed answers during evaluation."""

from typing import Any

from math_verify import parse, verify

from miles.rollout.on_policy_distillation import extract_teacher_support
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
    gold = parse(f"${label}$", fallback_mode="no_fallback", parsing_timeout=None)
    guess = parse(f"${predicted}$", fallback_mode="no_fallback", parsing_timeout=None)
    return 1.0 if verify(gold, guess, timeout_seconds=None) else 0.0


async def _score_teacher(args: Any, sample: Sample) -> dict[str, list]:
    tokenizer = load_tokenizer(args.hf_checkpoint, chat_template_path=args.chat_template_path)
    prompt_ids = tokenizer.encode(sample.metadata["teacher_prompt"], add_special_tokens=False)
    response_ids = sample.tokens[len(sample.tokens) - sample.response_length :]

    top_k = args.opd_log_prob_top_k
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

    scored = response["meta_info"]["input_token_logprobs"][-sample.response_length :] if sample.response_length else []
    assert [int(entry[1]) for entry in scored] == list(response_ids), "teacher/student token mismatch"
    return extract_teacher_support(response, sample.response_length, top_k)


async def reward_func(args: Any, sample: Sample, **kwargs: Any) -> float:
    if sample.metadata.get("opsd_eval"):
        return _is_correct(sample.response or "", str(sample.label))
    sample.train_metadata = {**(sample.train_metadata or {}), "opd": await _score_teacher(args, sample)}
    return 0.0
