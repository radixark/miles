"""Cross-tokenizer on-policy distillation reward path (arXiv 2606.09456).

This mirrors :mod:`miles.rollout.on_policy_distillation` but supports a teacher
whose tokenizer differs from the student's (e.g. a GLM teacher distilled into a
Qwen student). It is enabled with ``--opd-teacher-tokenizer`` and is restricted
to ``--opd-type sglang`` with ``--opd-log-prob-top-k 0`` (validated in
``miles/utils/arguments.py``).

Pipeline (entirely rollout-side; the loss / Megatron path is unchanged):

* :func:`reward_func` (per sample, async) renders the prompt with the *teacher's*
  own chat template applied to the same raw prompt the student saw, appends the
  student's response text tokenized by the teacher, and queries the teacher
  SGLang server for per-token logprobs of that teacher-tokenized response.
* :func:`post_process_rewards` (batch, sync) DPCA-aligns the student and teacher
  token sequences (see :mod:`miles.rollout.dpca`) and stores the per-student-token
  reverse-KL signal in ``sample.opd_reverse_kl``. The existing *precomputed*
  branch of ``apply_opd_kl_to_advantages`` then applies it during training. The
  signal reduces exactly to ``student_logp - teacher_logp`` when the tokenizers
  agree, so this is a strict generalization of the shared-tokenizer path.

Pure distillation: scalar task rewards are ``0.0``; the learning signal comes
entirely from the OPD KL penalty.
"""

import logging
from argparse import Namespace
from collections.abc import Callable, Sequence
from functools import lru_cache
from typing import Any

import torch

from miles.rollout.dpca import align_chunks, compute_cross_tokenizer_reverse_kl, unaligned_token_fraction
from miles.rollout.on_policy_distillation import _post_json, _score_payload, _teacher_sampled_log_probs
from miles.utils.types import Sample

logger = logging.getLogger(__name__)

# Key under which the dataset loader stashes the raw (pre-chat-template) prompt so
# the teacher can be re-templated with its own chat template. See miles/utils/data.py.
RAW_PROMPT_METADATA_KEY = "opd_raw_prompt"


@lru_cache(maxsize=8)
def _load_cached_tokenizer(name_or_path: str, chat_template_path: str | None):
    # Imported lazily so importing this module (e.g. in CPU unit tests of the
    # pure DPCA core) does not require transformers to be installed.
    from miles.utils.processing_utils import load_tokenizer

    return load_tokenizer(name_or_path, chat_template_path=chat_template_path, trust_remote_code=True)


def _teacher_tokenizer(args: Namespace):
    teacher = getattr(args, "opd_teacher_tokenizer", None)
    if not teacher:
        raise ValueError(
            "cross-tokenizer OPD requires --opd-teacher-tokenizer (HF id or path of the teacher tokenizer)."
        )
    return _load_cached_tokenizer(teacher, None)


def _student_tokenizer(args: Namespace):
    return _load_cached_tokenizer(args.hf_checkpoint, getattr(args, "chat_template_path", None))


def _teacher_prompt_ids(teacher_tok, raw_prompt: str | list[dict[str, str]], tools: Any | None) -> list[int]:
    """Tokenize the prompt with the teacher's chat template (mirrors data.py).

    ``raw_prompt`` is the original prompt the student saw before its chat template
    was applied: a conversation (list of message dicts) when ``--apply-chat-template``
    was used, otherwise a plain string.
    """
    if isinstance(raw_prompt, str):
        # No conversation structure available; feed the string verbatim.
        return teacher_tok.encode(raw_prompt, add_special_tokens=False)
    return teacher_tok.apply_chat_template(
        raw_prompt,
        tools=tools,
        add_generation_prompt=True,
        tokenize=True,
    )


async def reward_func(args: Namespace, sample: Sample, **kwargs: Any) -> dict[str, Any]:
    """Query the teacher for logprobs of the student's response under teacher tokenization."""
    teacher_tok = _teacher_tokenizer(args)

    raw_prompt = sample.metadata.get(RAW_PROMPT_METADATA_KEY, sample.prompt)
    teacher_resp_ids = teacher_tok.encode(sample.response, add_special_tokens=False)
    if len(teacher_resp_ids) == 0:
        return {"teacher_resp_ids": [], "teacher_resp_logprobs": []}

    teacher_prompt_ids = _teacher_prompt_ids(teacher_tok, raw_prompt, sample.metadata.get("tools"))
    input_ids = list(teacher_prompt_ids) + list(teacher_resp_ids)

    response = await _post_json(args.rm_url, _score_payload(input_ids))
    teacher_resp_logprobs = _teacher_sampled_log_probs(response, len(teacher_resp_ids))
    return {
        "teacher_resp_ids": list(teacher_resp_ids),
        "teacher_resp_logprobs": teacher_resp_logprobs.tolist(),
    }


def _trailing_special_count(ids: Sequence[int], special_ids: set[int]) -> int:
    n = 0
    while n < len(ids) and ids[-(n + 1)] in special_ids:
        n += 1
    return n


def _special_ids(tokenizer) -> set[int]:
    return {int(i) for i in (getattr(tokenizer, "all_special_ids", None) or [])}


def _decode_fn(tokenizer) -> Callable[[list[int]], str]:
    return lambda ids: tokenizer.decode(ids)


def post_process_rewards(args: Namespace, samples: list[Sample], **kwargs: Any) -> tuple[list[float], list[float]]:
    """Align student/teacher tokenizations via DPCA and fill ``sample.opd_reverse_kl``."""
    student_tok = _student_tokenizer(args)
    teacher_tok = _teacher_tokenizer(args)
    student_decode = _decode_fn(student_tok)
    teacher_decode = _decode_fn(teacher_tok)
    student_special = _special_ids(student_tok)
    teacher_special = _special_ids(teacher_tok)

    rewards = [sample.get_reward_value(args) for sample in samples]

    unaligned_fractions: list[float] = []
    for sample, reward in zip(samples, rewards, strict=True):
        response_length = sample.response_length
        if response_length == 0:
            sample.opd_reverse_kl = []
            continue

        if sample.rollout_log_probs is None:
            raise ValueError(
                "cross-tokenizer OPD requires student rollout_log_probs, but they are missing. "
                "Ensure the rollout engine returns student logprobs (sampling with return_logprob)."
            )

        student_ids = list(sample.tokens[-response_length:])
        student_logprobs = list(sample.rollout_log_probs)
        teacher_resp_ids = list(reward.get("teacher_resp_ids") or [])
        teacher_logprobs = list(reward.get("teacher_resp_logprobs") or [])

        reverse_kl = torch.zeros(response_length, dtype=torch.float32)

        # Drop trailing special tokens (e.g. EOS) on both sides: their cross-model
        # logprobs are meaningless and they would otherwise mis-align the tail.
        n_strip_s = _trailing_special_count(student_ids, student_special)
        n_strip_t = _trailing_special_count(teacher_resp_ids, teacher_special)
        keep_s = len(student_ids) - n_strip_s
        keep_t = len(teacher_resp_ids) - n_strip_t

        if keep_s > 0 and keep_t > 0 and len(teacher_logprobs) >= keep_t:
            chunks = align_chunks(
                student_ids[:keep_s],
                teacher_resp_ids[:keep_t],
                student_decode,
                teacher_decode,
            )
            partial = compute_cross_tokenizer_reverse_kl(
                chunks,
                student_logprobs[:keep_s],
                teacher_logprobs[:keep_t],
            )
            reverse_kl[:keep_s] = partial
            unaligned_fractions.append(unaligned_token_fraction(chunks))
        else:
            # No usable teacher signal (empty response or logprob count mismatch):
            # leave a zero penalty for this sample rather than raising mid-rollout.
            unaligned_fractions.append(1.0 if keep_s > 0 else 0.0)

        sample.opd_reverse_kl = reverse_kl.tolist()

    if unaligned_fractions:
        mean_unaligned = sum(unaligned_fractions) / len(unaligned_fractions)
        logger.info(
            "cross-tokenizer OPD: mean unaligned-token fraction %.4f over %d samples",
            mean_unaligned,
            len(unaligned_fractions),
        )

    # Pure distillation: zero task reward; learning signal is the stored OPD penalty.
    scalar_rewards = [0.0] * len(samples)
    return scalar_rewards, scalar_rewards
