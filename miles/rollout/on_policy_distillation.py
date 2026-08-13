import logging
import math
from argparse import Namespace
from collections.abc import Iterable
from copy import deepcopy
from typing import Any

import aiohttp
import torch

from miles.utils.processing_utils import load_tokenizer
from miles.utils.types import Sample

logger = logging.getLogger(__name__)

TopLogprobs = list[list[Any]]
LogprobMaps = list[dict[int, float]]

TOP_K_STRATEGIES = {"only-student", "only-teacher", "intersection", "union", "xor"}
REWARD_WEIGHT_MODES = {"student_p", "teacher_p", "none"}

STUDENT_TOP_STRATEGIES = TOP_K_STRATEGIES - {"only-teacher"}
TEACHER_TOP_STRATEGIES = TOP_K_STRATEGIES - {"only-student"}
TEACHER_ON_STUDENT_STRATEGIES = {"only-student", "union", "xor"}
STUDENT_ON_TEACHER_STRATEGIES = {"only-teacher", "union", "xor"}

# Reserved teacher name in --opd-teacher-urls used as the fallback route.
DEFAULT_TEACHER_NAME = "default"


def parse_teacher_urls(values: Iterable[str] | None) -> dict[str, str]:
    """Parse ``NAME=URL`` entries from ``--opd-teacher-urls`` into a routing map.

    Splits on the first ``=`` only, so URLs containing ``=`` (e.g. query
    strings) survive intact. Raises on malformed entries and duplicate names
    so misconfiguration fails at startup, not mid-rollout.
    """
    url_map: dict[str, str] = {}
    for value in values or []:
        name, sep, url = value.partition("=")
        name, url = name.strip(), url.strip()
        if not sep or not name or not url:
            raise ValueError(f"Invalid --opd-teacher-urls entry {value!r}; expected NAME=URL.")
        if name in url_map:
            raise ValueError(f"Duplicate teacher name {name!r} in --opd-teacher-urls.")
        url_map[name] = url
    return url_map


def _teacher_url_for_sample(args: Namespace, sample: Sample) -> str:
    """Resolve the teacher scoring endpoint for one sample.

    Without ``--opd-teacher-urls`` every sample goes to ``--rm-url`` (the
    original single-teacher path, unchanged). With it, the sample is routed by
    the teacher name in ``sample.metadata[--opd-teacher-key]``; samples whose
    name is missing or unknown fall back to the reserved ``default`` entry,
    and raise if no default is configured — silently distilling from the
    wrong teacher is worse than failing the rollout.
    """
    url_map = parse_teacher_urls(getattr(args, "opd_teacher_urls", None))
    if not url_map:
        return args.rm_url

    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    key = getattr(args, "opd_teacher_key", "opd_teacher")
    name = metadata.get(key)
    if name is not None:
        url = url_map.get(str(name))
        if url is not None:
            return url
        if DEFAULT_TEACHER_NAME in url_map:
            return url_map[DEFAULT_TEACHER_NAME]
        raise ValueError(
            f"Sample metadata[{key!r}]={name!r} matches no --opd-teacher-urls name "
            f"(known: {sorted(url_map)}) and no 'default' entry is configured."
        )
    if DEFAULT_TEACHER_NAME in url_map:
        return url_map[DEFAULT_TEACHER_NAME]
    raise ValueError(f"Sample metadata is missing teacher key {key!r} and --opd-teacher-urls has no 'default' entry.")


PROBE_MARKER = "MilesPrivilegedProbe"
_GENERATION_TAIL_CACHE: dict[tuple[str, str], str] = {}


_OVERFLOWED = [0]
_PRIVILEGED_SEEN = [0, 0]  # [samples actually scored with context, samples checked]


def _privileged_context(args: Namespace, sample: Sample) -> str | None:
    """Teacher-only context for one sample, or ``None`` when it has none.

    ``None`` covers both "the feature is off" and "this sample carries no context", so a
    single dataset can mix privileged and plain samples, the same fallback philosophy as
    ``_teacher_url_for_sample``. A present-but-unusable value raises instead, since
    silently distilling without the privileged signal defeats the point of enabling it.
    """
    key = getattr(args, "opd_privileged_context_key", None)
    if not key:
        return None
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    context = metadata.get(key)

    if context is None:
        return None
    if not isinstance(context, str) or not context.strip():
        raise ValueError(f"Sample metadata[{key!r}] must be a non-empty string, got {context!r}.")
    return context


def _render_chat(args: Namespace, messages: list[dict[str, Any]], tokenizer: Any, tools: Any = None) -> str:
    """Render through the same helper that produced ``sample.prompt``.

    Not ``tokenizer.apply_chat_template``: for DeepSeek and Inkling checkpoints
    ``chat_template_utils`` dispatches to a different renderer entirely, so a raw-tokenizer
    probe would derive a tail that the prompt never ends with.
    """
    # Local import: chat_template_utils pulls in sglang, and miles.utils.arguments
    # imports this module during validation.
    from miles.utils import chat_template_utils

    return chat_template_utils.apply_chat_template(
        messages,
        tokenizer=tokenizer,
        tools=tools,
        tokenize=False,
        add_generation_prompt=True,
        **(getattr(args, "apply_chat_template_kwargs", None) or {}),
    )


def _record_privileged(args: Namespace, *, applied: bool) -> None:
    """Log how many samples were actually scored WITH privileged context.

    Counted after the fact rather than at metadata lookup, so the context-overflow
    fallback is visible: a run where every sample carries the key but every sample
    overflows is not distilling with privileged context, and must not look healthy.
    A typo'd key, or a missing --metadata-key, otherwise degrades silently to ordinary
    self-distillation -- reverse-KL ~0, no learning, every training metric fine.
    """
    _PRIVILEGED_SEEN[1] += 1
    if applied:
        _PRIVILEGED_SEEN[0] += 1
    seen = _PRIVILEGED_SEEN[1]
    if seen in (16, 64) or seen % 256 == 0:
        found = _PRIVILEGED_SEEN[0]
        logger.log(
            logging.WARNING if found == 0 else logging.INFO,
            "privileged-context OPD: %d/%d samples scored with metadata[%r]%s",
            found,
            seen,
            getattr(args, "opd_privileged_context_key", None),
            (
                " -- check --opd-privileged-context-key, --metadata-key, and --rollout-max-context-len"
                if found == 0
                else ""
            ),
        )


def _generation_tail(args: Namespace, tokenizer: Any) -> str:
    """The text every rendered prompt ends with, after the last message's content.

    Derived from the tokenizer by rendering a probe message and splitting on it, rather
    than hard-coded per model, e.g. ``<|im_end|>\n<|im_start|>assistant\n`` for ChatML,
    ``<end_of_turn>\n<start_of_turn>model\n`` for Gemma. Splicing before this lands text
    at the end of the last user message, which is where privileged context belongs.
    """
    kwargs = getattr(args, "apply_chat_template_kwargs", None) or {}
    cache_key = (
        getattr(args, "hf_checkpoint", ""),
        getattr(args, "chat_template_path", None) or "",
        repr(sorted(kwargs.items())),
    )
    if cache_key in _GENERATION_TAIL_CACHE:
        return _GENERATION_TAIL_CACHE[cache_key]

    probe = _render_chat(args, [{"role": "user", "content": PROBE_MARKER}], tokenizer)
    # The probe renders only the marker, so a user's own prompt can never collide here.
    # What can go wrong is the template transforming the content (Gemma-2 trims it) or
    # echoing it more than once, either of which makes the split point meaningless.
    occurrences = probe.count(PROBE_MARKER)
    if occurrences == 0:
        raise ValueError(
            "Cannot derive the chat template's generation tail, so privileged context has "
            "nowhere safe to go. This template rewrites message content. Supply the "
            "teacher's prompt yourself, or disable --opd-privileged-context-key."
        )
    if occurrences > 1:
        raise ValueError(
            f"Chat template rendered the probe marker {occurrences} times, so the end of the "
            "last user message is ambiguous. Disable --opd-privileged-context-key."
        )

    tail = probe.partition(PROBE_MARKER)[2]
    # An empty tail would make prompt[:-0] == "", silently replacing the whole prompt.
    if not tail:
        raise ValueError(
            "The chat template appends nothing after the last user message, so privileged "
            "context cannot be placed before a generation tail."
        )
    _GENERATION_TAIL_CACHE[cache_key] = tail
    return tail


def _teacher_prompt_text(args: Namespace, sample: Sample, context: str, tokenizer: Any) -> str:
    """Insert ``context`` at the end of the last user turn, whichever form the prompt took.

    ``miles/utils/data.py`` produces three shapes, not two: a message list, a
    template-rendered string, or -- with --apply-chat-template off and a plain-text prompt
    column -- the raw question with no template at all. All three are handled here.
    """
    prompt = sample.prompt

    if isinstance(prompt, list):
        if not prompt or prompt[-1].get("role") != "user" or not isinstance(prompt[-1].get("content"), str):
            raise ValueError("Privileged-context OPD needs a message list ending in a text user message.")
        messages = deepcopy(prompt)
        messages[-1]["content"] += context
        metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
        rendered = _render_chat(args, messages, tokenizer, metadata.get("tools"))
        # Meaningful here and only here: the template can trim or drop content
        # (Gemma-2 applies `content | trim`), so verify it survived rendering.
        _assert_context_placed(rendered, context)
        return rendered

    if not isinstance(prompt, str):
        raise ValueError(f"Privileged-context OPD needs a str or message-list prompt, got {type(prompt).__name__}.")

    # Verify the shape rather than infer it. A prompt that ends in this template's
    # generation tail IS rendered, whatever --apply-chat-template says: the flag is off
    # for pre-rendered prompt columns too, and appending there would put the context
    # after the generation prompt, making it the assistant's opening tokens.
    try:
        tail = _generation_tail(args, tokenizer)
    except ValueError:
        # Its three messages (marker rewritten / echoed / empty tail) are the useful
        # diagnostic; only discard them where the prompt may legitimately be untemplated.
        if getattr(args, "apply_chat_template", False):
            raise
        tail = None

    if tail and prompt.endswith(tail):
        return prompt[: -len(tail)] + context + tail
    if getattr(args, "apply_chat_template", False):
        raise ValueError(
            f"Rendered prompt does not end with this template's generation tail {tail!r}, so "
            "privileged context cannot be placed inside the last turn. This usually means the "
            "prompt was rendered for a different conversation shape than the probe."
        )
    return prompt + context


def _teacher_input_ids(args: Namespace, sample: Sample, context: str, tokenizer: Any) -> list[int] | None:
    """Re-host the student's response on a prompt that also carries ``context``.

    The response tokens are reused verbatim, so the teacher predicts exactly what the
    student produced while conditioned on information the student never saw. Because the
    response stays the tail of the sequence, ``_trim_input_field`` lifts it out unchanged
    despite the longer prompt.
    """
    text = _teacher_prompt_text(args, sample, context, tokenizer)
    prompt_ids = list(tokenizer.encode(text, add_special_tokens=False))
    # Not tokens[-response_length:], which returns the whole list when response_length is 0.
    response_ids = sample.tokens[len(sample.tokens) - sample.response_length :]

    input_ids = prompt_ids + response_ids
    limit = getattr(args, "rollout_max_context_len", None)
    if limit and len(input_ids) > limit:
        # A sample that generated to the context cap cannot also fit the privileged
        # prompt. That is a per-sample condition, so it degrades to ordinary scoring
        # rather than raising -- an exception here propagates out of async_rm and kills
        # the whole run, and this can hit a large fraction of samples.
        _OVERFLOWED[0] += 1
        if _OVERFLOWED[0] & (_OVERFLOWED[0] - 1) == 0:  # 1st, 2nd, 4th, 8th ... occurrence
            logger.warning(
                "Privileged teacher sequence exceeds --rollout-max-context-len for %d sample(s) so far "
                "(latest: %d prompt + %d response > %d); scoring those without privileged context.",
                _OVERFLOWED[0],
                len(prompt_ids),
                len(response_ids),
                limit,
            )
        return None
    return input_ids


def _opd_tokenizer(args: Namespace) -> Any:
    # load_tokenizer caches on (name, chat_template_path, kwargs), so this reuses the
    # instance the rollout already built instead of re-reading it per sample.
    return load_tokenizer(args.hf_checkpoint, chat_template_path=args.chat_template_path, trust_remote_code=True)


def _teacher_scoring_tokens(args: Namespace, sample: Sample) -> list[int]:
    """Token ids for the teacher to score: ``sample.tokens`` unless privileged context applies."""
    context = _privileged_context(args, sample)
    if context is None:
        if getattr(args, "opd_privileged_context_key", None):
            # Counted even though nothing was applied: a typo'd key means EVERY sample
            # lands here, and that is the case the warning exists for.
            _record_privileged(args, applied=False)
        return sample.tokens
    input_ids = _teacher_input_ids(args, sample, context, _opd_tokenizer(args))
    _record_privileged(args, applied=input_ids is not None)
    return sample.tokens if input_ids is None else input_ids


def _get_opd_top_k(args: Namespace) -> int:
    return max(0, int(getattr(args, "opd_log_prob_top_k", 0) or 0))


def _get_top_k_strategy(args: Namespace) -> str:
    strategy = getattr(args, "opd_top_k_strategy", "only-student")
    if strategy not in TOP_K_STRATEGIES:
        raise ValueError(f"Unknown OPD top-k strategy: {strategy}")
    return strategy


def _get_reward_weight_mode(args: Namespace) -> str:
    mode = getattr(args, "opd_reward_weight_mode", "student_p")
    if mode not in REWARD_WEIGHT_MODES:
        raise ValueError(f"Unknown OPD reward weight mode: {mode}")
    return mode


def _score_payload(
    input_ids: list[int],
    top_k: int = 0,
    token_ids: list[int] | None = None,
    token_ids_positions: list[list[int]] | None = None,
) -> dict[str, Any]:
    payload = {
        "input_ids": input_ids,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 0,
            "skip_special_tokens": False,
        },
        "return_logprob": True,
        "logprob_start_len": 0,
    }
    if top_k > 0:
        payload["top_logprobs_num"] = top_k
    if token_ids_positions is not None:
        # Per-position scoring (patched sglang): one id-list per input position, so the
        # teacher returns each position's own ids (sparse) instead of the global union
        # broadcast to every position (dense O(R^2)). Aligned to logprob_start_len=0.
        payload["token_ids_logprob_positions"] = token_ids_positions
    elif token_ids:
        payload["token_ids_logprob"] = token_ids
    return payload


def _per_position_ids(top_logprobs: TopLogprobs, prompt_len: int) -> list[list[int]]:
    """Build one token-id list per scored input position for ``token_ids_logprob_positions``.

    ``top_logprobs`` is per response position (length == response_length). Prompt
    positions are padded with empty id-lists so the layout aligns with
    ``logprob_start_len=0`` and the existing ``_trim_input_field`` extraction
    (``values[1:][-response_length:]``) — i.e. response position r lands at index
    ``prompt_len + r``.
    """
    per_pos: list[list[int]] = [[] for _ in range(prompt_len)]
    for entries in top_logprobs:
        per_pos.append([_top_entry_token_id(e) for e in (entries or []) if e is not None])
    return per_pos


def _student_score_url(args: Namespace) -> str:
    return f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"


async def _post_json(url: str, payload: dict[str, Any], timeout_secs: int | float | None = None) -> dict[str, Any]:
    timeout = aiohttp.ClientTimeout(total=timeout_secs)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(url, json=payload) as resp:
            resp.raise_for_status()
            return await resp.json()


def _top_entry_token_id(entry: list[Any]) -> int:
    return int(entry[1])


def _top_entry_logprob(entry: list[Any]) -> float:
    return float(entry[0])


def _top_entries_to_map(entries: Iterable[list[Any]] | None) -> dict[int, float]:
    if not entries:
        return {}
    return {_top_entry_token_id(entry): _top_entry_logprob(entry) for entry in entries if entry is not None}


def _trim_input_field(meta_info: dict[str, Any], field: str, response_length: int) -> list[Any]:
    values = meta_info.get(field)
    if values is None:
        raise ValueError(f"Teacher response is missing meta_info.{field}.")
    # SGLang's first input logprob/top-logprob position is a placeholder.
    return values[1:][-response_length:] if response_length > 0 else []


def _input_logprob_maps(response: dict[str, Any], field: str, response_length: int) -> LogprobMaps:
    return [
        _top_entries_to_map(entries) for entries in _trim_input_field(response["meta_info"], field, response_length)
    ]


def _teacher_sampled_log_probs(response: dict[str, Any], response_length: int) -> torch.Tensor:
    input_token_logprobs = _trim_input_field(response["meta_info"], "input_token_logprobs", response_length)
    return torch.tensor([item[0] for item in input_token_logprobs], dtype=torch.float32)


def _student_top_logprobs(sample: Sample, response_length: int) -> TopLogprobs:
    top_logprobs = sample.metadata.get("opd_student_top_logprobs")
    if top_logprobs is None:
        raise ValueError(
            "Top-k OPD requires student output_top_logprobs. "
            "Ensure --opd-log-prob-top-k is set before rollout generation starts."
        )
    top_logprobs = top_logprobs[-response_length:] if response_length > 0 else []
    if len(top_logprobs) != response_length:
        raise ValueError(
            f"Student top-k logprob length mismatch: got {len(top_logprobs)}, expected {response_length}."
        )
    return top_logprobs


def _unique_ids(top_logprobs: Iterable[Iterable[list[Any]]]) -> list[int]:
    ids = set()
    for entries in top_logprobs:
        for entry in entries or []:
            if entry is not None:
                ids.add(_top_entry_token_id(entry))
    return sorted(ids)


def _ordered_unique(ids: Iterable[int]) -> list[int]:
    seen = set()
    ordered = []
    for token_id in ids:
        if token_id in seen:
            continue
        seen.add(token_id)
        ordered.append(token_id)
    return ordered


def _selected_token_ids(strategy: str, student_ids: list[int], teacher_ids: list[int]) -> list[int]:
    student_set = set(student_ids)
    teacher_set = set(teacher_ids)
    if strategy == "only-student":
        return student_ids
    if strategy == "only-teacher":
        return teacher_ids
    if strategy == "intersection":
        return [token_id for token_id in student_ids if token_id in teacher_set]
    if strategy == "union":
        return _ordered_unique([*student_ids, *teacher_ids])
    if strategy == "xor":
        return [
            token_id
            for token_id in [*student_ids, *teacher_ids]
            if (token_id in student_set) != (token_id in teacher_set)
        ]
    raise ValueError(f"Unknown OPD top-k strategy: {strategy}")


def _lookup_logprob(
    token_id: int,
    primary: dict[int, float],
    fallback: dict[int, float] | None,
    *,
    source: str,
) -> float:
    if token_id in primary:
        return primary[token_id]
    if fallback is not None and token_id in fallback:
        return fallback[token_id]
    raise ValueError(f"Missing {source} logprob for token id {token_id}.")


def _reward_weights(
    student_logps: list[float],
    teacher_logps: list[float],
    mode: str,
    *,
    normalize: bool,
) -> list[float]:
    if not student_logps:
        return []
    if mode == "student_p":
        logps = student_logps
    elif mode == "teacher_p":
        logps = teacher_logps
    elif mode == "none":
        logps = [0.0] * len(student_logps)
    else:
        raise ValueError(f"Unknown OPD reward weight mode: {mode}")

    if not normalize:
        return [math.exp(logp) for logp in logps]

    max_logp = max(logps)
    exp_vals = [math.exp(logp - max_logp) for logp in logps]
    denom = sum(exp_vals)
    if denom == 0.0:
        return [0.0] * len(logps)
    return [v / denom for v in exp_vals]


def _compute_topk_reverse_kl(
    args: Namespace,
    sample: Sample,
    reward_payload: dict[str, Any],
) -> torch.Tensor:
    response_length = sample.response_length
    if response_length == 0:
        return torch.zeros((0,), dtype=torch.float32)

    strategy = _get_top_k_strategy(args)
    weight_mode = _get_reward_weight_mode(args)

    student_top_maps = (
        [_top_entries_to_map(entries) for entries in _student_top_logprobs(sample, response_length)]
        if strategy in STUDENT_TOP_STRATEGIES
        else [{} for _ in range(response_length)]
    )

    teacher_response = reward_payload["teacher"]
    teacher_top_maps = (
        _input_logprob_maps(teacher_response, "input_top_logprobs", response_length)
        if strategy in TEACHER_TOP_STRATEGIES
        else [{} for _ in range(response_length)]
    )
    teacher_on_student_maps = (
        _input_logprob_maps(teacher_response, "input_token_ids_logprobs", response_length)
        if strategy in TEACHER_ON_STUDENT_STRATEGIES
        else [{} for _ in range(response_length)]
    )
    student_on_teacher_maps = (
        _input_logprob_maps(reward_payload["student_on_teacher"], "input_token_ids_logprobs", response_length)
        if strategy in STUDENT_ON_TEACHER_STRATEGIES
        else [{} for _ in range(response_length)]
    )

    reverse_kls = []
    normalize_weights = strategy != "xor"
    for i in range(response_length):
        student_ids = list(student_top_maps[i].keys())
        teacher_ids = list(teacher_top_maps[i].keys())
        selected_ids = _selected_token_ids(strategy, student_ids, teacher_ids)

        student_logps = []
        teacher_logps = []
        for token_id in selected_ids:
            student_logps.append(
                _lookup_logprob(
                    token_id,
                    student_top_maps[i],
                    student_on_teacher_maps[i],
                    source="student",
                )
            )
            teacher_logps.append(
                _lookup_logprob(
                    token_id,
                    teacher_top_maps[i],
                    teacher_on_student_maps[i],
                    source="teacher",
                )
            )

        weights = _reward_weights(student_logps, teacher_logps, weight_mode, normalize=normalize_weights)
        reverse_kl = sum(
            w * (s_logp - t_logp) for w, s_logp, t_logp in zip(weights, student_logps, teacher_logps, strict=True)
        )
        reverse_kls.append(reverse_kl)

    return torch.tensor(reverse_kls, dtype=torch.float32)


def _assert_context_placed(text: str, context: str) -> None:
    """Confirm the context survived rendering into the teacher prompt.

    Only useful on the render-from-messages path: a template that rewrites content
    (Gemma-2 applies ``content | trim``) can drop or alter it. The splice path needs no
    such check, since it concatenates the pieces itself.
    """
    if context not in text:
        raise ValueError(
            "Privileged context did not survive rendering into the teacher prompt; "
            "this chat template rewrites message content."
        )


async def _score_sampled_tokens(
    args: Namespace,
    sample: Sample,
    teacher_url: str,
    request_timeout: int | float | None,
) -> dict[str, Any]:
    """Sampled-token teacher scoring, with privileged context when the sample carries it.

    Without privileged context the teacher scores ``sample.tokens``, unchanged behavior.
    """
    teacher_tokens = _teacher_scoring_tokens(args, sample)
    return await _post_json(teacher_url, _score_payload(teacher_tokens), timeout_secs=request_timeout)


async def reward_func(args: Namespace, sample: Sample, **kwargs: Any) -> dict[str, Any]:
    top_k = _get_opd_top_k(args)
    # Optional per-request timeout so a hung teacher/student scoring call cannot stall
    # the whole rollout (no-op when unset).
    request_timeout = getattr(args, "sglang_router_request_timeout_secs", None)
    # Multi-teacher routing: pick this sample's teacher endpoint (falls back to
    # --rm-url when --opd-teacher-urls is unset).
    teacher_url = _teacher_url_for_sample(args, sample)
    if top_k == 0:
        return await _score_sampled_tokens(args, sample, teacher_url, request_timeout)

    strategy = _get_top_k_strategy(args)
    # Per-position scoring requires a patched teacher/student server that understands
    # token_ids_logprob_positions; default off so an unpatched server keeps working.
    per_position = getattr(args, "opd_topk_per_position", False)
    prompt_len = len(sample.tokens) - sample.response_length
    # A privileged prompt re-hosts the response on a different teacher prompt. Response
    # positions still line up (the maps below are all trimmed to the response tail), so
    # only the teacher's own ids and prompt padding shift; the student keeps scoring
    # exactly what it saw.
    teacher_tokens = _teacher_scoring_tokens(args, sample)
    teacher_prompt_len = len(teacher_tokens) - sample.response_length

    teacher_top_k = top_k if strategy in TEACHER_TOP_STRATEGIES else 0
    if strategy in TEACHER_ON_STUDENT_STRATEGIES:
        student_top = _student_top_logprobs(sample, sample.response_length)
        teacher_token_ids = _unique_ids(student_top)
    else:
        student_top = None
        teacher_token_ids = None

    if student_top is not None and per_position:
        teacher_payload = _score_payload(
            teacher_tokens, top_k=teacher_top_k, token_ids_positions=_per_position_ids(student_top, teacher_prompt_len)
        )
    elif teacher_token_ids is not None:
        teacher_payload = _score_payload(teacher_tokens, top_k=teacher_top_k, token_ids=teacher_token_ids)
    else:
        teacher_payload = _score_payload(teacher_tokens, top_k=teacher_top_k)
    teacher_response = await _post_json(teacher_url, teacher_payload, timeout_secs=request_timeout)

    reward_payload = {"teacher": teacher_response}
    if strategy in STUDENT_ON_TEACHER_STRATEGIES:
        teacher_top = _trim_input_field(teacher_response["meta_info"], "input_top_logprobs", sample.response_length)
        if per_position:
            student_payload = _score_payload(
                sample.tokens, token_ids_positions=_per_position_ids(teacher_top, prompt_len)
            )
        else:
            student_payload = _score_payload(sample.tokens, token_ids=_unique_ids(teacher_top))
        reward_payload["student_on_teacher"] = await _post_json(
            _student_score_url(args), student_payload, timeout_secs=request_timeout
        )

    return reward_payload


def post_process_rewards(args: Namespace, samples: list[Sample], **kwargs: Any) -> tuple[list[float], list[float]]:
    """Extract OPD signals from teacher responses.

    ``--opd-log-prob-top-k=0`` preserves the original sampled-token OPD path:
    store teacher log-probs and let training compute ``student_logp - teacher_logp``.

    ``--opd-log-prob-top-k>0`` follows the practical recipe from
    "Rethinking On-Policy Distillation" by forming a top-k token set per
    response position and storing a precomputed weighted reverse-KL estimate.
    """
    raw_rewards = [sample.get_reward_value(args) for sample in samples]
    response_lengths = [sample.response_length for sample in samples]

    if _get_opd_top_k(args) > 0:
        for sample, reward in zip(samples, raw_rewards, strict=True):
            sample.opd_reverse_kl = _compute_topk_reverse_kl(args, sample, reward)
        scalar_rewards = [0.0] * len(samples)
        return scalar_rewards, scalar_rewards

    teacher_log_probs = [
        _teacher_sampled_log_probs(reward, response_length)
        for reward, response_length in zip(raw_rewards, response_lengths, strict=True)
    ]

    for sample, t_log_probs in zip(samples, teacher_log_probs, strict=True):
        sample.teacher_log_probs = t_log_probs

    # Return scalar rewards for GRPO/PPO advantage estimator.
    # For pure on-policy distillation, we use 0.0 as the task reward.
    # The learning signal comes entirely from the OPD KL penalty.
    # If you have task rewards, you can add them here.
    scalar_rewards = [0.0] * len(samples)

    return scalar_rewards, scalar_rewards
