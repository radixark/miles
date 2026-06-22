import asyncio
import math
import random
from argparse import Namespace
from collections.abc import Iterable
from typing import Any

import aiohttp
import torch

from miles.utils.types import Sample

TopLogprobs = list[list[Any]]
LogprobMaps = list[dict[int, float]]
# One teacher endpoint inside a (possibly singleton) ensemble group: (url, mixture weight).
TeacherTarget = tuple[str, float]

TOP_K_STRATEGIES = {"only-student", "only-teacher", "intersection", "union", "xor"}
REWARD_WEIGHT_MODES = {"student_p", "teacher_p", "none"}

STUDENT_TOP_STRATEGIES = TOP_K_STRATEGIES - {"only-teacher"}
TEACHER_TOP_STRATEGIES = TOP_K_STRATEGIES - {"only-student"}
TEACHER_ON_STUDENT_STRATEGIES = {"only-student", "union", "xor"}
STUDENT_ON_TEACHER_STRATEGIES = {"only-teacher", "union", "xor"}

# Reserved teacher name in --opd-teacher-urls used as the fallback route.
DEFAULT_TEACHER_NAME = "default"
# Floor for the teacher's tail probability mass in --opd-topk-tail-bucket mode. Scoring
# logprobs arrive as JSON doubles, so 1 - sum(p) only goes non-positive through genuine
# float64 rounding; the floor (log ~= -27.6) is a last-resort guard, not a working range.
TAIL_PROB_FLOOR = 1e-12
# Bounded retry for teacher/student scoring requests: one retry on transient failures
# (timeout, connection error, HTTP 5xx) with jitter. 4xx responses never retry.
SCORING_MAX_RETRIES = 1


def _parse_teacher_target(part: str, entry: str) -> TeacherTarget:
    """Parse one ``URL[@WEIGHT]`` group member.

    Splits on the last ``@`` so userinfo URLs (``user:pass@host``) survive; a
    suffix that does not parse as a float is treated as part of the URL.
    """
    url, sep, weight_str = part.rpartition("@")
    if sep:
        try:
            weight = float(weight_str)
        except ValueError:
            return part, 1.0
        if not math.isfinite(weight) or weight <= 0.0:
            raise ValueError(f"Teacher weight must be a positive finite number in --opd-teacher-urls entry {entry!r}.")
        if not url.strip():
            raise ValueError(f"Empty teacher URL in --opd-teacher-urls entry {entry!r}.")
        return url.strip(), weight
    return part, 1.0


def parse_teacher_urls(values: Iterable[str] | None) -> dict[str, list[TeacherTarget]]:
    """Parse ``NAME=URL[@WEIGHT][,URL[@WEIGHT]...]`` entries from ``--opd-teacher-urls``.

    Each name maps to a group of one or more teacher endpoints. A group with a
    single URL is plain routing (PR 3/N behavior, unchanged); a group with
    several URLs is an ensemble — every member scores the sample and the
    targets are combined as a weighted mixture in probability space. Weights
    default to 1.0 (uniform mixture).

    Splits ``NAME=`` on the first ``=`` only, so URLs containing ``=`` (e.g.
    query strings) survive intact; group members are comma-separated. Raises
    on malformed entries, duplicate names, and duplicate URLs within a group
    so misconfiguration fails at startup, not mid-rollout.
    """
    url_map: dict[str, list[TeacherTarget]] = {}
    for value in values or []:
        name, sep, spec = value.partition("=")
        name, spec = name.strip(), spec.strip()
        if not sep or not name or not spec:
            raise ValueError(f"Invalid --opd-teacher-urls entry {value!r}; expected NAME=URL[@WEIGHT][,...].")
        if name in url_map:
            raise ValueError(f"Duplicate teacher name {name!r} in --opd-teacher-urls.")
        targets = []
        for part in spec.split(","):
            part = part.strip()
            if not part:
                raise ValueError(f"Empty teacher URL in --opd-teacher-urls entry {value!r}.")
            target = _parse_teacher_target(part, value)
            if not target[0].startswith(("http://", "https://")):
                # Catches comma-split fragments of a single URL (commas separate group
                # members and cannot appear inside member URLs) at startup instead of
                # mid-rollout as an invalid scoring endpoint.
                raise ValueError(
                    f"Teacher URL {target[0]!r} in --opd-teacher-urls entry {value!r} must start with "
                    "http:// or https://. Note: ',' separates ensemble group members and a trailing "
                    "'@<float>' is parsed as the member's mixture weight."
                )
            targets.append(target)
        if len({url for url, _ in targets}) != len(targets):
            raise ValueError(f"Duplicate URL within teacher group {name!r} in --opd-teacher-urls.")
        url_map[name] = targets
    return url_map


def _teacher_targets_for_sample(args: Namespace, sample: Sample) -> list[TeacherTarget]:
    """Resolve the teacher scoring endpoint group for one sample.

    Without ``--opd-teacher-urls`` every sample goes to ``--rm-url`` (the
    original single-teacher path, unchanged). With it, the sample is routed by
    the teacher name in ``sample.metadata[--opd-teacher-key]``; samples whose
    name is missing or unknown fall back to the reserved ``default`` entry,
    and raise if no default is configured — silently distilling from the
    wrong teacher is worse than failing the rollout. The resolved group has
    one member for routing and several for an ensemble.
    """
    url_map = parse_teacher_urls(getattr(args, "opd_teacher_urls", None))
    if not url_map:
        return [(args.rm_url, 1.0)]

    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    key = getattr(args, "opd_teacher_key", "opd_teacher")
    name = metadata.get(key)
    if name is not None:
        targets = url_map.get(str(name))
        if targets is not None:
            return targets
        if DEFAULT_TEACHER_NAME in url_map:
            return url_map[DEFAULT_TEACHER_NAME]
        raise ValueError(
            f"Sample metadata[{key!r}]={name!r} matches no --opd-teacher-urls name "
            f"(known: {sorted(url_map)}) and no 'default' entry is configured."
        )
    if DEFAULT_TEACHER_NAME in url_map:
        return url_map[DEFAULT_TEACHER_NAME]
    raise ValueError(f"Sample metadata is missing teacher key {key!r} and --opd-teacher-urls has no 'default' entry.")


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


def _scoring_timeout(args: Namespace) -> int | float | None:
    """Per-request timeout for teacher/student scoring.

    ``--opd-scoring-timeout-secs`` when set; otherwise falls back to the shared
    router request timeout. Kept separate because teachers (often much larger
    than the student) need a different bound than generation requests.
    """
    timeout = getattr(args, "opd_scoring_timeout_secs", None)
    if timeout is not None:
        return timeout
    return getattr(args, "sglang_router_request_timeout_secs", None)


def _is_retryable_scoring_error(exc: BaseException) -> bool:
    if isinstance(exc, aiohttp.ClientResponseError):
        return exc.status >= 500
    return isinstance(exc, (aiohttp.ClientError, asyncio.TimeoutError))


async def _post_json_once(url: str, payload: dict[str, Any], timeout: aiohttp.ClientTimeout) -> dict[str, Any]:
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(url, json=payload) as resp:
            resp.raise_for_status()
            return await resp.json()


async def _post_json(url: str, payload: dict[str, Any], timeout_secs: int | float | None = None) -> dict[str, Any]:
    timeout = aiohttp.ClientTimeout(total=timeout_secs)
    for attempt in range(SCORING_MAX_RETRIES + 1):
        try:
            return await _post_json_once(url, payload, timeout)
        except BaseException as exc:
            if attempt >= SCORING_MAX_RETRIES or not _is_retryable_scoring_error(exc):
                raise
            await asyncio.sleep(min(2**attempt, 4) * (0.5 + 0.5 * random.random()))
    raise AssertionError("unreachable")


def _mixture_log_probs(per_teacher: list[torch.Tensor], weights: list[float]) -> torch.Tensor:
    """Combine per-teacher logprob tensors into the weighted-mixture logprob.

    log p_bar = log(sum_m w_m * p_m / sum_m w_m), computed stably in log space
    (float64) so the result is the log-probability of the mixture distribution
    — not a mean of logprobs, which would be an unnormalized geometric mean.
    """
    stacked = torch.stack([t.double() for t in per_teacher])
    log_w = torch.tensor(weights, dtype=torch.float64).log()
    log_w = log_w.view(-1, *([1] * (stacked.dim() - 1)))
    return (torch.logsumexp(stacked + log_w, dim=0) - math.log(sum(weights))).float()


def _mixture_logprob_maps(maps_per_teacher: list[LogprobMaps], weights: list[float]) -> LogprobMaps:
    """Mix per-position ``{token_id: logprob}`` maps across teachers.

    All teachers are scored at the same requested token ids per position, so
    every map must contain every id — a missing id means a malformed teacher
    response and raises rather than silently treating the probability as 0.
    """
    log_total_w = math.log(sum(weights))
    log_weights = [math.log(w) for w in weights]
    mixed: LogprobMaps = []
    for position_maps in zip(*maps_per_teacher, strict=True):
        ids = set().union(*(m.keys() for m in position_maps))
        out: dict[int, float] = {}
        for token_id in ids:
            terms = []
            for log_w, teacher_map in zip(log_weights, position_maps, strict=True):
                if token_id not in teacher_map:
                    raise ValueError(
                        f"Ensemble teacher response is missing logprob for token id {token_id}; "
                        "all group members must be scored at the same token ids."
                    )
                terms.append(log_w + teacher_map[token_id])
            max_term = max(terms)
            out[token_id] = max_term + math.log(sum(math.exp(t - max_term) for t in terms)) - log_total_w
        mixed.append(out)
    return mixed


def _teacher_responses_from_payload(reward_payload: dict[str, Any]) -> tuple[list[dict[str, Any]], list[float]]:
    """Normalize single-teacher and ensemble reward payloads to (responses, weights)."""
    if "teachers" in reward_payload:
        return reward_payload["teachers"], reward_payload["teacher_weights"]
    return [reward_payload["teacher"]], [1.0]


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


def _tail_bucket_reverse_kl(student_logps: list[float], teacher_logps: list[float]) -> float:
    """Exact reverse KL over the (k+1)-bucket partition: selected ids + one tail bucket.

    Both logprob lists are exact full-softmax logprobs at the same token ids, so
    {p(v) for v in S} + tail with tail = 1 - sum_v p(v) is a proper distribution
    and KL = sum_v p_s(v)(s_v - t_v) + tail_s * (log tail_s - log tail_t) needs
    no renormalization — the truncated estimate stays sensitive to mass the
    student pushes outside S. Python floats keep the math in float64; the tail
    uses log1p for accuracy and a student tail rounded to <= 0 contributes 0
    (the x*log(x) -> 0 limit).
    """
    student_probs = [math.exp(logp) for logp in student_logps]
    kl = sum(
        p * (s_logp - t_logp) for p, s_logp, t_logp in zip(student_probs, student_logps, teacher_logps, strict=True)
    )
    student_mass = sum(student_probs)
    if student_mass >= 1.0:
        return kl
    teacher_mass = sum(math.exp(logp) for logp in teacher_logps)
    log_tail_s = math.log1p(-student_mass)
    log_tail_t = math.log1p(-min(teacher_mass, 1.0 - TAIL_PROB_FLOOR))
    return kl + (1.0 - student_mass) * (log_tail_s - log_tail_t)


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
    tail_bucket = bool(getattr(args, "opd_topk_tail_bucket", False))
    if tail_bucket and strategy not in ("only-student", "intersection"):
        # The k+1 partition is only exact when all student logprobs come from one
        # softmax; only-teacher/union/xor mix the rollout harvest with a
        # separate rescoring pass (also validated at startup).
        raise ValueError(
            "--opd-topk-tail-bucket requires --opd-top-k-strategy only-student "
            f"or intersection, got {strategy!r}."
        )

    student_top_maps = (
        [_top_entries_to_map(entries) for entries in _student_top_logprobs(sample, response_length)]
        if strategy in STUDENT_TOP_STRATEGIES
        else [{} for _ in range(response_length)]
    )

    teacher_responses, teacher_weights = _teacher_responses_from_payload(reward_payload)
    if len(teacher_responses) == 1:
        teacher_response = teacher_responses[0]
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
    else:
        # Ensemble: every group member was scored at the student's per-position
        # top-k ids (strategy validated to only-student at startup); mix raw teacher
        # probabilities per token id BEFORE any weighting — a mixture of
        # renormalized truncations is not the truncation of the mixture.
        if strategy != "only-student":
            raise ValueError(f"Teacher ensembles require --opd-top-k-strategy only-student, got {strategy!r}.")
        teacher_top_maps = [{} for _ in range(response_length)]
        teacher_on_student_maps = _mixture_logprob_maps(
            [
                _input_logprob_maps(response, "input_token_ids_logprobs", response_length)
                for response in teacher_responses
            ],
            teacher_weights,
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

        if tail_bucket:
            reverse_kl = _tail_bucket_reverse_kl(student_logps, teacher_logps)
        else:
            weights = _reward_weights(student_logps, teacher_logps, weight_mode, normalize=normalize_weights)
            reverse_kl = sum(
                w * (s_logp - t_logp) for w, s_logp, t_logp in zip(weights, student_logps, teacher_logps, strict=True)
            )
        reverse_kls.append(reverse_kl)

    return torch.tensor(reverse_kls, dtype=torch.float32)


async def _post_teacher_group(
    targets: list[TeacherTarget], payload: dict[str, Any], timeout_secs: int | float | None
) -> dict[str, Any]:
    """Score one payload against a teacher group.

    A singleton group returns the raw response (routing/single-teacher path,
    unchanged shape). An ensemble group fans the same payload out to every
    member in parallel — wall clock is max(teacher latencies), not the sum —
    and returns the responses with their mixture weights.
    """
    responses = await asyncio.gather(*[_post_json(url, payload, timeout_secs=timeout_secs) for url, _ in targets])
    if len(responses) == 1:
        return responses[0]
    return {"teachers": list(responses), "teacher_weights": [weight for _, weight in targets]}


async def reward_func(args, sample, **kwargs):
    top_k = _get_opd_top_k(args)
    # Optional per-request timeout so a hung teacher/student scoring call cannot stall
    # the whole rollout (no-op when unset).
    request_timeout = _scoring_timeout(args)
    # Multi-teacher routing/ensemble: pick this sample's teacher group (falls back to
    # --rm-url when --opd-teacher-urls is unset).
    teacher_targets = _teacher_targets_for_sample(args, sample)
    if top_k == 0:
        return await _post_teacher_group(teacher_targets, _score_payload(sample.tokens), request_timeout)

    strategy = _get_top_k_strategy(args)
    # Per-position scoring requires a patched teacher/student server that understands
    # token_ids_logprob_positions; default off so an unpatched server keeps working.
    per_position = getattr(args, "opd_topk_per_position", False)
    prompt_len = len(sample.tokens) - sample.response_length

    teacher_top_k = top_k if strategy in TEACHER_TOP_STRATEGIES else 0
    if strategy in TEACHER_ON_STUDENT_STRATEGIES:
        student_top = _student_top_logprobs(sample, sample.response_length)
        teacher_token_ids = _unique_ids(student_top)
    else:
        student_top = None
        teacher_token_ids = None

    if student_top is not None and per_position:
        teacher_payload = _score_payload(
            sample.tokens, top_k=teacher_top_k, token_ids_positions=_per_position_ids(student_top, prompt_len)
        )
    elif teacher_token_ids is not None:
        teacher_payload = _score_payload(sample.tokens, top_k=teacher_top_k, token_ids=teacher_token_ids)
    else:
        teacher_payload = _score_payload(sample.tokens, top_k=teacher_top_k)
    group_response = await _post_teacher_group(teacher_targets, teacher_payload, request_timeout)

    reward_payload = group_response if "teachers" in group_response else {"teacher": group_response}
    if strategy in STUDENT_ON_TEACHER_STRATEGIES:
        if "teachers" in reward_payload:
            raise ValueError(f"Teacher ensembles require --opd-top-k-strategy only-student, got {strategy!r}.")
        teacher_top = _trim_input_field(
            reward_payload["teacher"]["meta_info"], "input_top_logprobs", sample.response_length
        )
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
            # The harvested per-position top-logprob lists are large (O(R*k) Python
            # objects); once the KL is computed they only bloat Ray transfers.
            sample.metadata.pop("opd_student_top_logprobs", None)
        scalar_rewards = [0.0] * len(samples)
        return scalar_rewards, scalar_rewards

    def _sampled_log_probs(reward: dict[str, Any], response_length: int) -> torch.Tensor:
        if isinstance(reward, dict) and "teachers" in reward:
            per_teacher = [_teacher_sampled_log_probs(response, response_length) for response in reward["teachers"]]
            return _mixture_log_probs(per_teacher, reward["teacher_weights"])
        return _teacher_sampled_log_probs(reward, response_length)

    teacher_log_probs = [
        _sampled_log_probs(reward, response_length)
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
