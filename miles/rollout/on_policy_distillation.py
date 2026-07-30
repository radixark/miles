import asyncio
import math
import time
import weakref
from argparse import Namespace
from collections.abc import Iterable
from typing import Any

import httpx
import torch

from miles.utils.http_utils import post
from miles.utils.types import Sample

TopLogprobs = list[list[Any]]
LogprobMaps = list[dict[int, float]]

TOP_K_STRATEGIES = {"only-student", "only-teacher", "intersection", "union", "xor"}
REWARD_WEIGHT_MODES = {"student_p", "teacher_p", "none"}

_SCORING_RETRY_BACKOFF_S = 2.0

# One semaphore per (event loop, limit) so the in-flight bound is scoped to the
# loop actually issuing the requests and follows a changed CLI limit.
_SCORING_SEMAPHORES: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()

STUDENT_TOP_STRATEGIES = TOP_K_STRATEGIES - {"only-teacher"}
TEACHER_TOP_STRATEGIES = TOP_K_STRATEGIES - {"only-student"}
TEACHER_ON_STUDENT_STRATEGIES = {"only-student", "union", "xor"}
STUDENT_ON_TEACHER_STRATEGIES = {"only-teacher", "union", "xor"}


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


def _get_top_k_scoring_block_size(args: Namespace) -> int:
    return max(0, int(args.opd_top_k_scoring_block_size))


def _score_payload(
    input_ids: list[int],
    response_length: int,
    top_k: int = 0,
    token_ids: list[int] | None = None,
) -> dict[str, Any]:
    if not 0 <= response_length <= len(input_ids):
        raise ValueError(
            f"OPD scoring response window is out of bounds: response_length={response_length}, "
            f"tokens={len(input_ids)}."
        )
    prompt_length = len(input_ids) - response_length
    if prompt_length <= 0:
        raise ValueError(
            "OPD scoring needs at least one prompt token before the response window: "
            f"tokens={len(input_ids)}, response_length={response_length}."
        )
    payload = {
        "input_ids": input_ids,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 0,
            "skip_special_tokens": False,
        },
        "return_logprob": True,
        # SGLang aligns input logprobs to tokens from logprob_start_len, with a
        # placeholder first entry. Keep the complete prefix in input_ids, but
        # only materialize logprobs from one token before the response so the
        # reply covers exactly the response window (see
        # generate_utils/prefill_logprobs.py for the same convention).
        "logprob_start_len": prompt_length - 1,
    }
    if top_k > 0:
        payload["top_logprobs_num"] = top_k
    if token_ids:
        payload["token_ids_logprob"] = token_ids
    return payload


def _student_score_url(args: Namespace) -> str:
    return f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"


def _scoring_semaphore(args: Namespace) -> asyncio.Semaphore | None:
    """Bound concurrent scoring requests per event loop.

    A whole rollout batch finishes generation together, so without a bound
    every sample's scoring request dogpiles the external server at once and
    queue time burns each request's deadline.
    """
    limit = int(args.opd_scoring_max_inflight)
    if limit <= 0:
        return None
    loop = asyncio.get_running_loop()
    semaphores_by_limit = _SCORING_SEMAPHORES.setdefault(loop, {})
    semaphore = semaphores_by_limit.get(limit)
    if semaphore is None:
        semaphore = asyncio.Semaphore(limit)
        semaphores_by_limit[limit] = semaphore
    return semaphore


async def _scoring_request(
    args: Namespace,
    url: str,
    payload: dict[str, Any],
    *,
    sample: Sample,
    target: str,
    action: str,
) -> dict[str, Any]:
    """Issue one scoring request through the shared HTTP client.

    Adds the bounded-transport contract on top of http_utils.post: an
    in-flight bound, a total deadline shared across retries, a bounded retry
    policy, with failures annotated using sample identity.
    """
    timeout_s = float(args.opd_scoring_timeout)
    retries = max(0, int(args.opd_scoring_retries))
    max_attempts = retries + 1

    deadline_s = time.monotonic() + timeout_s
    semaphore = _scoring_semaphore(args)
    semaphore_acquired = False
    attempts = 0

    def scoring_error(error: BaseException) -> RuntimeError:
        return RuntimeError(
            f"OPD scoring request to {url} failed after {attempts} attempt(s): {error!r} "
            f"(target={target}, timeout={timeout_s}s, "
            f"input_tokens={len(payload.get('input_ids', []))}, "
            f"sample index={sample.index}, group={sample.group_index})"
        )

    try:
        if semaphore is not None:
            remaining_s = deadline_s - time.monotonic()
            try:
                if remaining_s <= 0:
                    raise TimeoutError(f"deadline of {timeout_s}s exhausted while waiting for an in-flight slot")
                await asyncio.wait_for(semaphore.acquire(), timeout=remaining_s)
                semaphore_acquired = True
            except (TimeoutError, asyncio.TimeoutError) as error:
                raise scoring_error(error) from error

        for _ in range(max_attempts):
            remaining_s = deadline_s - time.monotonic()
            if remaining_s <= 0:
                error = TimeoutError(f"deadline of {timeout_s}s exhausted before the next attempt")
                raise scoring_error(error) from error

            attempts += 1
            try:
                # max_retries=1 gives exactly one attempt: the retry policy,
                # attempt count and deadline are owned here where they can be
                # bounded and reported.
                request = (
                    post(url, payload, max_retries=1)
                    if action == "post"
                    else post(url, None, max_retries=1, action=action)
                )
                return await asyncio.wait_for(request, timeout=remaining_s)
            except (TimeoutError, asyncio.TimeoutError, httpx.HTTPError) as error:
                remaining_s = deadline_s - time.monotonic()
                if attempts >= max_attempts or remaining_s <= 0:
                    raise scoring_error(error) from error
                # Keep part of the remaining deadline available for the retry.
                backoff_s = min(_SCORING_RETRY_BACKOFF_S, remaining_s / 2)
                await asyncio.sleep(backoff_s)
    finally:
        if semaphore_acquired:
            semaphore.release()


async def _scoring_post(
    args: Namespace,
    url: str,
    payload: dict[str, Any],
    *,
    sample: Sample,
    target: str,
) -> dict[str, Any]:
    return await _scoring_request(args, url, payload, sample=sample, target=target, action="post")


async def _scoring_get(
    args: Namespace,
    url: str,
    *,
    sample: Sample,
    target: str,
) -> dict[str, Any]:
    return await _scoring_request(args, url, {}, sample=sample, target=target, action="get")


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
    trimmed = values[1:][-response_length:] if response_length > 0 else []
    if len(trimmed) != response_length:
        raise ValueError(
            f"Scoring response meta_info.{field} covers {len(trimmed)} positions, expected {response_length}."
        )
    return trimmed


def _input_logprob_maps(response: dict[str, Any], field: str, response_length: int) -> LogprobMaps:
    return [
        _top_entries_to_map(entries) for entries in _trim_input_field(response["meta_info"], field, response_length)
    ]


def _teacher_sampled_log_probs(response: dict[str, Any], sample: Sample) -> torch.Tensor:
    """Extract exactly one finite teacher log-prob per response token.

    Raises when the scored positions do not line up one-to-one with the
    sample's response tokens, instead of silently training on shifted scores.
    """
    response_length = sample.response_length
    meta_info = response.get("meta_info")
    if not isinstance(meta_info, dict):
        raise ValueError(
            f"Scoring response has no meta_info dict (sample index={sample.index}, group={sample.group_index})."
        )
    entries = _trim_input_field(meta_info, "input_token_logprobs", response_length)
    if response_length == 0:
        return torch.zeros((0,), dtype=torch.float32)

    response_tokens = [int(token) for token in sample.tokens[-response_length:]]
    scored_tokens = [int(entry[1]) for entry in entries]
    if scored_tokens != response_tokens:
        raise ValueError(
            "Scoring token alignment mismatch: "
            f"expected response tail {response_tokens[:8]}... len={len(response_tokens)}, "
            f"got {scored_tokens[:8]}... len={len(scored_tokens)} "
            f"(sample index={sample.index}, group={sample.group_index})."
        )

    log_probs = [entry[0] for entry in entries]
    if any(log_prob is None for log_prob in log_probs):
        raise ValueError(
            f"Scoring returned None for a response-token logprob "
            f"(sample index={sample.index}, group={sample.group_index})."
        )
    tensor = torch.tensor([float(log_prob) for log_prob in log_probs], dtype=torch.float32)
    if not torch.isfinite(tensor).all():
        raise ValueError(
            f"Scoring returned a non-finite response-token logprob "
            f"(sample index={sample.index}, group={sample.group_index})."
        )
    return tensor


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


def _normalize_top_logprob_rows(
    top_logprobs: Iterable[Iterable[list[Any]]],
    *,
    response_length: int,
    top_k: int,
    source: str,
) -> TopLogprobs:
    rows = list(top_logprobs)
    if len(rows) != response_length:
        raise ValueError(f"{source} top-k covers {len(rows)} positions, expected {response_length}.")

    normalized = []
    for position, entries in enumerate(rows):
        row = []
        seen_ids = set()
        for entry in entries or []:
            if entry is None:
                continue
            if not isinstance(entry, (list, tuple)) or len(entry) < 2:
                raise ValueError(f"{source} top-k row {position} contains a malformed entry: {entry!r}.")
            logprob = float(entry[0])
            token_id = int(entry[1])
            if not math.isfinite(logprob):
                raise ValueError(f"{source} top-k row {position} contains a non-finite logprob.")
            if token_id < 0:
                raise ValueError(f"{source} top-k row {position} contains a negative token id: {token_id}.")
            if token_id in seen_ids:
                raise ValueError(f"{source} top-k row {position} contains duplicate token id {token_id}.")
            seen_ids.add(token_id)
            row.append([logprob, token_id])
        if len(row) > top_k:
            raise ValueError(
                f"{source} top-k row {position} contains {len(row)} entries, configured top-k is {top_k}."
            )
        normalized.append(row)
    return normalized


def _top_k_scoring_blocks(top_logprobs: TopLogprobs, block_size: int) -> Iterable[tuple[int, int, list[int]]]:
    if block_size <= 0:
        raise ValueError("OPD top-k scoring block size must be positive.")
    for start in range(0, len(top_logprobs), block_size):
        end = min(start + block_size, len(top_logprobs))
        yield start, end, _unique_ids(top_logprobs[start:end])


def _weight_version(value: Any, *, source: str) -> str:
    if value is None or str(value) == "":
        raise ValueError(f"{source} did not report a usable weight_version.")
    return str(value)


async def _student_weight_version(args: Namespace, sample: Sample) -> str:
    response = await _scoring_get(
        args,
        f"http://{args.sglang_router_ip}:{args.sglang_router_port}/model_info",
        sample=sample,
        target="student-version",
    )
    if not isinstance(response, dict) or "weight_version" not in response:
        raise ValueError("Student model_info response is missing weight_version.")
    return _weight_version(response["weight_version"], source="Student model_info")


def _response_weight_version(response: dict[str, Any], *, target: str) -> str:
    meta_info = response.get("meta_info")
    if not isinstance(meta_info, dict) or "weight_version" not in meta_info:
        raise ValueError(f"{target} scoring response is missing meta_info.weight_version.")
    return _weight_version(meta_info["weight_version"], source=f"{target} scoring response")


def _validate_block_token_alignment(
    response: dict[str, Any],
    sample: Sample,
    *,
    start: int,
    end: int,
    target: str,
) -> None:
    meta_info = response.get("meta_info")
    if not isinstance(meta_info, dict):
        raise ValueError(
            f"{target} scoring response has no meta_info dict "
            f"(sample index={sample.index}, group={sample.group_index})."
        )

    entries = _trim_input_field(meta_info, "input_token_logprobs", end - start)
    response_start = len(sample.tokens) - sample.response_length
    expected_tokens = [int(token) for token in sample.tokens[response_start + start : response_start + end]]
    try:
        scored_tokens = [int(entry[1]) for entry in entries]
    except (IndexError, TypeError, ValueError) as error:
        raise ValueError(
            f"{target} scoring returned malformed token-alignment entries for response positions [{start}, {end})."
        ) from error
    if scored_tokens != expected_tokens:
        raise ValueError(
            f"{target} scoring token alignment mismatch for response positions [{start}, {end}): "
            f"expected {expected_tokens[:8]}... len={len(expected_tokens)}, "
            f"got {scored_tokens[:8]}... len={len(scored_tokens)} "
            f"(sample index={sample.index}, group={sample.group_index})."
        )


def _compact_block_logprobs(
    response: dict[str, Any],
    candidate_rows: TopLogprobs,
    *,
    start: int,
    end: int,
    target: str,
) -> TopLogprobs:
    scored_rows = _trim_input_field(response["meta_info"], "input_token_ids_logprobs", end - start)
    compact_rows = []
    for position, (candidates, scored_entries) in enumerate(
        zip(candidate_rows[start:end], scored_rows, strict=True),
        start=start,
    ):
        scored_map = {}
        for entry in scored_entries or []:
            if entry is None or not isinstance(entry, (list, tuple)) or len(entry) < 2:
                raise ValueError(f"{target} scoring row {position} contains a malformed candidate entry: {entry!r}.")
            logprob = float(entry[0])
            token_id = int(entry[1])
            if not math.isfinite(logprob):
                raise ValueError(
                    f"{target} scoring row {position} contains a non-finite logprob for token id {token_id}."
                )
            scored_map[token_id] = logprob

        compact_row = []
        for candidate in candidates:
            token_id = _top_entry_token_id(candidate)
            if token_id not in scored_map:
                raise ValueError(f"{target} scoring row {position} is missing candidate token id {token_id}.")
            compact_row.append([scored_map[token_id], token_id])
        compact_rows.append(compact_row)
    return compact_rows


async def _score_top_k_in_blocks(
    args: Namespace,
    url: str,
    sample: Sample,
    candidate_rows: TopLogprobs,
    *,
    target: str,
    require_stable_weight_version: bool = False,
) -> dict[str, Any]:
    """Score position-local candidates without a response-wide Cartesian product.

    SGLang accepts one arbitrary-ID set per request, not one set per token
    position. Each request therefore covers a bounded response-position block,
    uses only that block's candidate union, and immediately compacts the reply
    back to the original per-position rows.
    """
    response_length = sample.response_length
    if len(candidate_rows) != response_length:
        raise ValueError(f"{target} candidate rows cover {len(candidate_rows)} positions, expected {response_length}.")
    if response_length == 0:
        return {"meta_info": {"input_token_ids_logprobs": [None]}}

    block_size = _get_top_k_scoring_block_size(args)
    if block_size <= 0:
        raise ValueError("Blocked OPD top-k scoring requires --opd-top-k-scoring-block-size > 0.")

    prompt_length = len(sample.tokens) - response_length
    transaction_attempts = max(0, int(args.opd_scoring_retries)) + 1 if require_stable_weight_version else 1
    last_version_error: RuntimeError | None = None

    for transaction_attempt in range(1, transaction_attempts + 1):
        expected_version = await _student_weight_version(args, sample) if require_stable_weight_version else None
        compact_rows: TopLogprobs = [[] for _ in range(response_length)]
        version_error = None

        for start, end, candidate_ids in _top_k_scoring_blocks(candidate_rows, block_size):
            if not candidate_ids:
                continue

            block_input_ids = sample.tokens[: prompt_length + end]
            response = await _scoring_post(
                args,
                url,
                _score_payload(block_input_ids, end - start, token_ids=candidate_ids),
                sample=sample,
                target=target,
            )
            _validate_block_token_alignment(response, sample, start=start, end=end, target=target)
            compact_rows[start:end] = _compact_block_logprobs(
                response,
                candidate_rows,
                start=start,
                end=end,
                target=target,
            )

            if require_stable_weight_version:
                block_version = _response_weight_version(response, target=target)
                if block_version != expected_version:
                    version_error = RuntimeError(
                        f"{target} weight version changed during blocked OPD scoring: "
                        f"expected {expected_version!r}, got {block_version!r} for response positions "
                        f"[{start}, {end}) (sample index={sample.index}, group={sample.group_index})."
                    )
                    break

        if version_error is None:
            meta_info = {"input_token_ids_logprobs": [None, *compact_rows]}
            if expected_version is not None:
                meta_info["weight_version"] = expected_version
            return {"meta_info": meta_info}

        last_version_error = version_error
        if transaction_attempt < transaction_attempts:
            await asyncio.sleep(_SCORING_RETRY_BACKOFF_S)

    raise RuntimeError(
        f"Unable to score all {target} blocks with one weight version after "
        f"{transaction_attempts} attempt(s); refusing to assemble mixed-version scores."
    ) from last_version_error


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


async def reward_func(args: Namespace, sample: Sample, **kwargs: Any) -> dict[str, Any]:
    top_k = _get_opd_top_k(args)
    if top_k == 0:
        return await _scoring_post(
            args,
            args.rm_url,
            _score_payload(sample.tokens, sample.response_length),
            sample=sample,
            target="teacher",
        )

    strategy = _get_top_k_strategy(args)
    block_size = _get_top_k_scoring_block_size(args)

    if block_size > 0 and strategy == "only-student":
        student_top = _normalize_top_logprob_rows(
            _student_top_logprobs(sample, sample.response_length),
            response_length=sample.response_length,
            top_k=top_k,
            source="Student",
        )
        teacher_response = await _score_top_k_in_blocks(
            args,
            args.rm_url,
            sample,
            student_top,
            target="teacher",
        )
        return {"teacher": teacher_response}

    teacher_token_ids = None
    if strategy in TEACHER_ON_STUDENT_STRATEGIES:
        student_top = _student_top_logprobs(sample, sample.response_length)
        teacher_token_ids = _unique_ids(student_top)

    teacher_payload = _score_payload(
        sample.tokens,
        sample.response_length,
        top_k=top_k if strategy in TEACHER_TOP_STRATEGIES else 0,
        token_ids=teacher_token_ids,
    )
    teacher_response = await _scoring_post(args, args.rm_url, teacher_payload, sample=sample, target="teacher")

    reward_payload = {"teacher": teacher_response}
    if strategy in STUDENT_ON_TEACHER_STRATEGIES:
        teacher_top = _trim_input_field(teacher_response["meta_info"], "input_top_logprobs", sample.response_length)
        if block_size > 0 and strategy == "only-teacher":
            _teacher_sampled_log_probs(teacher_response, sample)
            teacher_top = _normalize_top_logprob_rows(
                teacher_top,
                response_length=sample.response_length,
                top_k=top_k,
                source="Teacher",
            )
            reward_payload["student_on_teacher"] = await _score_top_k_in_blocks(
                args,
                _student_score_url(args),
                sample,
                teacher_top,
                target="student",
                require_stable_weight_version=True,
            )
        else:
            student_token_ids = _unique_ids(teacher_top)
            reward_payload["student_on_teacher"] = await _scoring_post(
                args,
                _student_score_url(args),
                _score_payload(sample.tokens, sample.response_length, token_ids=student_token_ids),
                sample=sample,
                target="student",
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

    if _get_opd_top_k(args) > 0:
        for sample, reward in zip(samples, raw_rewards, strict=True):
            sample.opd_reverse_kl = _compute_topk_reverse_kl(args, sample, reward)
        scalar_rewards = [0.0] * len(samples)
        return scalar_rewards, scalar_rewards

    teacher_log_probs = [
        _teacher_sampled_log_probs(reward, sample) for reward, sample in zip(raw_rewards, samples, strict=True)
    ]

    for sample, t_log_probs in zip(samples, teacher_log_probs, strict=True):
        sample.teacher_log_probs = t_log_probs

    # Return scalar rewards for GRPO/PPO advantage estimator.
    # For pure on-policy distillation, we use 0.0 as the task reward.
    # The learning signal comes entirely from the OPD KL penalty.
    # If you have task rewards, you can add them here.
    scalar_rewards = [0.0] * len(samples)

    return scalar_rewards, scalar_rewards
