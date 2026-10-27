import asyncio
import math
from argparse import Namespace
from collections.abc import Iterable
from typing import Any

import aiohttp
import torch

from miles.utils.types import Sample

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
    url_map = parse_teacher_urls(args.opd_teacher_urls)
    if not url_map:
        return args.rm_url

    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    key = args.opd_teacher_key
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


def _get_opd_top_k(args: Namespace) -> int:
    return max(0, int(args.opd_log_prob_top_k or 0))


def _get_top_k_strategy(args: Namespace) -> str:
    strategy = args.opd_top_k_strategy
    if strategy not in TOP_K_STRATEGIES:
        raise ValueError(f"Unknown OPD top-k strategy: {strategy}")
    return strategy


def _get_reward_weight_mode(args: Namespace) -> str:
    mode = args.opd_reward_weight_mode
    if mode not in REWARD_WEIGHT_MODES:
        raise ValueError(f"Unknown OPD reward weight mode: {mode}")
    return mode


def _score_payload(
    input_ids: list[int],
    top_k: int = 0,
    token_ids: list[int] | None = None,
    token_ids_positions: list[list[int]] | None = None,
    extra_key: str | None = None,
    logprob_start_len: int = 0,
) -> dict[str, Any]:
    payload = {
        "input_ids": input_ids,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 0,
            "skip_special_tokens": False,
        },
        "return_logprob": True,
        "logprob_start_len": logprob_start_len,
    }
    if top_k > 0:
        payload["top_logprobs_num"] = top_k
    if token_ids_positions is not None:
        # Per-position scoring (patched sglang): one id-list per input position, so the
        # teacher returns each position's own ids (sparse) instead of the global union
        # broadcast to every position (dense O(R^2)). The server indexes the list by
        # absolute input position, so it is independent of logprob_start_len.
        payload["token_ids_logprob_positions"] = token_ids_positions
    elif token_ids:
        payload["token_ids_logprob"] = token_ids
    if extra_key is not None:
        payload["extra_key"] = extra_key
    return payload


def _per_position_ids(top_logprobs: TopLogprobs, prompt_len: int) -> list[list[int]]:
    """Build one token-id list per scored input position for ``token_ids_logprob_positions``.

    ``top_logprobs`` is per response position (length == response_length). The server
    reads the list by absolute input position (sgl-project/sglang#27421), so prompt
    positions are padded with empty id-lists and response position r lands at index
    ``prompt_len + r`` whatever ``logprob_start_len`` the payload asks for.
    """
    per_pos: list[list[int]] = [[] for _ in range(prompt_len)]
    for entries in top_logprobs:
        per_pos.append([_top_entry_token_id(e) for e in (entries or []) if e is not None])
    return per_pos


def _response_logprob_start_len(sample: Sample) -> int:
    """First position to score so SGLang returns ``[None, *response logprobs]``.

    SGLang returns one input-logprob row per position from ``logprob_start_len`` with
    the first row a ``None`` placeholder, and reuses a cached prefix only up to
    ``logprob_start_len``. Starting at 0 made the teacher return (and recompute,
    since no radix-cache prefix can match) a row for every prompt token that
    ``_trim_input_field`` then discarded.
    """
    prompt_len = len(sample.tokens) - sample.response_length
    return max(prompt_len - 1, 0)


def _student_score_url(args: Namespace) -> str:
    return f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"


# One pooled session per event loop, reused by every teacher/student scoring call.
_scoring_sessions: dict[asyncio.AbstractEventLoop, aiohttp.ClientSession] = {}


def _scoring_session() -> aiohttp.ClientSession:
    loop = asyncio.get_running_loop()
    session = _scoring_sessions.get(loop)
    if session is None or session.closed:
        session = aiohttp.ClientSession(
            # limit=0: rollout concurrency bounds the scoring calls, not the pool, as
            # it did when every call opened its own session.
            connector=aiohttp.TCPConnector(limit=0),
            timeout=aiohttp.ClientTimeout(total=None),
            cookie_jar=aiohttp.DummyCookieJar(),
        )
        _scoring_sessions[loop] = session
    return session


async def close_scoring_sessions() -> None:
    """Close every pooled scoring session; called once when the rollout shuts down."""
    current = asyncio.get_running_loop()
    sessions = list(_scoring_sessions.items())
    _scoring_sessions.clear()
    for loop, session in sessions:
        if loop is current:
            await session.close()
        elif loop.is_running():
            await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(session.close(), loop))


async def _post_json(url: str, payload: dict[str, Any], timeout_secs: int | float | None = None) -> dict[str, Any]:
    timeout = aiohttp.ClientTimeout(total=timeout_secs)
    async with _scoring_session().post(url, json=payload, timeout=timeout) as resp:
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
    if response_length <= 0:
        return []
    # SGLang's first input logprob/top-logprob position is a placeholder; the response
    # rows are the tail whatever logprob_start_len the request used.
    response_values = values[1:][-response_length:]
    if len(response_values) != response_length:
        raise ValueError(
            f"Teacher meta_info.{field} has {len(values)} rows, too few for a response of {response_length} tokens."
        )
    return response_values


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


async def reward_func(args: Namespace, sample: Sample, **kwargs: Any) -> float:
    """Score ``sample`` with its teacher and keep only the OPD signal training reads.

    The teacher reply holds a log-prob row for every scored position (top-k entries too with
    ``--opd-log-prob-top-k>0``). It is reduced on arrival to ``sample.teacher_log_probs`` or
    ``sample.opd_reverse_kl`` so the reply does not ride on the sample through Ray, and the
    reward slot gets the scalar task reward of pure OPD, 0.0, which every rollout metric can read.
    """
    _store_opd_signal(args, sample, await _query_teacher(args, sample))
    return 0.0


async def _query_teacher(args: Namespace, sample: Sample) -> dict[str, Any]:
    top_k = _get_opd_top_k(args)
    # Optional per-request timeout so a hung teacher/student scoring call cannot stall
    # the whole rollout (no-op when unset).
    request_timeout = args.sglang_router_request_timeout_secs
    # Multi-teacher routing: pick this sample's teacher endpoint (falls back to
    # --rm-url when --opd-teacher-urls is unset).
    teacher_url = _teacher_url_for_sample(args, sample)
    # the teacher is frozen, so reusing its cached prompt prefix cannot change its logprobs
    logprob_start_len = _response_logprob_start_len(sample)
    if top_k == 0:
        return await _post_json(
            teacher_url,
            _score_payload(sample.tokens, logprob_start_len=logprob_start_len),
            timeout_secs=request_timeout,
        )

    strategy = _get_top_k_strategy(args)
    # Per-position scoring requires a patched teacher/student server that understands
    # token_ids_logprob_positions; default off so an unpatched server keeps working.
    per_position = args.opd_topk_per_position
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
            sample.tokens,
            top_k=teacher_top_k,
            token_ids_positions=_per_position_ids(student_top, prompt_len),
            logprob_start_len=logprob_start_len,
        )
    elif teacher_token_ids is not None:
        teacher_payload = _score_payload(
            sample.tokens, top_k=teacher_top_k, token_ids=teacher_token_ids, logprob_start_len=logprob_start_len
        )
    else:
        teacher_payload = _score_payload(sample.tokens, top_k=teacher_top_k, logprob_start_len=logprob_start_len)
    teacher_response = await _post_json(teacher_url, teacher_payload, timeout_secs=request_timeout)

    reward_payload = {"teacher": teacher_response}
    if strategy in STUDENT_ON_TEACHER_STRATEGIES:
        teacher_top = _trim_input_field(teacher_response["meta_info"], "input_top_logprobs", sample.response_length)
        student_extra_key = sample.kv_cache_namespace
        # The live student keeps scoring from position 0: a prefix match would reuse prompt
        # KV cached under the weights the sample was generated with, so these logprobs
        # would change once the weights moved. Only the frozen teacher starts at the response.
        if per_position:
            student_payload = _score_payload(
                sample.tokens,
                token_ids_positions=_per_position_ids(teacher_top, prompt_len),
                extra_key=student_extra_key,
            )
        else:
            student_payload = _score_payload(
                sample.tokens, token_ids=_unique_ids(teacher_top), extra_key=student_extra_key
            )
        reward_payload["student_on_teacher"] = await _post_json(
            _student_score_url(args), student_payload, timeout_secs=request_timeout
        )

    return reward_payload


def _store_opd_signal(args: Namespace, sample: Sample, reward_payload: dict[str, Any]) -> None:
    """Reduce a teacher reply to the per-token OPD signal stored on ``sample``.

    ``--opd-log-prob-top-k=0`` preserves the original sampled-token OPD path:
    store teacher log-probs and let training compute ``student_logp - teacher_logp``.

    ``--opd-log-prob-top-k>0`` follows the practical recipe from
    "Rethinking On-Policy Distillation" by forming a top-k token set per
    response position and storing a precomputed weighted reverse-KL estimate.
    """
    if _get_opd_top_k(args) > 0:
        sample.opd_reverse_kl = _compute_topk_reverse_kl(args, sample, reward_payload)
    else:
        sample.teacher_log_probs = _teacher_sampled_log_probs(reward_payload, sample.response_length)


def post_process_rewards(args: Namespace, samples: list[Sample], **kwargs: Any) -> tuple[list[float], list[float]]:
    """Hand the advantage estimator the zero task reward of pure on-policy distillation.

    ``reward_func`` already stored each sample's OPD signal. A custom reward function may still
    return the raw teacher reply, as ``reward_func`` used to; its signal is extracted here.
    """
    for sample in samples:
        reward = sample.get_reward_value(args)
        if isinstance(reward, dict):
            _store_opd_signal(args, sample, reward)
        elif (sample.opd_reverse_kl if _get_opd_top_k(args) > 0 else sample.teacher_log_probs) is None:
            raise ValueError(
                f"OPD sample {sample.index} has neither a teacher reply as its reward nor a stored OPD signal; "
                "score it with miles.rollout.on_policy_distillation.reward_func."
            )

    # Return scalar rewards for GRPO/PPO advantage estimator.
    # For pure on-policy distillation, we use 0.0 as the task reward.
    # The learning signal comes entirely from the OPD KL penalty.
    # If you have task rewards, you can add them here.
    scalar_rewards = [0.0] * len(samples)

    return scalar_rewards, scalar_rewards
