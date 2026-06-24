"""CPU-only micro-benchmark for Session Server per-turn overhead.

This benchmark measures the session-layer work without starting uvicorn, opening
HTTP sockets, or calling a model backend. It drives the same
``SessionRegistry`` / ``LinearTrajectory`` TITO path and the same response
parse/validate helper that the standalone session server uses after the backend
returns bytes.

Run it directly:

  python tests/benchmark/bench_session_server_overhead.py \
      --sessions 32 --turns 4 --input-tokens 64 --output-tokens 64 --r3-scale 1000

Add ``--incremental-r3`` to mock a backend that returns only the new per-turn
``routed_experts`` payload instead of the full accumulated sequence payload.

The reported "reply latency" is CPU-only overhead for one synthetic turn:
request JSON parse, TITO tokenization, request JSON dump with Miles-owned
``input_ids``, response parse/validate, and writing the record into in-memory
session state. Synthetic response construction is done before the measured loop,
so the numbers do not include model/backend generation.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import math
import multiprocessing
import os
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from miles.rollout.session.linear_trajectory import MAX_ASSISTANT_ROLLBACK_STEPS, SessionRegistry
from miles.rollout.session.session_types import SessionRecord
from miles.rollout.session.sessions import _dump_request_body, _parse_and_validate_response, _parse_request_body
from miles.utils.chat_template_utils import get_tito_tokenizer, resolve_fixed_chat_template
from miles.utils.http_utils import find_available_port, wait_for_server_ready
from miles.utils.processing_utils import load_tokenizer

DEFAULT_HF_CHECKPOINT = "Qwen/Qwen3-0.6B"
DEFAULT_TITO_MODEL = "qwen3"
DEFAULT_ALLOWED_APPEND_ROLES = ["user"]


@dataclass(frozen=True)
class TurnSpec:
    request_body: bytes
    response_body: bytes
    expected_prompt_token_ids: list[int]
    content_input_tokens: int
    content_output_tokens: int
    completion_tokens: int
    r3_token_count: int
    full_sequence_r3_token_count: int
    r3_raw_bytes: int
    r3_json_chars: int


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value!r}")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError(f"expected a non-negative integer, got {value!r}")
    return parsed


def _pct(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1))
    return ordered[idx]


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "mean_ms": statistics.mean(values) if values else float("nan"),
        "p50_ms": _pct(values, 0.50),
        "p95_ms": _pct(values, 0.95),
        "p99_ms": _pct(values, 0.99),
        "max_ms": max(values) if values else float("nan"),
    }


def _find_repeatable_token_id(tokenizer) -> int:
    for text in (" x", " a", " the", " token", " 0", "A"):
        for token_id in tokenizer.encode(text, add_special_tokens=False):
            decoded = tokenizer.decode([token_id], skip_special_tokens=False)
            if tokenizer.encode(decoded, add_special_tokens=False) == [token_id]:
                return token_id
    raise RuntimeError("could not find a repeatable one-token text unit for this tokenizer")


def _make_text_with_token_count(tokenizer, token_id: int, token_count: int) -> tuple[str, list[int]]:
    if token_count == 0:
        return "", []
    token_ids = [token_id] * token_count
    text = tokenizer.decode(token_ids, skip_special_tokens=False)
    roundtrip_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(roundtrip_ids) != token_count:
        raise RuntimeError(
            "repeatable token changed length after decode/encode roundtrip: "
            f"requested={token_count}, actual={len(roundtrip_ids)}"
        )
    return text, roundtrip_ids


def _make_r3_blob(raw_bytes: int) -> str:
    if raw_bytes == 0:
        return ""
    pattern = bytes(range(251))
    repeats = math.ceil(raw_bytes / len(pattern))
    raw = (pattern * repeats)[:raw_bytes]
    return base64.b64encode(raw).decode("ascii")


def _completion_token_ids(
    tito_tokenizer, tokenizer, messages: list[dict[str, Any]], assistant_message: dict[str, Any]
):
    prompt_text = tito_tokenizer.render_messages(messages, add_generation_prompt=True, tokenize=False)
    full_text = tito_tokenizer.render_messages(
        messages + [assistant_message],
        add_generation_prompt=False,
        tokenize=False,
    )
    if not full_text.startswith(prompt_text):
        raise RuntimeError("assistant response does not extend the rendered prompt")
    return tokenizer.encode(full_text[len(prompt_text) :], add_special_tokens=False)


def _build_response_body(
    assistant_message: dict[str, Any],
    completion_token_ids: list[int],
    r3_blob: str,
) -> bytes:
    output_token_logprobs = [
        [-((idx % 1024) + 1) / 1024.0, token_id] for idx, token_id in enumerate(completion_token_ids)
    ]
    response = {
        "id": "synthetic-session-overhead",
        "object": "chat.completion",
        "created": 0,
        "model": "synthetic",
        "choices": [
            {
                "index": 0,
                "message": assistant_message,
                "finish_reason": "stop",
                "meta_info": {
                    "completion_tokens": len(completion_token_ids),
                    "output_token_logprobs": output_token_logprobs,
                    "routed_experts": r3_blob,
                },
            }
        ],
    }
    return json.dumps(response, separators=(",", ":")).encode()


def _build_turn_specs(
    tokenizer,
    tito_tokenizer,
    turns: int,
    input_tokens: int,
    output_tokens: int,
    r3_scale: int,
    incremental_r3: bool,
):
    token_id = _find_repeatable_token_id(tokenizer)
    history: list[dict[str, Any]] = []
    specs: list[TurnSpec] = []
    previous_full_sequence_r3_token_count = 0

    for _turn_idx in range(turns):
        input_text, input_content_ids = _make_text_with_token_count(tokenizer, token_id, input_tokens)
        output_text, output_content_ids = _make_text_with_token_count(tokenizer, token_id, output_tokens)

        user_message = {"role": "user", "content": input_text}
        assistant_message = {"role": "assistant", "content": output_text}
        request_messages = [dict(message) for message in history] + [user_message]

        prompt_token_ids = tito_tokenizer.render_messages(
            request_messages,
            add_generation_prompt=True,
            tokenize=True,
        )
        completion_token_ids = _completion_token_ids(tito_tokenizer, tokenizer, request_messages, assistant_message)

        full_sequence_r3_token_count = max(0, len(prompt_token_ids) + len(completion_token_ids) - 1)
        if incremental_r3:
            if full_sequence_r3_token_count < previous_full_sequence_r3_token_count:
                raise RuntimeError(
                    "full-sequence r3 token count decreased between turns: "
                    f"previous={previous_full_sequence_r3_token_count}, current={full_sequence_r3_token_count}"
                )
            r3_token_count = full_sequence_r3_token_count - previous_full_sequence_r3_token_count
        else:
            r3_token_count = full_sequence_r3_token_count
        r3_raw_bytes = r3_token_count * r3_scale
        r3_blob = _make_r3_blob(r3_raw_bytes)

        request_body = json.dumps({"messages": request_messages}, separators=(",", ":")).encode()
        response_body = _build_response_body(assistant_message, completion_token_ids, r3_blob)
        specs.append(
            TurnSpec(
                request_body=request_body,
                response_body=response_body,
                expected_prompt_token_ids=prompt_token_ids,
                content_input_tokens=len(input_content_ids),
                content_output_tokens=len(output_content_ids),
                completion_tokens=len(completion_token_ids),
                r3_token_count=r3_token_count,
                full_sequence_r3_token_count=full_sequence_r3_token_count,
                r3_raw_bytes=r3_raw_bytes,
                r3_json_chars=len(r3_blob),
            )
        )

        history = request_messages + [assistant_message]
        previous_full_sequence_r3_token_count = full_sequence_r3_token_count

    return specs


def _make_registry(tokenizer, tito_tokenizer) -> SessionRegistry:
    args = SimpleNamespace(generate_multi_samples=False)
    return SessionRegistry(args, tokenizer, tito_tokenizer=tito_tokenizer)


async def _run_one_turn(session, registry: SessionRegistry, spec: TurnSpec, samples: dict[str, list[float]]) -> None:
    turn_start = time.perf_counter()

    stage_start = time.perf_counter()
    request_body = _parse_request_body(spec.request_body)
    samples["request_parse_ms"].append((time.perf_counter() - stage_start) * 1000)

    request_body["logprobs"] = True
    request_body["return_meta_info"] = True
    request_body["return_routed_experts"] = True
    request_body["no_stop_trim"] = False
    request_messages = request_body["messages"]

    stage_start = time.perf_counter()
    async with session.lock:
        prompt_token_ids = session.prepare_pretokenized(
            request_messages,
            tools=request_body.get("tools"),
            tito_tokenizer=registry.tito_tokenizer,
        )
        expected_num_assistant = session.num_assistant
    samples["tokenization_ms"].append((time.perf_counter() - stage_start) * 1000)

    request_body["input_ids"] = prompt_token_ids
    stage_start = time.perf_counter()
    _dump_request_body(request_body)
    samples["request_dump_ms"].append((time.perf_counter() - stage_start) * 1000)

    stage_start = time.perf_counter()
    response, assistant_message, completion_token_ids = _parse_and_validate_response(spec.response_body)
    samples["response_parse_validate_ms"].append((time.perf_counter() - stage_start) * 1000)

    stage_start = time.perf_counter()
    async with session.lock:
        if session.num_assistant != expected_num_assistant:
            raise RuntimeError("session state changed during a single-threaded benchmark turn")
        session.update_pretokenized_state(
            request_messages,
            assistant_message,
            prompt_token_ids=prompt_token_ids,
            completion_token_ids=completion_token_ids,
            max_trim_tokens=registry.tito_tokenizer.max_trim_tokens,
        )
        record = SessionRecord(
            timestamp=time.time(),
            method="POST",
            path="/v1/chat/completions",
            status_code=200,
            request=request_body,
            response=response,
        )
        session.append_record(record)
    samples["record_store_ms"].append((time.perf_counter() - stage_start) * 1000)

    samples["reply_latency_ms"].append((time.perf_counter() - turn_start) * 1000)


async def _validate_specs_once(tokenizer, tito_tokenizer, specs: list[TurnSpec]) -> None:
    registry = _make_registry(tokenizer, tito_tokenizer)
    session = registry.get_session(registry.create_session())

    for spec in specs:
        request_body = _parse_request_body(spec.request_body)
        request_messages = request_body["messages"]

        async with session.lock:
            prompt_token_ids = session.prepare_pretokenized(
                request_messages,
                tools=request_body.get("tools"),
                tito_tokenizer=registry.tito_tokenizer,
            )
            expected_num_assistant = session.num_assistant

        if prompt_token_ids != spec.expected_prompt_token_ids:
            raise RuntimeError(
                "TITO prompt ids differ from canonical full render: "
                f"expected={len(spec.expected_prompt_token_ids)} tokens, actual={len(prompt_token_ids)} tokens"
            )

        response, assistant_message, completion_token_ids = _parse_and_validate_response(spec.response_body)

        async with session.lock:
            if session.num_assistant != expected_num_assistant:
                raise RuntimeError("session state changed during benchmark spec validation")
            session.update_pretokenized_state(
                request_messages,
                assistant_message,
                prompt_token_ids=prompt_token_ids,
                completion_token_ids=completion_token_ids,
                max_trim_tokens=registry.tito_tokenizer.max_trim_tokens,
            )
            session.append_record(
                SessionRecord(
                    timestamp=time.time(),
                    method="POST",
                    path="/v1/chat/completions",
                    status_code=200,
                    request=request_body,
                    response=response,
                )
            )


async def _run_workload(tokenizer, tito_tokenizer, specs: list[TurnSpec], num_sessions: int):
    registry = _make_registry(tokenizer, tito_tokenizer)
    sessions = [registry.get_session(registry.create_session()) for _ in range(num_sessions)]
    samples: dict[str, list[float]] = {
        "request_parse_ms": [],
        "tokenization_ms": [],
        "request_dump_ms": [],
        "response_parse_validate_ms": [],
        "record_store_ms": [],
        "reply_latency_ms": [],
    }

    wall_start = time.perf_counter()
    for spec in specs:
        for session in sessions:
            await _run_one_turn(session, registry, spec, samples)
    wall_s = time.perf_counter() - wall_start
    return samples, wall_s


def run_bench(args) -> dict[str, Any]:
    if args.chat_template_path is not None:
        chat_template_path = args.chat_template_path
        chat_template_kwargs = None
    elif args.tito_model == "default":
        chat_template_path = None
        chat_template_kwargs = None
    else:
        chat_template_path, chat_template_kwargs = resolve_fixed_chat_template(
            args.tito_model, args.allowed_append_roles
        )

    tokenizer = load_tokenizer(args.hf_checkpoint, chat_template_path=chat_template_path, trust_remote_code=True)
    tito_tokenizer = get_tito_tokenizer(
        tokenizer,
        tokenizer_type=args.tito_model,
        chat_template_kwargs=chat_template_kwargs,
        allowed_append_roles=args.allowed_append_roles,
    )
    specs = _build_turn_specs(
        tokenizer,
        tito_tokenizer,
        turns=args.turns,
        input_tokens=args.input_tokens,
        output_tokens=args.output_tokens,
        r3_scale=args.r3_scale,
        incremental_r3=args.incremental_r3,
    )

    asyncio.run(_validate_specs_once(tokenizer, tito_tokenizer, specs))
    samples, wall_s = asyncio.run(_run_workload(tokenizer, tito_tokenizer, specs, args.sessions))
    total_turns = args.sessions * args.turns
    content_tokens = args.sessions * sum(spec.content_input_tokens + spec.content_output_tokens for spec in specs)
    output_tokens = args.sessions * sum(spec.content_output_tokens for spec in specs)
    completion_tokens = args.sessions * sum(spec.completion_tokens for spec in specs)
    retained_r3_raw_bytes = args.sessions * sum(
        spec.r3_raw_bytes for spec in specs[-(MAX_ASSISTANT_ROLLBACK_STEPS + 1) :]
    )
    total_r3_raw_bytes = args.sessions * sum(spec.r3_raw_bytes for spec in specs)

    return {
        "sessions": args.sessions,
        "turns_per_session": args.turns,
        "total_turns": total_turns,
        "input_tokens_per_turn": args.input_tokens,
        "output_tokens_per_turn": args.output_tokens,
        "r3_mode": "incremental" if args.incremental_r3 else "accumulated",
        "r3_scale_raw_bytes_per_token": args.r3_scale,
        "hf_checkpoint": args.hf_checkpoint,
        "tito_model": args.tito_model,
        "allowed_append_roles": args.allowed_append_roles,
        "chat_template_path": chat_template_path,
        "chat_template_kwargs": chat_template_kwargs,
        "wall_s": wall_s,
        "throughput_turns_per_s": total_turns / wall_s if wall_s > 0 else float("nan"),
        "throughput_content_tokens_per_s": content_tokens / wall_s if wall_s > 0 else float("nan"),
        "throughput_completion_tokens_per_s": completion_tokens / wall_s if wall_s > 0 else float("nan"),
        "throughput_output_content_tokens_per_s": output_tokens / wall_s if wall_s > 0 else float("nan"),
        "total_r3_raw_bytes_transfer_estimate": total_r3_raw_bytes,
        "retained_r3_raw_bytes_estimate": retained_r3_raw_bytes,
        "turn_specs": [
            {
                "turn_index": idx,
                "prompt_tokens": len(spec.expected_prompt_token_ids),
                "completion_tokens": spec.completion_tokens,
                "content_input_tokens": spec.content_input_tokens,
                "content_output_tokens": spec.content_output_tokens,
                "r3_token_count": spec.r3_token_count,
                "full_sequence_r3_token_count": spec.full_sequence_r3_token_count,
                "r3_raw_bytes": spec.r3_raw_bytes,
                "r3_json_chars": spec.r3_json_chars,
                "request_body_bytes": len(spec.request_body),
                "response_body_bytes": len(spec.response_body),
            }
            for idx, spec in enumerate(specs)
        ],
        "metrics": {name: _summary(values) for name, values in samples.items()},
        "raw_samples_ms": samples if args.include_raw_samples else None,
    }


def _fmt_ms_stats(stats: dict[str, float]) -> str:
    return (
        f"mean={stats['mean_ms']:.3f}ms  p50={stats['p50_ms']:.3f}ms  "
        f"p95={stats['p95_ms']:.3f}ms  p99={stats['p99_ms']:.3f}ms  max={stats['max_ms']:.3f}ms"
    )


def _fmt_block(result: dict[str, Any]) -> str:
    metrics = result["metrics"]
    last_spec = result["turn_specs"][-1]
    lines = [
        "=" * 72,
        "Session Server CPU overhead benchmark",
        "=" * 72,
        f"  sessions x turns             : {result['sessions']} x {result['turns_per_session']} "
        f"({result['total_turns']} turns)",
        f"  content tokens / turn         : input={result['input_tokens_per_turn']}  "
        f"output={result['output_tokens_per_turn']}",
        f"  r3 mode                       : {result['r3_mode']}",
        f"  r3 raw bytes / token          : {result['r3_scale_raw_bytes_per_token']}",
        f"  tokenizer / TITO              : {result['hf_checkpoint']} / {result['tito_model']}",
        f"  final-turn prompt/completion  : {last_spec['prompt_tokens']} / {last_spec['completion_tokens']} tokens",
        f"  final-turn response body      : {last_spec['response_body_bytes'] / 1024:.1f} KiB",
        f"  total r3 transfer estimate    : {result['total_r3_raw_bytes_transfer_estimate'] / 1024 / 1024:.1f} MiB raw",
        f"  retained r3 estimate          : {result['retained_r3_raw_bytes_estimate'] / 1024 / 1024:.1f} MiB raw",
        "-" * 72,
        f"  tokenization                  : {_fmt_ms_stats(metrics['tokenization_ms'])}",
        f"  record store                  : {_fmt_ms_stats(metrics['record_store_ms'])}",
        f"  reply latency                 : {_fmt_ms_stats(metrics['reply_latency_ms'])}",
        "-" * 72,
        f"  request parse                 : {_fmt_ms_stats(metrics['request_parse_ms'])}",
        f"  request dump                  : {_fmt_ms_stats(metrics['request_dump_ms'])}",
        f"  response parse+validate       : {_fmt_ms_stats(metrics['response_parse_validate_ms'])}",
        "-" * 72,
        f"  wall clock                    : {result['wall_s']:.3f}s",
        f"  throughput                    : {result['throughput_turns_per_s']:.1f} turns/s",
        f"  content-token throughput      : {result['throughput_content_tokens_per_s']:.1f} tokens/s",
        f"  completion-token throughput   : {result['throughput_completion_tokens_per_s']:.1f} tokens/s",
        "=" * 72,
    ]
    return "\n".join(lines)


def _write_json(result: dict[str, Any], path: str) -> None:
    payload = dict(result)
    if payload.get("raw_samples_ms") is None:
        payload.pop("raw_samples_ms", None)
    with Path(path).open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"[bench] wrote {path}")


# ---------------------------------------------------------------------------
# HTTP end-to-end mode: drive the REAL deployed session server over HTTP.
#
# The CPU micro-benchmark above measures the session-layer work in-process and
# serially. This mode instead stands up the actual server the rollout path runs
# (single-process ``run_session_server`` for OLD / multi-process supervisor for
# NEW), points it at a mock backend that returns the SAME pre-built large-R3
# response bytes, and drives the production-shaped workload over loopback HTTP:
# ``sessions`` concurrent sessions, each doing ``turns`` sequential chat turns.
# So the server does the real json.loads / validate / record work end to end —
# OLD pays it on one event loop under one GIL; NEW spreads it across N worker
# processes (N GILs) behind a thin router. We measure per-request reply latency
# and aggregate, plus peak RSS of the whole server process group.
# ---------------------------------------------------------------------------


def _build_server_args(
    bench_args,
    *,
    chat_template_path,
    chat_template_kwargs,
    ip: str,
    port: int,
    workers: int,
) -> SimpleNamespace:
    """A Miles-args namespace carrying exactly what the session server reads.

    ``use_rollout_routing_replay=True`` makes the server inject
    ``return_routed_experts=True`` upstream and exercise the R3-strip path, i.e.
    the production-shaped large-R3 scenario the overhead doc is about.
    """
    return SimpleNamespace(
        hf_checkpoint=bench_args.hf_checkpoint,
        chat_template_path=chat_template_path,
        apply_chat_template_kwargs=chat_template_kwargs,
        tito_model=bench_args.tito_model,
        tito_allowed_append_roles=bench_args.allowed_append_roles,
        generate_multi_samples=False,
        use_rollout_routing_replay=True,
        use_rollout_indexer_replay=False,
        miles_router_timeout=600.0,
        session_server_ip=ip,
        session_server_port=port,
        session_server_workers=workers,
        session_server_instance_id=f"bench-w{workers}",
    )


def _start_backend(response_bodies: list[bytes], ip: str) -> tuple[multiprocessing.process.BaseProcess, str]:
    """Spawn the mock R3 backend in its own process; return (proc, base_url)."""
    # ``_mock_r3_backend`` lives beside this script; ``sys.path[0]`` is this
    # script's dir, which the spawn child inherits, so a top-level import resolves
    # in both parent and child without a tests/benchmark package.
    from _mock_r3_backend import run_mock_r3_backend

    ctx = multiprocessing.get_context("spawn")
    port = find_available_port(28000)
    proc = ctx.Process(
        target=run_mock_r3_backend,
        args=(response_bodies, ip, port),
        name="bench-mock-r3-backend",
        daemon=False,
    )
    proc.start()
    wait_for_server_ready(ip, port, proc, timeout=120.0)
    return proc, f"http://{ip}:{port}"


def _start_single_process_server(
    server_args, backend_url: str, ip: str, port: int
) -> multiprocessing.process.BaseProcess:
    """OLD path: ``run_session_server`` (workers=1) as one spawned process."""
    from miles.rollout.session.session_server import run_session_server

    ctx = multiprocessing.get_context("spawn")
    proc = ctx.Process(
        target=run_session_server,
        args=(server_args, backend_url),
        name="bench-session-server-single",
        daemon=False,
    )
    proc.start()
    wait_for_server_ready(ip, port, proc, timeout=600.0)
    return proc


def _proc_tree_rss_bytes(root_pids: list[int]) -> int:
    """Summed RSS of each pid in ``root_pids`` and all its descendants.

    For OLD this is the single server process tree; for NEW it is the router
    and worker process trees. The bench driver process is deliberately excluded
    so the OLD-vs-NEW RSS comparison reflects only the server, not the driver's
    tokenizer + synthetic-body memory.
    """
    import psutil

    seen: set[int] = set()
    total = 0
    for root_pid in root_pids:
        try:
            root = psutil.Process(root_pid)
        except psutil.NoSuchProcess:
            continue
        for p in [root] + root.children(recursive=True):
            if p.pid in seen:
                continue
            seen.add(p.pid)
            try:
                total += p.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    return total


class _RSSSampler:
    """Background thread sampling peak RSS of one or more process trees."""

    def __init__(self, root_pids: list[int], interval: float = 0.25):
        import threading

        self._root_pids = [p for p in root_pids if p is not None]
        self._interval = interval
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="bench-rss-sampler", daemon=True)
        self.peak_bytes = 0

    def _run(self) -> None:
        while not self._stop.is_set():
            self.peak_bytes = max(self.peak_bytes, _proc_tree_rss_bytes(self._root_pids))
            self._stop.wait(self._interval)

    def __enter__(self) -> _RSSSampler:
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        # One final reading in case the peak landed between samples.
        self.peak_bytes = max(self.peak_bytes, _proc_tree_rss_bytes(self._root_pids))


async def _idempotent_request(client, method: str, url: str):
    """Issue an idempotent control-plane request, retrying ONCE on a connection-
    level transport fault with a fresh connection.

    Rationale (not error-masking): under OLD the single-process uvicorn can be
    GIL-blocked for many seconds serializing other sessions' 100+ MiB bodies, so
    it occasionally resets an idle keep-alive connection in its pool. A reset
    surfaces client-side as ``httpx.ReadError``/``ConnectError`` BEFORE any
    handler ran, so the request had no effect — reissuing it on a new connection
    is the correct transport-fault handling, not retrying a server-side failure
    (a non-2xx is NOT retried; it is surfaced). Only create/get/delete go through
    here; chat is never retried (it mutates session state).
    """
    import httpx

    try:
        return await client.request(method, url)
    except (httpx.ReadError, httpx.ConnectError, httpx.RemoteProtocolError):
        return await client.request(method, url)


async def _drive_one_session(
    client, base_url: str, specs: list[TurnSpec], samples: dict[str, list[float]], *, get_records: bool
) -> None:
    """create -> N x chat (-> GET records) -> delete for one session over HTTP.

    Records per-chat reply latency. The chat request body is the synthetic
    ``{"messages": [...]}`` the spec already built; the server injects input_ids
    and proxies to the mock, which returns the large-R3 response for this turn.
    The full-records GET is opt-in (``--get-records``): the retained-R3 body is
    hundreds of MiB and its purpose is the read path, not the latency/throughput
    headline, so it is off by default.
    """
    import httpx

    r = await _idempotent_request(client, "POST", f"{base_url}/sessions")
    r.raise_for_status()
    sid = r.json()["session_id"]

    for spec in specs:
        t0 = time.perf_counter()
        resp = await client.post(
            f"{base_url}/sessions/{sid}/v1/chat/completions",
            content=spec.request_body,
            headers={"content-type": "application/json"},
        )
        samples["reply_latency_ms"].append((time.perf_counter() - t0) * 1000)
        if resp.status_code != 200:
            raise httpx.HTTPStatusError(
                f"chat turn failed: {resp.status_code} {resp.text[:200]}", request=resp.request, response=resp
            )

    if get_records:
        # GET the full records (full R3 retained) to exercise the big read path.
        t0 = time.perf_counter()
        g = await _idempotent_request(client, "GET", f"{base_url}/sessions/{sid}")
        samples["get_records_ms"].append((time.perf_counter() - t0) * 1000)
        g.raise_for_status()

    d = await _idempotent_request(client, "DELETE", f"{base_url}/sessions/{sid}")
    if d.status_code != 204:
        raise RuntimeError(f"delete failed: {d.status_code} {d.text[:200]}")


async def _run_http_workload(base_url: str, specs: list[TurnSpec], num_sessions: int, *, get_records: bool):
    """Run ``num_sessions`` sessions concurrently, each doing its turns serially."""
    import httpx

    samples: dict[str, list[float]] = {"reply_latency_ms": []}
    if get_records:
        samples["get_records_ms"] = []
    # keepalive_expiry strictly below uvicorn's default timeout_keep_alive (5s):
    # under OLD a session's pooled connection can sit idle for many seconds while
    # the single event loop is GIL-blocked parsing OTHER sessions' 100+ MiB
    # bodies. If the client out-waits the server's keep-alive window, the server
    # closes the idle connection and the client's next reuse fails with a
    # ``ReadError``. Expiring the client side first forces a fresh connection and
    # removes that benchmark-harness race (we cannot raise the server's window
    # without editing the session-server source modules). ``_idempotent_request``
    # additionally retries control-plane requests once on a transport reset.
    limits = httpx.Limits(
        max_connections=num_sessions * 2,
        max_keepalive_connections=num_sessions * 2,
        keepalive_expiry=2.0,
    )
    async with httpx.AsyncClient(timeout=httpx.Timeout(600.0), limits=limits) as client:
        # Warm the connection pool / readiness with a health ping.
        await client.get(f"{base_url}/health")
        wall_start = time.perf_counter()
        await asyncio.gather(
            *(
                _drive_one_session(client, base_url, specs, samples, get_records=get_records)
                for _ in range(num_sessions)
            )
        )
        wall_s = time.perf_counter() - wall_start
    return samples, wall_s


def run_http_bench(args) -> dict[str, Any]:
    if args.chat_template_path is not None:
        chat_template_path = args.chat_template_path
        chat_template_kwargs = None
    elif args.tito_model == "default":
        chat_template_path = None
        chat_template_kwargs = None
    else:
        chat_template_path, chat_template_kwargs = resolve_fixed_chat_template(
            args.tito_model, args.allowed_append_roles
        )

    # Build the synthetic per-turn request/response bodies once (CPU, off the clock).
    tokenizer = load_tokenizer(args.hf_checkpoint, chat_template_path=chat_template_path, trust_remote_code=True)
    tito_tokenizer = get_tito_tokenizer(
        tokenizer,
        tokenizer_type=args.tito_model,
        chat_template_kwargs=chat_template_kwargs,
        allowed_append_roles=args.allowed_append_roles,
    )
    specs = _build_turn_specs(
        tokenizer,
        tito_tokenizer,
        turns=args.turns,
        input_tokens=args.input_tokens,
        output_tokens=args.output_tokens,
        r3_scale=args.r3_scale,
        incremental_r3=args.incremental_r3,
    )
    response_bodies = [spec.response_body for spec in specs]

    ip = "127.0.0.1"
    workers = args.session_server_workers
    backend_proc, backend_url = _start_backend(response_bodies, ip)

    server_objs: list[Any] = []
    server_root_pids: list[int] = []
    try:
        port = find_available_port(33000)
        server_args = _build_server_args(
            args,
            chat_template_path=chat_template_path,
            chat_template_kwargs=chat_template_kwargs,
            ip=ip,
            port=port,
            workers=workers,
        )
        base_url = f"http://{ip}:{port}"
        if workers == 1:
            proc = _start_single_process_server(server_args, backend_url, ip, port)
            server_objs.append(proc)
            server_root_pids = [proc.pid]
        else:
            from miles.rollout.session.session_supervisor import SessionServerSupervisor

            supervisor = SessionServerSupervisor(server_args, backend_url, ip, port)
            supervisor.start()
            server_objs.append(supervisor)
            # Measure ONLY the server group (router + workers), not the bench
            # driver process which holds the tokenizer + synthetic response bodies.
            server_root_pids = [w.pid for w in supervisor._workers]
            if supervisor._router is not None:
                server_root_pids.append(supervisor._router.pid)

        with _RSSSampler(server_root_pids) as rss:
            samples, wall_s = asyncio.run(
                _run_http_workload(base_url, specs, args.sessions, get_records=args.get_records)
            )
        peak_rss_bytes = rss.peak_bytes
    finally:
        for obj in server_objs:
            if isinstance(obj, multiprocessing.process.BaseProcess):
                _terminate_proc(obj)
            else:
                obj.shutdown()  # supervisor
        _terminate_proc(backend_proc)

    total_turns = args.sessions * args.turns
    content_tokens = args.sessions * sum(spec.content_input_tokens + spec.content_output_tokens for spec in specs)
    output_tokens = args.sessions * sum(spec.content_output_tokens for spec in specs)
    completion_tokens = args.sessions * sum(spec.completion_tokens for spec in specs)
    retained_r3_raw_bytes = args.sessions * sum(
        spec.r3_raw_bytes for spec in specs[-(MAX_ASSISTANT_ROLLBACK_STEPS + 1) :]
    )
    total_r3_raw_bytes = args.sessions * sum(spec.r3_raw_bytes for spec in specs)

    return {
        "mode": "http",
        "session_server_workers": workers,
        "get_records": bool(args.get_records),
        "sessions": args.sessions,
        "turns_per_session": args.turns,
        "total_turns": total_turns,
        "input_tokens_per_turn": args.input_tokens,
        "output_tokens_per_turn": args.output_tokens,
        "r3_mode": "incremental" if args.incremental_r3 else "accumulated",
        "r3_scale_raw_bytes_per_token": args.r3_scale,
        "hf_checkpoint": args.hf_checkpoint,
        "tito_model": args.tito_model,
        "allowed_append_roles": args.allowed_append_roles,
        "chat_template_path": chat_template_path,
        "chat_template_kwargs": chat_template_kwargs,
        "wall_s": wall_s,
        "throughput_turns_per_s": total_turns / wall_s if wall_s > 0 else float("nan"),
        "throughput_content_tokens_per_s": content_tokens / wall_s if wall_s > 0 else float("nan"),
        "throughput_completion_tokens_per_s": completion_tokens / wall_s if wall_s > 0 else float("nan"),
        "throughput_output_content_tokens_per_s": output_tokens / wall_s if wall_s > 0 else float("nan"),
        "total_r3_raw_bytes_transfer_estimate": total_r3_raw_bytes,
        "retained_r3_raw_bytes_estimate": retained_r3_raw_bytes,
        "peak_rss_bytes": peak_rss_bytes,
        "turn_specs": [
            {
                "turn_index": idx,
                "prompt_tokens": len(spec.expected_prompt_token_ids),
                "completion_tokens": spec.completion_tokens,
                "content_input_tokens": spec.content_input_tokens,
                "content_output_tokens": spec.content_output_tokens,
                "r3_token_count": spec.r3_token_count,
                "full_sequence_r3_token_count": spec.full_sequence_r3_token_count,
                "r3_raw_bytes": spec.r3_raw_bytes,
                "r3_json_chars": spec.r3_json_chars,
                "request_body_bytes": len(spec.request_body),
                "response_body_bytes": len(spec.response_body),
            }
            for idx, spec in enumerate(specs)
        ],
        "metrics": {name: _summary(values) for name, values in samples.items()},
        "raw_samples_ms": samples if args.include_raw_samples else None,
    }


def _terminate_proc(proc: multiprocessing.process.BaseProcess) -> None:
    import signal as _signal

    try:
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=5.0)
        if proc.is_alive() and proc.pid is not None:
            os.kill(proc.pid, _signal.SIGKILL)
            proc.join(timeout=2.0)
    except (ValueError, ProcessLookupError, AttributeError):
        pass


def _fmt_http_block(result: dict[str, Any]) -> str:
    metrics = result["metrics"]
    last_spec = result["turn_specs"][-1]
    lines = [
        "=" * 72,
        f"Session Server HTTP end-to-end benchmark (workers={result['session_server_workers']})",
        "=" * 72,
        f"  sessions x turns             : {result['sessions']} x {result['turns_per_session']} "
        f"({result['total_turns']} turns)",
        f"  content tokens / turn         : input={result['input_tokens_per_turn']}  "
        f"output={result['output_tokens_per_turn']}",
        f"  r3 mode                       : {result['r3_mode']}",
        f"  r3 raw bytes / token          : {result['r3_scale_raw_bytes_per_token']}",
        f"  tokenizer / TITO              : {result['hf_checkpoint']} / {result['tito_model']}",
        f"  final-turn prompt/completion  : {last_spec['prompt_tokens']} / {last_spec['completion_tokens']} tokens",
        f"  final-turn response body      : {last_spec['response_body_bytes'] / 1024:.1f} KiB",
        f"  total r3 transfer estimate    : {result['total_r3_raw_bytes_transfer_estimate'] / 1024 / 1024:.1f} MiB raw",
        "-" * 72,
        f"  reply latency (per chat)      : {_fmt_ms_stats(metrics['reply_latency_ms'])}",
        *(
            [f"  GET full-records latency      : {_fmt_ms_stats(metrics['get_records_ms'])}"]
            if "get_records_ms" in metrics
            else []
        ),
        "-" * 72,
        f"  wall clock                    : {result['wall_s']:.3f}s",
        f"  throughput                    : {result['throughput_turns_per_s']:.1f} turns/s",
        f"  content-token throughput      : {result['throughput_content_tokens_per_s']:.1f} tokens/s",
        f"  completion-token throughput   : {result['throughput_completion_tokens_per_s']:.1f} tokens/s",
        f"  peak RSS (server proc group)  : {result['peak_rss_bytes'] / 1024 / 1024 / 1024:.2f} GiB",
        "=" * 72,
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="CPU-only Session Server overhead benchmark")
    parser.add_argument("--sessions", type=_positive_int, default=32, help="number of sessions to create")
    parser.add_argument("--turns", type=_positive_int, default=4, help="turns per session")
    parser.add_argument("--input-tokens", type=_non_negative_int, default=64, help="new user-content tokens per turn")
    parser.add_argument(
        "--output-tokens",
        type=_non_negative_int,
        default=64,
        help="assistant-content tokens per turn",
    )
    parser.add_argument(
        "--r3-scale",
        type=_non_negative_int,
        default=1000,
        help="raw routed_experts bytes per r3 token",
    )
    parser.add_argument(
        "--incremental-r3",
        action="store_true",
        help="mock routed_experts as only the per-turn delta instead of the full accumulated sequence",
    )
    parser.add_argument("--hf-checkpoint", default=DEFAULT_HF_CHECKPOINT, help="tokenizer checkpoint or local path")
    parser.add_argument("--tito-model", default=DEFAULT_TITO_MODEL, help="TITO tokenizer family")
    parser.add_argument(
        "--allowed-append-roles",
        nargs="+",
        default=DEFAULT_ALLOWED_APPEND_ROLES,
        help="roles allowed after the pretokenized prefix",
    )
    parser.add_argument("--chat-template-path", default=None, help="explicit chat template path")
    parser.add_argument("--json-out", default=None, help="persist the run as a JSON artifact")
    parser.add_argument(
        "--include-raw-samples", action="store_true", help="include every per-turn sample in JSON output"
    )
    parser.add_argument(
        "--mode",
        choices=("cpu", "http"),
        default="cpu",
        help="cpu = serial in-process CPU micro-benchmark (floor); "
        "http = drive the real deployed server end to end over HTTP",
    )
    parser.add_argument(
        "--session-server-workers",
        type=_positive_int,
        default=1,
        help="http mode only: 1 = single-process server (OLD); N>1 = multi-process supervisor (NEW)",
    )
    parser.add_argument(
        "--get-records",
        action="store_true",
        help="http mode only: per session, GET the full (retained-R3) records before delete; off by default "
        "because the hundreds-of-MiB read is the read path, not the latency/throughput headline",
    )
    args = parser.parse_args()

    if args.mode == "http":
        result = run_http_bench(args)
        print(_fmt_http_block(result))
    else:
        result = run_bench(args)
        print(_fmt_block(result))
    if args.json_out:
        _write_json(result, args.json_out)


if __name__ == "__main__":
    main()
