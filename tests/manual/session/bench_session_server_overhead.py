"""HTTP end-to-end benchmark for Session Server per-turn overhead.

Stands up N independent single-process session servers (`run_session_server`,
the production chassis; N=1 is exactly today's single-instance deployment),
points them at a mock R3 backend that replays pre-built synthetic response
bytes, and drives `sessions` concurrent sessions x `turns` sequential chat
turns over loopback HTTP. Each session picks one instance at create time and
stays on it — the URL is the router, mirroring `OpenAIEndpointTracer.create`.
The server does its real work — request/response `json.loads`, validate, TITO
tokenization, record store — while the mock backend removes model generation
from the measurement. Reports per-chat reply latency, throughput, and peak
server-group RSS.

Run it directly:

  python tests/manual/session/bench_session_server_overhead.py \
      --sessions 32 --turns 50 --r3-scale 1000 --session-server-instances 16

Add `--incremental-r3` to mock a backend that returns only the new per-turn
`routed_experts` payload instead of the full accumulated sequence payload;
add `--get-records` to also exercise the full-records GET (read path).
With `--allowed-append-roles tool` (no `user`), turns after the first append
tool responses to assistant tool calls instead of user messages, matching
tool-only TITO surfaces such as DeepSeek V4.
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

# No miles imports at module level, deliberately: multiprocessing "spawn"
# children (the load generators and the mock backend) re-import this script as
# __mp_main__, and the miles chain drags in transformers (~10s per child). All
# miles imports live inside the driver-side functions that use them.

DEFAULT_HF_CHECKPOINT = "Qwen/Qwen3-0.6B"
DEFAULT_TITO_MODEL = "qwen3"
DEFAULT_ALLOWED_APPEND_ROLES = ["user"]

# Tool-append turns declare this in every request: some encoders (DeepSeek V4)
# render assistant history differently when no tools are declared, which breaks
# the TITO incremental diff's prefix stability server-side.
BENCH_TOOL = {
    "type": "function",
    "function": {
        "name": "bench_tool",
        "description": "Synthetic benchmark tool",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}


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


def _non_negative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError(f"expected a non-negative number, got {value!r}")
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
    tito_tokenizer,
    tokenizer,
    messages: list[dict[str, Any]],
    assistant_message: dict[str, Any],
    tools: list[dict[str, Any]] | None,
):
    prompt_text = tito_tokenizer.render_messages(messages, add_generation_prompt=True, tokenize=False, tools=tools)
    full_text = tito_tokenizer.render_messages(
        messages + [assistant_message],
        add_generation_prompt=False,
        tokenize=False,
        tools=tools,
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
    tool_appends: bool,
):
    token_id = _find_repeatable_token_id(tokenizer)
    history: list[dict[str, Any]] = []
    specs: list[TurnSpec] = []
    previous_full_sequence_r3_token_count = 0

    for turn_idx in range(turns):
        input_text, input_content_ids = _make_text_with_token_count(tokenizer, token_id, input_tokens)
        output_text, output_content_ids = _make_text_with_token_count(tokenizer, token_id, output_tokens)

        if tool_appends and history:
            input_message: dict[str, Any] = {
                "role": "tool",
                "content": input_text,
                "tool_call_id": f"call_bench_{turn_idx - 1:05d}",
                "name": "bench_tool",
            }
        else:
            input_message = {"role": "user", "content": input_text}
        assistant_message: dict[str, Any] = {"role": "assistant", "content": output_text}
        if tool_appends:
            # The appended tool turn must answer a prior assistant tool_call,
            # or templates have no call boundary to render the tool turn against.
            assistant_message["tool_calls"] = [
                {
                    "id": f"call_bench_{turn_idx:05d}",
                    "type": "function",
                    "function": {"name": "bench_tool", "arguments": {}},
                }
            ]
        request_messages = [dict(message) for message in history] + [input_message]
        tools = [BENCH_TOOL] if tool_appends else None

        prompt_token_ids = tito_tokenizer.render_messages(
            request_messages,
            add_generation_prompt=True,
            tokenize=True,
            tools=tools,
        )
        completion_token_ids = _completion_token_ids(
            tito_tokenizer, tokenizer, request_messages, assistant_message, tools
        )

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

        payload: dict[str, Any] = {"messages": request_messages}
        if tools is not None:
            payload["tools"] = tools
        request_body = json.dumps(payload, separators=(",", ":")).encode()
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


def _fmt_ms_stats(stats: dict[str, float]) -> str:
    return (
        f"mean={stats['mean_ms']:.3f}ms  p50={stats['p50_ms']:.3f}ms  "
        f"p95={stats['p95_ms']:.3f}ms  p99={stats['p99_ms']:.3f}ms  max={stats['max_ms']:.3f}ms"
    )


def _write_json(result: dict[str, Any], path: str) -> None:
    payload = dict(result)
    if payload.get("raw_samples_ms") is None:
        payload.pop("raw_samples_ms", None)
    with Path(path).open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"[bench] wrote {path}")


# ---------------------------------------------------------------------------
# Drive the REAL deployed session server over HTTP.
#
# Stands up the actual server the rollout path runs (`run_session_server`, one
# independent process per port), points it at a mock backend that returns
# pre-built large-R3 response bytes, and drives the production-shaped workload
# over loopback HTTP: `sessions` concurrent sessions, each doing `turns`
# sequential chat turns. So the server does the real json.loads / validate /
# record work end to end — one instance pays it on one event loop under one
# GIL; N instances spread the sessions across N processes with the client
# picking an instance per session (no router hop). We measure per-request
# reply latency and aggregate, plus peak RSS of the whole server process group.
# ---------------------------------------------------------------------------


def _build_server_args(
    bench_args,
    *,
    chat_template_path,
    chat_template_kwargs,
    ip: str,
    port: int,
    instance_idx: int,
) -> SimpleNamespace:
    """A Miles-args namespace carrying exactly what one session-server instance reads.

    `use_rollout_routing_replay=True` makes the server inject
    `return_routed_experts=True` upstream and exercise the R3-strip path, i.e.
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
        session_server_instance_id=f"bench-i{instance_idx}",
    )


def _start_backend(
    response_bodies: list[bytes], ip: str, inference_interval: float = 0.0, procs: int = 1
) -> tuple[list[multiprocessing.process.BaseProcess], str]:
    """Spawn the mock R3 backend as `procs` processes; return (procs, base_url).

    All shards bind the same port via SO_REUSEPORT (see `run_mock_r3_backend`)
    so the workers keep a single backend URL while the kernel spreads their
    connections across shards — the mock's stateless turn mapping makes
    any-request-to-any-shard safe. Each shard holds its own copy of
    `response_bodies`. `inference_interval` (seconds) is the simulated
    generation delay the mock waits before returning each chat response.
    """
    # `_mock_r3_backend` lives beside this script; `sys.path[0]` is this
    # script's dir, which the spawn child inherits, so a top-level import resolves
    # in both parent and child without a tests/manual/session package.
    from _mock_r3_backend import run_mock_r3_backend

    from miles.utils.http_utils import find_available_port, wait_for_server_ready

    ctx = multiprocessing.get_context("spawn")
    port = find_available_port(28000)
    backend_procs = []
    for _ in range(procs):
        proc = ctx.Process(
            target=run_mock_r3_backend,
            args=(response_bodies, ip, port, inference_interval),
            name="bench-mock-r3-backend",
            daemon=False,
        )
        proc.start()
        backend_procs.append(proc)
    # The port probe passes once ANY shard listens; shards bind before serving
    # and workers only open connections after the (much later) server readiness,
    # so all shards are accepting by the time load arrives.
    for proc in backend_procs:
        wait_for_server_ready(ip, port, proc, timeout=120.0)
    return backend_procs, f"http://{ip}:{port}"


def _proc_tree_rss_bytes(root_pids: list[int]) -> int:
    """Summed RSS of each pid in `root_pids` and all its descendants.

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


def _merge_agg(into: dict[str, Any], src: dict[str, Any]) -> None:
    for k, v in src.items():
        if k == "first_errors":
            room = 10 - len(into["first_errors"])
            if room > 0:
                into["first_errors"].extend(v[:room])
        else:
            into[k] = into.get(k, 0) + v


def _drive_workload(
    base_urls: list[str],
    request_bodies: list[bytes],
    num_sessions: int,
    *,
    get_records: bool,
    tool_interval: float,
    driver_procs: int,
):
    """Drive the HTTP workload; return (samples, agg, wall_s).

    `driver_procs == 1` runs all sessions in this process's event loop.
    `driver_procs > 1` shards the sessions across that many spawned load
    generators (each its own event loop + httpx client) so a single loop's
    connection-churn ceiling does not cap the achievable session count — see
    `_bench_load_generator`. The generators run NO tokenizer (they replay
    pre-built request bodies), so they are cheap to spawn and are deliberately
    excluded from the server RSS measurement.
    """
    # Imported here (not at module top) so the spawn children that target the
    # mock backend / session server do not transitively import httpx eagerly.
    from _bench_load_generator import lg_drive_all, load_generator_entry

    if driver_procs <= 1:
        wall_start = time.perf_counter()
        samples, agg = asyncio.run(lg_drive_all(base_urls, request_bodies, num_sessions, get_records, tool_interval))
        return samples, agg, time.perf_counter() - wall_start

    base, rem = divmod(num_sessions, driver_procs)
    shards = [base + (1 if i < rem else 0) for i in range(driver_procs)]
    shards = [s for s in shards if s > 0]

    ctx = multiprocessing.get_context("spawn")
    result_q: Any = ctx.Queue()
    procs: list[multiprocessing.process.BaseProcess] = []
    samples = {"reply_latency_ms": []}
    if get_records:
        samples["get_records_ms"] = []
    agg: dict[str, Any] = {
        "completed_turns": 0,
        "chat_server_errors": 0,
        "chat_transport_errors": 0,
        "get_ok": 0,
        "get_server_errors": 0,
        "get_transport_errors": 0,
        "sessions_ok": 0,
        "first_errors": [],
    }
    wall_start = time.perf_counter()
    try:
        for n in shards:
            p = ctx.Process(
                target=load_generator_entry,
                args=(base_urls, request_bodies, n, get_records, tool_interval, result_q),
                name="bench-loadgen",
                daemon=False,
            )
            p.start()
            procs.append(p)
        # Drain all results BEFORE joining (a large queued item would otherwise
        # block the feeder and deadlock the join).
        for _ in shards:
            s, a = result_q.get(timeout=2400)
            samples["reply_latency_ms"].extend(s["reply_latency_ms"])
            if get_records:
                samples["get_records_ms"].extend(s.get("get_records_ms", []))
            _merge_agg(agg, a)
        for p in procs:
            p.join(timeout=120)
    finally:
        for p in procs:
            _terminate_proc(p)
    wall_s = time.perf_counter() - wall_start
    return samples, agg, wall_s


def run_http_bench(args) -> dict[str, Any]:
    from miles.utils.chat_template_utils import get_tito_tokenizer, resolve_fixed_chat_template
    from miles.utils.http_utils import find_available_port, is_port_available, wait_for_server_ready
    from miles.utils.processing_utils import load_tokenizer

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
        tool_appends="user" not in args.allowed_append_roles,
    )
    response_bodies = [spec.response_body for spec in specs]

    ip = "127.0.0.1"
    instances = args.session_server_instances
    backend_procs, backend_url = _start_backend(
        response_bodies, ip, inference_interval=args.inference_interval, procs=args.mock_backend_procs
    )

    server_procs: list[multiprocessing.process.BaseProcess] = []
    try:
        from miles.rollout.session.server import run_session_server

        # Consecutive ports from one free base, mirroring the production
        # `--session-server-port start end` range deployment.
        base_port = find_available_port(33000)
        ports = list(range(base_port, base_port + instances))
        for port in ports[1:]:
            if not is_port_available(port):
                raise RuntimeError(f"port {port} in the bench range {ports[0]}..{ports[-1]} is already in use")
        base_urls = [f"http://{ip}:{port}" for port in ports]

        # Spawn every instance before waiting on any (matches
        # start_session_server): N instances pay ~one transformers import of
        # wall-time, not N.
        ctx = multiprocessing.get_context("spawn")
        for idx, port in enumerate(ports):
            server_args = _build_server_args(
                args,
                chat_template_path=chat_template_path,
                chat_template_kwargs=chat_template_kwargs,
                ip=ip,
                port=port,
                instance_idx=idx,
            )
            proc = ctx.Process(
                target=run_session_server,
                args=(server_args, backend_url),
                name=f"bench-session-server-{idx}",
                daemon=False,
            )
            proc.start()
            server_procs.append(proc)
        for port, proc in zip(ports, server_procs, strict=True):
            wait_for_server_ready(ip, port, proc, timeout=180.0)
        # Measure ONLY the server group, not the bench driver process which
        # holds the tokenizer + synthetic response bodies.
        server_root_pids = [p.pid for p in server_procs]

        request_bodies = [spec.request_body for spec in specs]
        with _RSSSampler(server_root_pids) as rss:
            samples, agg, wall_s = _drive_workload(
                base_urls,
                request_bodies,
                args.sessions,
                get_records=args.get_records,
                tool_interval=args.tool_interval,
                driver_procs=args.bench_driver_procs,
            )
        peak_rss_bytes = rss.peak_bytes
    finally:
        for proc in server_procs:
            _terminate_proc(proc)
        for proc in backend_procs:
            _terminate_proc(proc)

    total_turns = args.sessions * args.turns
    completed_turns = agg.get("completed_turns", total_turns)
    # token-throughput is reported on COMPLETED work (frac=1 for clean runs, so
    # this matches the strict-mode numbers; honest when sessions fail partway).
    frac = (completed_turns / total_turns) if total_turns else 0.0
    content_tokens = (
        frac * args.sessions * sum(spec.content_input_tokens + spec.content_output_tokens for spec in specs)
    )
    output_tokens = frac * args.sessions * sum(spec.content_output_tokens for spec in specs)
    completion_tokens = frac * args.sessions * sum(spec.completion_tokens for spec in specs)
    # Records are never pruned (the training data path reads them via GET
    # /sessions/{id}), so the retained R3 equals everything transferred.
    total_r3_raw_bytes = args.sessions * sum(spec.r3_raw_bytes for spec in specs)

    return {
        "mode": "http",
        "session_server_instances": instances,
        "bench_driver_procs": args.bench_driver_procs,
        "mock_backend_procs": args.mock_backend_procs,
        "get_records": bool(args.get_records),
        "inference_interval_s": args.inference_interval,
        "tool_interval_s": args.tool_interval,
        "sessions": args.sessions,
        "turns_per_session": args.turns,
        "total_turns": total_turns,
        "completed_turns": completed_turns,
        "sessions_ok": agg.get("sessions_ok"),
        "chat_server_errors": agg.get("chat_server_errors"),
        "chat_transport_errors": agg.get("chat_transport_errors"),
        "get_ok": agg.get("get_ok"),
        "get_server_errors": agg.get("get_server_errors"),
        "get_transport_errors": agg.get("get_transport_errors"),
        "first_errors": agg.get("first_errors", []),
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
        "throughput_turns_per_s": completed_turns / wall_s if wall_s > 0 else float("nan"),
        "throughput_content_tokens_per_s": content_tokens / wall_s if wall_s > 0 else float("nan"),
        "throughput_completion_tokens_per_s": completion_tokens / wall_s if wall_s > 0 else float("nan"),
        "throughput_output_content_tokens_per_s": output_tokens / wall_s if wall_s > 0 else float("nan"),
        "total_r3_raw_bytes_transfer_estimate": total_r3_raw_bytes,
        "retained_r3_raw_bytes_estimate": total_r3_raw_bytes,
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
        f"Session Server HTTP end-to-end benchmark (instances={result['session_server_instances']}, "
        f"driver_procs={result.get('bench_driver_procs', 1)}, backend_procs={result.get('mock_backend_procs', 1)})",
        "=" * 72,
        f"  sessions x turns             : {result['sessions']} x {result['turns_per_session']} "
        f"({result['total_turns']} turns)",
        f"  content tokens / turn         : input={result['input_tokens_per_turn']}  "
        f"output={result['output_tokens_per_turn']}",
        f"  inference / tool interval     : {result['inference_interval_s']}s / {result['tool_interval_s']}s",
        f"  r3 mode                       : {result['r3_mode']}",
        f"  r3 raw bytes / token          : {result['r3_scale_raw_bytes_per_token']}",
        f"  tokenizer / TITO              : {result['hf_checkpoint']} / {result['tito_model']}",
        f"  final-turn prompt/completion  : {last_spec['prompt_tokens']} / {last_spec['completion_tokens']} tokens",
        f"  final-turn response body      : {last_spec['response_body_bytes'] / 1024:.1f} KiB",
        f"  total r3 transfer estimate    : {result['total_r3_raw_bytes_transfer_estimate'] / 1024 / 1024:.1f} MiB raw",
        "-" * 72,
        f"  sessions ok / total           : {result.get('sessions_ok')} / {result['sessions']}    "
        f"completed turns: {result.get('completed_turns')} / {result['total_turns']}",
        f"  chat errors (srv / transport) : {result.get('chat_server_errors')} / {result.get('chat_transport_errors')}",
        *(
            [
                f"  GET ok / errors (srv / trans) : {result.get('get_ok')} / "
                f"{result.get('get_server_errors')} / {result.get('get_transport_errors')}"
            ]
            if result.get("get_records")
            else []
        ),
        *([f"  first errors                  : {result['first_errors'][:5]}"] if result.get("first_errors") else []),
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
    parser = argparse.ArgumentParser(description="HTTP end-to-end Session Server overhead benchmark")
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
        help="roles allowed after the pretokenized prefix; without 'user' the bench appends "
        "tool responses to assistant tool calls instead of user turns (tool-only TITO surfaces)",
    )
    parser.add_argument("--chat-template-path", default=None, help="explicit chat template path")
    parser.add_argument("--json-out", default=None, help="persist the run as a JSON artifact")
    parser.add_argument(
        "--include-raw-samples", action="store_true", help="include every per-turn sample in JSON output"
    )
    parser.add_argument(
        "--session-server-instances",
        type=_positive_int,
        default=1,
        help="number of independent single-process session servers on consecutive ports; "
        "each session picks one at create time (the URL is the router)",
    )
    parser.add_argument(
        "--get-records",
        action="store_true",
        help="per session, GET the full (retained-R3) records before delete; off by default "
        "because the hundreds-of-MiB read is the read path, not the latency/throughput headline",
    )
    parser.add_argument(
        "--inference-interval",
        type=_non_negative_float,
        default=0.0,
        help="seconds the mock backend waits before returning each chat response "
        "(simulated generation time). Default 0 = respond immediately (continuous back-to-back).",
    )
    parser.add_argument(
        "--tool-interval",
        type=_non_negative_float,
        default=0.0,
        help="seconds a session idles after each chat response before sending its next "
        "turn (simulated tool/env step producing the next input). Default 0 = fire turns back-to-back.",
    )
    parser.add_argument(
        "--mock-backend-procs",
        type=_positive_int,
        default=8,
        help="shard the mock backend across this many SO_REUSEPORT processes on one port. A single "
        "process saturates one core streaming large-R3 bodies to 16 workers and caps the measurement; "
        "the default keeps the mock off the critical path. Set 1 to reproduce older single-process runs.",
    )
    parser.add_argument(
        "--bench-driver-procs",
        type=_positive_int,
        default=1,
        help="shard the client load across this many load-generator processes (each its "
        "own event loop + httpx client). Default 1. Raise it to drive many hundreds of sessions without a "
        "single event loop's connection-churn ceiling; mirrors the real multi-worker rollout.",
    )
    args = parser.parse_args()

    result = run_http_bench(args)
    print(_fmt_http_block(result))
    if args.json_out:
        _write_json(result, args.json_out)


if __name__ == "__main__":
    main()
