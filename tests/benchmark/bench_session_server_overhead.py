"""CPU-only micro-benchmark for Session Server per-turn overhead.

This benchmark measures the session-layer work without starting uvicorn, opening
HTTP sockets, or calling a model backend. It drives the same
``SessionRegistry`` / ``LinearTrajectory`` TITO path and the same response
parse/validate helper that the standalone session server uses after the backend
returns bytes.

Run it directly:

  python tests/benchmark/bench_session_server_overhead.py \
      --sessions 32 --turns 4 --input-tokens 64 --output-tokens 64 --r3-scale 1000

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


def _build_turn_specs(tokenizer, tito_tokenizer, turns: int, input_tokens: int, output_tokens: int, r3_scale: int):
    token_id = _find_repeatable_token_id(tokenizer)
    history: list[dict[str, Any]] = []
    specs: list[TurnSpec] = []

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

        r3_token_count = max(0, len(prompt_token_ids) + len(completion_token_ids) - 1)
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
                r3_raw_bytes=r3_raw_bytes,
                r3_json_chars=len(r3_blob),
            )
        )

        history = request_messages + [assistant_message]

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

    return {
        "sessions": args.sessions,
        "turns_per_session": args.turns,
        "total_turns": total_turns,
        "input_tokens_per_turn": args.input_tokens,
        "output_tokens_per_turn": args.output_tokens,
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
        "retained_r3_raw_bytes_estimate": retained_r3_raw_bytes,
        "turn_specs": [
            {
                "turn_index": idx,
                "prompt_tokens": len(spec.expected_prompt_token_ids),
                "completion_tokens": spec.completion_tokens,
                "content_input_tokens": spec.content_input_tokens,
                "content_output_tokens": spec.content_output_tokens,
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
        f"  r3 raw bytes / token          : {result['r3_scale_raw_bytes_per_token']}",
        f"  tokenizer / TITO              : {result['hf_checkpoint']} / {result['tito_model']}",
        f"  final-turn prompt/completion  : {last_spec['prompt_tokens']} / {last_spec['completion_tokens']} tokens",
        f"  final-turn response body      : {last_spec['response_body_bytes'] / 1024:.1f} KiB",
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
        help="raw routed_experts bytes per accumulated token",
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
    args = parser.parse_args()

    result = run_bench(args)
    print(_fmt_block(result))
    if args.json_out:
        _write_json(result, args.json_out)


if __name__ == "__main__":
    main()
