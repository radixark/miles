"""Event-loop-responsiveness micro-benchmark for the standalone SessionServer.

Why this exists (directional / non-blocking):
The session server offloads stateless CPU work (large JSON parse/dump of the
chat request + response, including hundreds-of-KB `routed_experts` blobs) off the
single asyncio event loop onto a bounded `SessionServer.cpu_executor`. The claim
is that this keeps the one event loop responsive under heavy-response load so the
liveness probe `GET /health` (handled inline on the loop) stays fast. It does NOT
claim higher total CPU throughput: the GIL still serializes the Python JSON work,
so this measures latency/responsiveness, not aggregate CPU.

Honest reporting note: deeper per-stage server-internal timing is intentionally
NOT instrumented into the production hot path (no probes added to chat_completions).
The offloaded stage's cost is reported indirectly via the measured single-response
parse cost (`json.loads` of one heavy response body), which is the per-call CPU
component the executor offload moves off the event loop.

This file is named `bench_*` so pytest does NOT auto-collect it (no flaky timing
test in CI). Run it directly:

  # Run once, print a human-readable block:
  python tests/benchmark/bench_session_responsiveness.py

  # Run once, persist a reviewable JSON artifact (used for before/after):
  python tests/benchmark/bench_session_responsiveness.py --label after \
      --json-out .humanize/.../benchmarks/session-responsiveness-after.json

  # Compare two persisted runs into a markdown verdict:
  python tests/benchmark/bench_session_responsiveness.py --compare \
      before.json after.json --out compare.md

Method: fire K concurrent chats across K DISTINCT sessions (distinct sessions are
not gated, so they run in parallel), each producing a large response that forces a
CPU-heavy `json.loads` in `_parse_and_validate_response`. Concurrently, a separate
thread polls `GET /health` every ~10ms and records each round-trip latency. We
report chat throughput, response body size, and `/health` latency percentiles. All
timing is client-side `time.perf_counter`.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import requests

# Quiet the uvicorn access logs from both servers so the summary block is the
# only thing on stdout (the per-request "GET /health 200 OK" lines bury it).
for _name in ("uvicorn", "uvicorn.access", "uvicorn.error"):
    logging.getLogger(_name).setLevel(logging.WARNING)

from miles.rollout.session.session_server import SessionServer
from miles.utils.http_utils import find_available_port
from miles.utils.test_utils.mock_sglang_server import MockSGLangServer, ProcessResult, with_mock_server
from miles.utils.test_utils.uvicorn_thread_server import UvicornThreadServer

# --- Tunable constants (kept modest so a run finishes well under a minute) ---
HF_CHECKPOINT = "Qwen/Qwen3-0.6B"
K_CHATS = 64  # concurrent chats, one per distinct (ungated) session
# The session server json.loads the WHOLE response body to validate it, but the
# `routed_experts` value is opaque to it (never decoded). To isolate the CPU
# parse cost the offload targets — without inflating the on-loop body I/O that
# scales with raw byte size — we make routed_experts a STRUCTURED blob (nested
# numeric arrays). Structured JSON parses ~7-8x slower per byte than one big
# string, so a modest ~2 MiB body costs ~15-30ms to parse. Run INLINE on the loop
# (pre-offload) that stalls every /health probe queued behind it; offloaded to
# cpu_executor it runs off-loop and the probe stays fast.
BLOB_ROWS = 900  # rows of BLOB_ROW_WIDTH floats -> ~1.5 MiB body, ~18ms parse/call
BLOB_ROW_WIDTH = 256
HEALTH_POLL_INTERVAL_S = 0.01  # ~10ms between /health probes
HEALTH_POLL_DURATION_S = 4.0  # how long the health poller runs (also caps the load window)
CHAT_TEXT = "responsiveness-ok"


def _make_large_blob() -> list:
    # Nested numeric arrays, like a routed_experts logits buffer: expensive to
    # json.loads per byte (many tokens), which is exactly the CPU work the
    # session server offloads off the event loop.
    row = [round(i * 0.001, 4) for i in range(BLOB_ROW_WIDTH)]
    return [list(row) for _ in range(BLOB_ROWS)]


def _patch_mock_chat_response_heavy(blob: list):
    """Patch the mock chat response into the shape the session server validates
    (output_token_logprobs as (logprob, token_id)), and inject a large STRUCTURED
    routed_experts blob into meta_info so the session server's json.loads of the
    response body is genuinely CPU-heavy. The blob is opaque to the session server
    (it only validates output_token_logprobs / message), matching production where
    routed_experts is an opaque buffer the session layer never decodes.
    """
    original_chat_response = MockSGLangServer._compute_chat_completions_response

    def patched_chat_response(self, payload: dict) -> dict:
        response = original_chat_response(self, payload)
        choice = response["choices"][0]
        logprobs_content = choice["logprobs"]["content"]
        output_token_logprobs = [
            (item["logprob"], self.tokenizer.convert_tokens_to_ids(item["token"])) for item in logprobs_content
        ]
        choice["meta_info"]["output_token_logprobs"] = output_token_logprobs
        choice["meta_info"]["completion_tokens"] = len(output_token_logprobs)
        choice["meta_info"]["routed_experts"] = blob
        return response

    return patch.object(MockSGLangServer, "_compute_chat_completions_response", new=patched_chat_response)


@contextmanager
def _router_env(process_fn, blob: list, *, latency: float = 0.0):
    with _patch_mock_chat_response_heavy(blob):
        with with_mock_server(model_name=HF_CHECKPOINT, process_fn=process_fn, latency=latency) as backend:
            args = SimpleNamespace(
                miles_router_timeout=30,
                hf_checkpoint=HF_CHECKPOINT,
                chat_template_path=None,
                trajectory_manager="linear_trajectory",
                tito_allowed_append_roles=["tool", "system"],
                # Make the session server request + echo a big routed_experts blob,
                # so the per-response json.loads is genuinely CPU-heavy.
                use_rollout_routing_replay=True,
            )
            server_obj = SessionServer(args, backend_url=backend.url)

            port = find_available_port(31000)
            server = UvicornThreadServer(server_obj.app, host="127.0.0.1", port=port)
            server.start()
            # uvicorn.Config(log_level="info") reconfigures these on startup, so
            # re-quiet them now that both servers are up (keeps stdout to the summary).
            for _name in ("uvicorn", "uvicorn.access", "uvicorn.error"):
                logging.getLogger(_name).setLevel(logging.WARNING)
            url = f"http://127.0.0.1:{port}"
            try:
                yield SimpleNamespace(url=url, backend=backend, server=server)
            finally:
                server.stop()


def _create_session(url: str) -> str:
    resp = requests.post(f"{url}/sessions", timeout=5.0)
    assert resp.status_code == 200, resp.text
    return resp.json()["session_id"]


def _chat(url: str, session_id: str) -> tuple[int, int]:
    """Returns (status_code, response_body_size_bytes)."""
    resp = requests.post(
        f"{url}/sessions/{session_id}/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "drive-load"}]},
        timeout=60.0,
    )
    return resp.status_code, len(resp.content)


class _HealthPoller(threading.Thread):
    """Polls GET /health on a fixed cadence and records each round-trip latency
    (seconds, client-side perf_counter). Runs until stop() or duration elapses."""

    def __init__(self, url: str, interval_s: float, max_duration_s: float):
        super().__init__(daemon=True)
        self.url = f"{url}/health"
        self.interval_s = interval_s
        self.max_duration_s = max_duration_s
        self.latencies_s: list[float] = []
        self.errors = 0
        self._stop_evt = threading.Event()
        self._session = requests.Session()

    def stop(self) -> None:
        self._stop_evt.set()

    def run(self) -> None:
        deadline = time.perf_counter() + self.max_duration_s
        while not self._stop_evt.is_set() and time.perf_counter() < deadline:
            t0 = time.perf_counter()
            try:
                resp = self._session.get(self.url, timeout=5.0)
                dt = time.perf_counter() - t0
                if resp.status_code == 200:
                    self.latencies_s.append(dt)
                else:
                    self.errors += 1
            except requests.RequestException:
                self.errors += 1
            time.sleep(self.interval_s)


def _pct(values_ms: list[float], q: float) -> float:
    if not values_ms:
        return float("nan")
    ordered = sorted(values_ms)
    idx = min(len(ordered) - 1, int(q * len(ordered)))
    return ordered[idx]


def _measure_per_stage_cpu_ms(blob: list) -> dict:
    """Benchmark-side (NOT production hot-path) timing of the three stateless CPU
    stages the session server offloads to its cpu_executor: request parse, request
    dump (with Miles-owned input_ids), and response parse+validate. `parse_ms` uses
    plain json.loads so it is identical and comparable across before/after builds;
    `validate_ms` calls the real helper and is recorded only where it exists (the
    pre-offload build predates the module-level helpers -> null)."""
    # Heavy response parse (the dominant offloaded stage): plain json.loads, so it
    # runs identically on the pre-offload build too.
    sample_resp = {
        "choices": [
            {
                "message": {"role": "assistant", "content": "x"},
                "meta_info": {
                    "output_token_logprobs": [[-0.01 * i, 100 + i] for i in range(64)],
                    "completion_tokens": 64,
                    "routed_experts": blob,
                },
            }
        ]
    }
    sample_resp_body = json.dumps(sample_resp).encode()
    _t = time.perf_counter()
    json.loads(sample_resp_body)
    parse_ms = (time.perf_counter() - _t) * 1000

    # Request dump (encode the body with Miles-owned input_ids): plain json.dumps.
    sample_req = {
        "messages": [{"role": "user", "content": "drive-load"}],
        "input_ids": list(range(4096)),
        "logprobs": True,
        "return_meta_info": True,
    }
    _t = time.perf_counter()
    json.dumps(sample_req).encode()
    dump_ms = (time.perf_counter() - _t) * 1000

    # Full parse+validate via the real helper (after-only; null where absent).
    validate_ms = None
    try:
        from miles.rollout.session.sessions import _parse_and_validate_response

        _t = time.perf_counter()
        _parse_and_validate_response(sample_resp_body)
        validate_ms = (time.perf_counter() - _t) * 1000
    except Exception:
        validate_ms = None

    return {"parse_ms": parse_ms, "dump_ms": dump_ms, "validate_ms": validate_ms}


def run_bench() -> dict:
    blob = _make_large_blob()
    # The CPU cost the offload targets, measured up front (benchmark-side, not in
    # the production hot path) so the artifact states how heavy each offloaded
    # stage is. parse_ms is the dominant one and is comparable across builds.
    per_stage = _measure_per_stage_cpu_ms(blob)
    parse_ms = per_stage["parse_ms"]

    def process_fn(_prompt: str) -> ProcessResult:
        # routed_experts is injected by the response patch (kept off the str-typed
        # dataclass field); this just drives a small assistant message.
        return ProcessResult(text=CHAT_TEXT, finish_reason="stop")

    with _router_env(process_fn, blob) as env:
        session_ids = [_create_session(env.url) for _ in range(K_CHATS)]

        # Warm one chat so we can log the real response body size before the run.
        warm_status, warm_size = _chat(env.url, session_ids[0])
        assert warm_status == 200, f"warm-up chat failed: {warm_status}"

        # Baseline: /health latency with no chat load (single thread, short burst).
        baseline = _HealthPoller(env.url, HEALTH_POLL_INTERVAL_S, max_duration_s=0.7)
        baseline.start()
        baseline.join()
        baseline_ms = [s * 1000 for s in baseline.latencies_s]

        # Under load: poll /health while waves of K concurrent chats keep the
        # backend + cpu_executor busy. A single wave of K first-turn chats is
        # fast, so we fire back-to-back waves until the poll window elapses; this
        # sustains heavy-response load long enough to sample /health meaningfully.
        # Each chat uses its own fresh, distinct session (distinct sessions are
        # ungated, so they run in parallel).
        poller = _HealthPoller(env.url, HEALTH_POLL_INTERVAL_S, HEALTH_POLL_DURATION_S)
        poller.start()

        results: list[tuple[int, int]] = []
        waves = 0
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=K_CHATS) as pool:
            while time.perf_counter() - t0 < HEALTH_POLL_DURATION_S:
                fresh_ids = [_create_session(env.url) for _ in range(K_CHATS)]
                futures = [pool.submit(_chat, env.url, sid) for sid in fresh_ids]
                results.extend(f.result(timeout=120.0) for f in futures)
                waves += 1
        chat_wall_s = time.perf_counter() - t0

        poller.stop()
        poller.join()

        statuses = [s for s, _ in results]
        sizes = [sz for _, sz in results]
        ok = sum(1 for s in statuses if s == 200)
        load_ms = [s * 1000 for s in poller.latencies_s]

    total = len(results)
    return {
        "k_chats": K_CHATS,
        "waves": waves,
        "parse_ms": parse_ms,
        "dump_ms": per_stage["dump_ms"],
        "validate_ms": per_stage["validate_ms"],
        "total_chats": total,
        "ok_chats": ok,
        "failed_chats": total - ok,
        "response_body_bytes": warm_size if not sizes else max(sizes),
        "chat_wall_s": chat_wall_s,
        "chat_throughput_per_s": ok / chat_wall_s if chat_wall_s > 0 else float("nan"),
        "health_samples_under_load": len(load_ms),
        "health_errors_under_load": poller.errors,
        "health_baseline_samples": len(baseline_ms),
        # Computed percentiles (persisted so the artifact is reviewable without
        # re-deriving from raw samples); raw samples kept alongside for audit.
        "health_baseline_p50_ms": _pct(baseline_ms, 0.50),
        "health_baseline_p95_ms": _pct(baseline_ms, 0.95),
        "health_baseline_max_ms": max(baseline_ms) if baseline_ms else float("nan"),
        "health_load_p50_ms": _pct(load_ms, 0.50),
        "health_load_p95_ms": _pct(load_ms, 0.95),
        "health_load_p99_ms": _pct(load_ms, 0.99),
        "health_load_max_ms": max(load_ms) if load_ms else float("nan"),
        "health_load_mean_ms": statistics.mean(load_ms) if load_ms else float("nan"),
        "baseline_ms": baseline_ms,
        "load_ms": load_ms,
    }


def _fmt_block(r: dict) -> str:
    base = r["baseline_ms"]
    load = r["load_ms"]
    validate_str = f"{r['validate_ms']:.1f}ms" if r.get("validate_ms") is not None else "n/a (pre-offload)"
    lines = [
        "=" * 64,
        "SessionServer event-loop responsiveness benchmark",
        "=" * 64,
        f"  concurrency / wave (K)      : {r['k_chats']}",
        f"  waves driven                : {r['waves']}  (total {r['total_chats']} chats)",
        f"  response body size (actual) : {r['response_body_bytes'] / 1024:.1f} KiB",
        f"  per-stage CPU (offloaded)   : parse={r['parse_ms']:.1f}ms  dump={r.get('dump_ms', float('nan')):.2f}ms  "
        f"validate={validate_str}",
        f"  chats ok / failed           : {r['ok_chats']} / {r['failed_chats']}",
        f"  chat wall-clock             : {r['chat_wall_s']:.3f} s",
        f"  chat throughput             : {r['chat_throughput_per_s']:.1f} chats/s",
        "-" * 64,
        f"  /health baseline (no load)  : n={r['health_baseline_samples']}, "
        f"p50={_pct(base, 0.50):.2f}ms  p95={_pct(base, 0.95):.2f}ms  "
        f"max={(max(base) if base else float('nan')):.2f}ms",
        f"  /health UNDER LOAD          : n={r['health_samples_under_load']}, "
        f"errors={r['health_errors_under_load']}",
        f"      p50 = {_pct(load, 0.50):.2f} ms",
        f"      p95 = {_pct(load, 0.95):.2f} ms",
        f"      p99 = {_pct(load, 0.99):.2f} ms",
        f"      max = {(max(load) if load else float('nan')):.2f} ms",
        f"      mean= {(statistics.mean(load) if load else float('nan')):.2f} ms",
        "=" * 64,
    ]
    return "\n".join(lines)


def _persist(result: dict, path: str, label: str | None, commit: str | None, dirty: bool | None) -> None:
    payload = dict(result)
    payload["label"] = label
    payload["commit"] = commit
    payload["dirty"] = dirty
    payload["blob_rows"] = BLOB_ROWS
    payload["blob_row_width"] = BLOB_ROW_WIDTH
    payload["health_poll_interval_s"] = HEALTH_POLL_INTERVAL_S
    payload["health_poll_duration_s"] = HEALTH_POLL_DURATION_S
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[bench] wrote {path}")


def _compare(before_path: str, after_path: str, out_path: str | None) -> str:
    with open(before_path) as f:
        b = json.load(f)
    with open(after_path) as f:
        a = json.load(f)

    def _delta(metric: str) -> str:
        bv, av = b.get(metric), a.get(metric)
        if bv is None or av is None:
            return "n/a"
        d = av - bv
        pct = (d / bv * 100) if bv else float("nan")
        return f"{d:+.2f} ms ({pct:+.1f}%)"

    rows = [
        ("/health p50 (load)", "health_load_p50_ms"),
        ("/health p95 (load)", "health_load_p95_ms"),
        ("/health p99 (load)", "health_load_p99_ms"),
        ("/health max (load)", "health_load_max_ms"),
        ("/health p95 (baseline)", "health_baseline_p95_ms"),
    ]
    # p95/p99 verdict: separate material improvements from noise-level changes.
    noise_ms = 25.0
    verdict_lines = []
    for label, key in (("p95", "health_load_p95_ms"), ("p99", "health_load_p99_ms")):
        bv, av = b.get(key), a.get(key)
        if bv is None or av is None:
            verdict_lines.append(f"- {label}: n/a")
            continue
        if av < bv - noise_ms:
            verdict_lines.append(
                f"- {label}: IMPROVED (before {bv:.1f}ms -> after {av:.1f}ms, beyond ±{noise_ms:.0f}ms noise)"
            )
        elif abs(av - bv) <= noise_ms:
            verdict_lines.append(
                f"- {label}: NO REGRESSION (before {bv:.1f}ms -> after {av:.1f}ms, within ±{noise_ms:.0f}ms noise)"
            )
        else:
            verdict_lines.append(
                f"- {label}: REGRESSED (before {bv:.1f}ms -> after {av:.1f}ms, beyond {noise_ms:.0f}ms noise)"
            )

    lines = [
        "# Session-server responsiveness: before vs after offload",
        "",
        f"- before: `{before_path}` commit `{b.get('commit')}` dirty={b.get('dirty')} label={b.get('label')}",
        f"- after : `{after_path}` commit `{a.get('commit')}` dirty={a.get('dirty')} label={a.get('label')}",
        f"- K={a.get('k_chats')} chats/wave, response body ~{a.get('response_body_bytes', 0) / 1024:.0f} KiB, "
        f"blob {a.get('blob_rows')}x{a.get('blob_row_width')} floats",
        "",
        "| metric | before | after | delta |",
        "|---|---|---|---|",
    ]
    for label, key in rows:
        bv, av = b.get(key), a.get(key)
        bs = f"{bv:.2f}" if isinstance(bv, (int, float)) else "n/a"
        as_ = f"{av:.2f}" if isinstance(av, (int, float)) else "n/a"
        lines.append(f"| {label} | {bs} ms | {as_} ms | {_delta(key)} |")
    b_validate = f"{b['validate_ms']:.1f} ms" if b.get("validate_ms") is not None else "n/a"
    a_validate = f"{a['validate_ms']:.1f} ms" if a.get("validate_ms") is not None else "n/a"
    lines += [
        "| chat throughput | "
        f"{b.get('chat_throughput_per_s', float('nan')):.1f}/s | {a.get('chat_throughput_per_s', float('nan')):.1f}/s | "
        f"{(a.get('chat_throughput_per_s', 0) - b.get('chat_throughput_per_s', 0)):+.1f}/s |",
        "| /health errors (load) | "
        f"{b.get('health_errors_under_load')} | {a.get('health_errors_under_load')} | — |",
        "| per-call parse (json.loads) | "
        f"{b.get('parse_ms', float('nan')):.1f} ms | {a.get('parse_ms', float('nan')):.1f} ms | {_delta('parse_ms')} |",
        f"| per-call validate (helper) | {b_validate} | {a_validate} | — |",
        "",
        "## p95/p99 verdict (under load)",
        *verdict_lines,
        "",
        "## Interpretation",
        "Under K concurrent heavy responses the inline build serializes every per-response `json.loads` "
        "ON the single event loop, so the parses stack: the loop is blocked for roughly K x parse_ms before "
        "it can service a queued `/health` probe (the measured before-p95 ~= K x single-parse cost). Offloading "
        "the parse to the bounded cpu_executor frees the loop to service `/health` between awaits and to drive "
        "more chat waves, so both `/health` tail latency and chat throughput improve markedly at this scale. The "
        "GIL still serializes the Python parse work, so this is a responsiveness/tail-latency effect, not an "
        "aggregate-CPU gain. Single short windows are noisy (a before-p95 can rest on one stall), so pool samples "
        "across iterations and inspect the per-iteration spread before trusting any delta.",
        "",
    ]
    text = "\n".join(lines)
    if out_path:
        with open(out_path, "w") as f:
            f.write(text)
        print(f"[bench] wrote {out_path}")
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description="SessionServer event-loop responsiveness benchmark")
    parser.add_argument("--json-out", default=None, help="persist the run as a JSON artifact at this path")
    parser.add_argument("--label", default=None, help="label recorded in the JSON artifact (e.g. before/after)")
    parser.add_argument("--commit", default=None, help="commit SHA recorded in the artifact")
    parser.add_argument("--dirty", action="store_true", help="record that the checkout was dirty")
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("BEFORE", "AFTER"),
        default=None,
        help="compare two persisted JSON artifacts instead of running the benchmark",
    )
    parser.add_argument("--out", default=None, help="write the comparison markdown to this path")
    parsed = parser.parse_args()

    if parsed.compare:
        print(_compare(parsed.compare[0], parsed.compare[1], parsed.out))
        return

    result = run_bench()
    print(_fmt_block(result))
    if parsed.json_out:
        _persist(result, parsed.json_out, parsed.label, parsed.commit, parsed.dirty)


if __name__ == "__main__":
    main()
