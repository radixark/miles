"""CPU performance coverage for the real session server with a mock inference backend."""

import json
import statistics
from argparse import Namespace

import pytest
from tests.ci.ci_register import register_cpu_ci
from tests.manual.session.bench_session_server_overhead import DEFAULT_HF_CHECKPOINT, run_http_bench

register_cpu_ci(est_time=120, suite="stage-b-cpu", labels=["rollout"])


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_session_server_benchmark(version, tmp_path, capsys):
    args = Namespace(
        sessions=8,
        turns=20,
        input_tokens=600,
        output_tokens=400,
        r3_scale=256,
        incremental_r3=True,
        append_role="tool",
        hf_checkpoint=DEFAULT_HF_CHECKPOINT,
        tito_model="qwen3",
        chat_template_path=None,
        use_session_server=version,
        session_server_instances=1,
        mock_backend_procs=1,
        bench_driver_procs=1,
        get_records=False,
        inference_interval=0.0,
        tool_interval=0.0,
        include_raw_samples=False,
    )
    runs = []
    for _ in range(3):
        result = run_http_bench(args, warmup_sessions=2)
        assert result["sessions_ok"] == args.sessions, result["first_errors"]
        assert result["completed_turns"] == args.sessions * args.turns, result["first_errors"]
        assert result["chat_server_errors"] == result["chat_transport_errors"] == 0, result["first_errors"]
        runs.append(result)

    report = {
        "benchmark": "session_server",
        "version": version,
        "median_turns_per_s": statistics.median(r["throughput_turns_per_s"] for r in runs),
        "median_cpu_ms_per_turn": statistics.median(r["server_cpu_s"] / r["completed_turns"] * 1000 for r in runs),
        "median_p99_ms": statistics.median(r["metrics"]["reply_latency_ms"]["p99_ms"] for r in runs),
        "peak_rss_bytes": max(r["peak_rss_bytes"] for r in runs),
    }
    (tmp_path / f"session-server-{version}.json").write_text(json.dumps({"summary": report, "runs": runs}))
    with capsys.disabled():
        print(json.dumps(report), flush=True)
    # Hosted CPU baseline is about 8 ms/turn; allow 50% headroom for runner variation.
    assert report["median_cpu_ms_per_turn"] <= 12.0, report


@pytest.mark.parametrize("failure", ["start", "ready"])
def test_mock_backend_startup_failure_cleans_up(monkeypatch, failure):
    from tests.manual.session import bench_session_server_overhead as benchmark

    from miles.utils import http_utils

    created, started, terminated = [], [], []

    class Process:
        def __init__(self, **kwargs):
            created.append(self)

        def start(self):
            if failure == "start" and len(created) == 2:
                raise RuntimeError("injected startup failure")
            started.append(self)

    def fail_readiness(*args, **kwargs):
        raise RuntimeError("injected startup failure")

    monkeypatch.setattr(benchmark.multiprocessing, "get_context", lambda *_: Namespace(Process=Process))
    monkeypatch.setattr(http_utils, "find_available_port", lambda *_: 28000)
    monkeypatch.setattr(http_utils, "wait_for_server_ready", fail_readiness)
    monkeypatch.setattr(benchmark, "_terminate_proc", terminated.append)
    with pytest.raises(RuntimeError, match="injected startup failure"):
        benchmark._start_backend([], "127.0.0.1", procs=3)
    assert len(started) == (1 if failure == "start" else 3)
    assert terminated == started
