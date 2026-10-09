"""CPU performance coverage for the real session server with a mock inference backend."""

import json
import os
import statistics
import sys
from argparse import Namespace
from pathlib import Path

import pytest
from tests.ci.ci_register import register_cpu_ci
from tests.manual.session._benchmark_process import run_benchmark_process
from tests.manual.session.bench_session_server_overhead import (
    DEFAULT_HF_CHECKPOINT,
    benchmark_resources,
    run_http_bench,
)

register_cpu_ci(est_time=600, suite="stage-b-cpu", labels=["rollout"])


def _measure(version):
    resources = benchmark_resources()
    assert resources["effective_cpus"] >= 4, resources
    assert resources["available_memory_bytes"] >= 12_000_000_000, resources
    args = Namespace(
        sessions=32,
        turns=100,
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
        "resources": resources,
        "sessions": args.sessions,
        "turns": args.turns,
        "version": version,
        "median_turns_per_s": statistics.median(r["throughput_turns_per_s"] for r in runs),
        "median_cpu_ms_per_turn": statistics.median(r["server_cpu_s"] / r["completed_turns"] * 1000 for r in runs),
        "median_p99_ms": statistics.median(r["metrics"]["reply_latency_ms"]["p99_ms"] for r in runs),
        "peak_rss_bytes": max(r["peak_rss_bytes"] for r in runs),
        "tree_peak_rss_bytes": max(r["tree_peak_rss_bytes"] for r in runs),
        "median_backend_cpu_s": statistics.median(r["backend_cpu_s"] for r in runs),
        "median_driver_cpu_s": statistics.median(r["driver_parent_cpu_s"] for r in runs),
    }
    return {"summary": report, "runs": runs}


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_session_server_benchmark(version, tmp_path, capsys):
    result_path = tmp_path / f"session-server-{version}.json"
    root = Path(__file__).resolve().parents[4]
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="",
        TOKENIZERS_PARALLELISM="false",
        OMP_NUM_THREADS="1",
    )
    env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
    run_benchmark_process(
        [
            sys.executable,
            "-m",
            "tests.fast.rollout.session.test_session_server_benchmark",
            version,
            str(result_path),
        ],
        cwd=root,
        env=env,
    )
    report = json.loads(result_path.read_text())["summary"]
    with capsys.disabled():
        print(json.dumps(report), flush=True)
    assert report["tree_peak_rss_bytes"] <= 12_000_000_000, report
    assert report["median_cpu_ms_per_turn"] <= 50.0, report


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


@pytest.mark.parametrize("limit", ["memory", "deadline", "orphan"])
def test_benchmark_process_stops_children(tmp_path, limit):
    import psutil

    identities = tmp_path / "processes.json"
    child_code = "import time; " + ("data = bytearray(80_000_000); " if limit == "memory" else "") + "time.sleep(60)"
    command = (
        "import json, subprocess, sys, time, psutil; from pathlib import Path; "
        f"child = subprocess.Popen([sys.executable, '-c', {child_code!r}]); "
        "processes = [psutil.Process(), psutil.Process(child.pid)]; "
        f"Path({str(identities)!r}).write_text(json.dumps([(p.pid,p.create_time()) for p in processes])); "
        + ("" if limit == "orphan" else "time.sleep(60)")
    )
    expected = {"memory": "RSS", "deadline": "deadline", "orphan": "left child processes"}
    with pytest.raises(AssertionError, match=expected[limit]):
        run_benchmark_process(
            [sys.executable, "-c", command],
            cwd=tmp_path,
            env=dict(os.environ),
            timeout=2,
            memory_bytes=50_000_000,
        )
    for pid, created in json.loads(identities.read_text()):
        try:
            process = psutil.Process(pid)
            assert process.create_time() != created or process.status() == psutil.STATUS_ZOMBIE
        except psutil.NoSuchProcess:
            pass


if __name__ == "__main__":
    Path(sys.argv[2]).write_text(json.dumps(_measure(sys.argv[1])))
