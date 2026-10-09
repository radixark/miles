"""Opt-in CPU performance coverage of the complete agentic rollout function."""

import json
import os
import signal
import subprocess
import sys
from pathlib import Path

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=180, suite="stage-b-cpu", labels=["rollout"])


def test_agentic_rollout_benchmark(tmp_path, capsys):
    result_path = tmp_path / "rollout-benchmark.json"
    command = [
        sys.executable,
        "-m",
        "tests.manual.session.bench_agentic_rollout",
        "--json-out",
        str(result_path),
    ]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="1")
    # Isolate production singletons and kill the entire owned process group on timeout.
    with subprocess.Popen(
        command,
        cwd=Path(__file__).resolve().parents[3],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    ) as process:
        try:
            output, _ = process.communicate(timeout=180)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                output, _ = process.communicate(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                output, _ = process.communicate()
            raise AssertionError(f"Rollout benchmark timed out:\n{output}") from None
        assert process.returncode == 0, output
    result = json.loads(result_path.read_text())
    assert len(result["steps"]) == 4
    assert result["steps"][0]["warmup"]
    for step in result["steps"]:
        assert step["samples"] == 8
        assert step["turns"] == 64
        assert step["r3_bytes"] > 0
        assert step["wall_s"] > 0
        assert step["manager_cpu_s"] > 0
    with capsys.disabled():
        print("ROLLOUT_BENCHMARK " + json.dumps(result, sort_keys=True))
