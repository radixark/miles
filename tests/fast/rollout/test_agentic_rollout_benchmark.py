"""Opt-in CPU performance coverage of the complete agentic rollout function."""

import json
import os
import sys
from pathlib import Path

import pytest
from tests.ci.ci_register import register_cpu_ci
from tests.manual.session._benchmark_process import MEMORY_BUDGET_BYTES, run_benchmark_process
from tests.manual.session.bench_session_server_overhead import benchmark_resources

register_cpu_ci(est_time=180, suite="stage-b-cpu", labels=["rollout"])


def _run_benchmark(source_root, result_path):
    command = [
        sys.executable,
        "-m",
        "tests.manual.session.bench_agentic_rollout",
        "--json-out",
        str(result_path),
    ]
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="",
        TOKENIZERS_PARALLELISM="false",
        OMP_NUM_THREADS="1",
    )
    env["PYTHONPATH"] = str(source_root) + os.pathsep + env.get("PYTHONPATH", "")
    run_benchmark_process(command, cwd=source_root, env=env)
    return json.loads(result_path.read_text())


@pytest.fixture(scope="module")
def rollout_benchmark_result(tmp_path_factory):
    resources = benchmark_resources()
    assert resources["effective_cpus"] >= 4, resources
    assert resources["available_memory_bytes"] >= MEMORY_BUDGET_BYTES, resources
    result_path = tmp_path_factory.mktemp("rollout-benchmark") / "result.json"
    return _run_benchmark(Path(__file__).resolve().parents[3], result_path)


def test_agentic_rollout_benchmark(rollout_benchmark_result, capsys):
    result = rollout_benchmark_result
    assert len(result["steps"]) == 4
    assert result["steps"][0]["warmup"]
    for step in result["steps"]:
        assert step["samples"] == result["config"]["sessions"]
        assert step["turns"] == result["config"]["sessions"] * result["config"]["turns"]
        assert step["r3_bytes"] >= 1_000_000_000
        assert step["tree_peak_rss_bytes"] <= MEMORY_BUDGET_BYTES
        assert step["wall_s"] > 0
        assert step["manager_cpu_s"] > 0
    with capsys.disabled():
        print("ROLLOUT_BENCHMARK " + json.dumps(result, sort_keys=True))
    # Hosted four-vCPU runs measured about 5.5 s/batch; allow runner variation.
    assert result["median_manager_cpu_s"] <= 8.0, result
