"""FSDP2 gradient parity across r1s4, r2s2, and r4s1."""

from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=90, suite="stage-c-4-gpu-h200", labels=["fsdp"])

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

_WORKER = Path(__file__).with_name("_fsdp_hybrid_shard_worker.py")
_REPO_ROOT = Path(__file__).parents[2]


def _run_worker(replicate_size: int, output: Path) -> None:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(_REPO_ROOT), env.get("PYTHONPATH")]))
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes=1",
            "--nproc-per-node=4",
            str(_WORKER),
            "--replicate-size",
            str(replicate_size),
            "--output",
            str(output),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"PASS r{replicate_size}s{4 // replicate_size}" in result.stdout


def test_fsdp_hybrid_shard_gradient_parity(tmp_path: Path) -> None:
    topology_gradients = {}
    for replicate_size in (1, 2, 4):
        output = tmp_path / f"grad_r{replicate_size}.pt"
        _run_worker(replicate_size, output)
        topology_gradients[replicate_size] = torch.load(output, weights_only=True)

    reference = topology_gradients[1]
    for replicate_size in (2, 4):
        actual = topology_gradients[replicate_size]
        assert actual.keys() == reference.keys()
        for name, expected_gradient in reference.items():
            torch.testing.assert_close(actual[name], expected_gradient, rtol=2e-5, atol=2e-6)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
