"""FSDP per-token loss scaling restores the global token-mean gradient."""

from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, suite="stage-b-2-gpu-h200", labels=["fsdp"], hardware=["hopper", "blackwell"])

import os
import subprocess
import sys
from pathlib import Path

import pytest

_WORKER = Path(__file__).with_name("_fsdp_per_token_loss_worker.py")
_REPO_ROOT = Path(__file__).parents[2]


def test_fsdp_per_token_loss_scaling_lands_on_global_token_mean() -> None:
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
            "--nproc-per-node=2",
            str(_WORKER),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS fsdp-per-token-loss" in result.stdout


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
