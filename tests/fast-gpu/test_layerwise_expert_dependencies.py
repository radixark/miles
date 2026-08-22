from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=90, suite="stage-b-2-gpu-h200", labels=["lora"])

import os
import subprocess
import sys
from pathlib import Path


def test_grouped_expert_lora_layerwise_norm_and_clip() -> None:
    worker = Path(__file__).with_name("_layerwise_expert_dependency_worker.py")
    repo_root = Path(__file__).parents[2]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(repo_root), env.get("PYTHONPATH")]))
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=2",
            str(worker),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )

    assert result.returncode == 0, result.stdout + result.stderr
