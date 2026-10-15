"""torchtitan Step5 under context parallelism must compute what it computes on the whole sequence."""

from tests.ci.ci_register import register_cuda_ci

# the sparse region attention kernel is tuned for, and only run on, Blackwell
register_cuda_ci(est_time=300, suite="stage-c-4-gpu-b200", labels=["torchtitan"], hardware=["blackwell"], num_gpus=2)

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("torchtitan")

_WORKER = Path(__file__).with_name("_step5_torchtitan_cp_worker.py")
_REPO_ROOT = Path(__file__).parents[2]


@pytest.fixture(scope="module")
def errors(tmp_path_factory) -> dict[str, float]:
    output = tmp_path_factory.mktemp("step5-titan-cp") / "errors.json"
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(_REPO_ROOT), env.get("PYTHONPATH")]))
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc-per-node=2", str(_WORKER)]
    result = subprocess.run([*command, "--output", str(output)], env=env, capture_output=True, text=True, timeout=900)
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-8000:]
    assert "PASS" in result.stdout
    return json.loads(output.read_text())


def test_context_parallel_matches_the_whole_sequence(errors):
    print({key: f"{err:.2e}" for key, err in errors.items()})
    # different GEMM row counts and attention spans round differently in BF16, ~1e-3 apart
    assert errors["logits"] < 1e-2, errors
    assert all(err < 2e-2 for err in errors.values()), errors
