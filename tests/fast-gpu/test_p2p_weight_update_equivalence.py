"""A p2p weight update must leave a rollout engine identical to sglang's own update from the same HF tensors."""

from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=300, suite="stage-c-4-gpu-b200", num_gpus=1, labels=["weight-update"], hardware=["blackwell"]
)

import os
import subprocess
import sys
from pathlib import Path

import pytest
from huggingface_hub import snapshot_download

_WORKER = Path(__file__).with_name("_p2p_weight_update_equivalence_worker.py")
_REPO_ROOT = Path(__file__).parents[2]


@pytest.fixture(scope="module")
def glm52_config_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    config_dir = tmp_path_factory.mktemp("glm52_config")
    snapshot_download("zai-org/GLM-5.2", allow_patterns=["config.json"], local_dir=config_dir)
    return config_dir


# each case in its own process: sglang keeps its server args and DeepGEMM choice in module globals
@pytest.mark.parametrize(
    "fmt, extra_env, moe_runner_backend",
    [
        # unquantized MoE on flashinfer TRT-LLM, Blackwell's default, which permutes experts in postprocess
        ("bf16", {}, "flashinfer_trtllm"),
        # DeepGEMM requantizes, so miles sends UE8M0 scales
        ("fp8_block", {"SGLANG_ENABLE_JIT_DEEPGEMM": "1"}, "auto"),
        ("fp8_block", {"SGLANG_ENABLE_JIT_DEEPGEMM": "0"}, "auto"),
        ("mxfp8", {}, "auto"),
        ("nvfp4", {}, "auto"),
    ],
    ids=["bf16", "fp8_block_ue8m0_scales", "fp8_block_fp32_scales", "mxfp8", "nvfp4"],
)
def test_a_p2p_update_leaves_the_engine_identical_to_sglangs_own(
    fmt: str, extra_env: dict[str, str], moe_runner_backend: str, glm52_config_dir: Path, tmp_path: Path
) -> None:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(_REPO_ROOT), env.get("PYTHONPATH")]))
    # miles' launcher always sets it, and its fp8 quantizer reads it
    env["NVTE_FP8_BLOCK_SCALING_FP32_SCALES"] = "1"
    env.update(extra_env)
    result = subprocess.run(
        [
            sys.executable,
            str(_WORKER),
            "--config-dir",
            str(glm52_config_dir),
            "--model-dir",
            str(tmp_path / "model"),
            "--fmt",
            fmt,
            "--moe-runner-backend",
            moe_runner_backend,
            # rank 1 so every TP shard offset is non-zero
            "--rank",
            "1",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS" in result.stdout


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
