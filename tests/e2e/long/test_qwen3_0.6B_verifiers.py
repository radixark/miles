import math
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

import pytest
import torch
from tests.ci.ci_register import register_cuda_ci

import miles.utils.external_utils.command_utils as U

register_cuda_ci(est_time=900, suite="stage-c-2-gpu-h200", labels=["long"])

MODEL_NAME = "Qwen3-0.6B"
MODEL_TYPE = "qwen3-0.6B"
NUM_GPUS = 2
MODEL_DIR = Path(os.environ.get("MILES_E2E_MODEL_DIR", "/root/models"))
MEGATRON_PATH = Path(os.environ.get("MILES_E2E_MEGATRON_PATH", "/root/Megatron-LM"))
RUN_DIR = Path(os.environ.get("MILES_E2E_RUN_DIR", "/tmp/miles-verifiers-e2e"))
VERIFIERS_DIR = Path("/tmp/verifiers-v0.2.0")


def prepare():
    U.exec_command(f"mkdir -p {MODEL_DIR} {RUN_DIR}")
    U.exec_command(f"hf download Qwen/{MODEL_NAME} --local-dir {MODEL_DIR}/{MODEL_NAME}")
    U.exec_command(f"{sys.executable} -m pip install -e '{U.repo_base_dir}[verifiers]'")
    U.exec_command("uv tool install 'prime==0.6.19'")
    if not VERIFIERS_DIR.exists():
        U.exec_command(
            f"git clone --depth 1 --branch v0.2.0 "
            f"https://github.com/PrimeIntellect-ai/verifiers.git {VERIFIERS_DIR}"
        )
    shutil.copytree(
        VERIFIERS_DIR / "environments" / "code_golf_v1",
        RUN_DIR / "environments" / "code_golf_v1",
        dirs_exist_ok=True,
    )
    U.exec_command(f"cd {RUN_DIR} && prime --plain env install code-golf-v1")
    U.convert_checkpoint(
        model_name=MODEL_NAME,
        megatron_model_type=MODEL_TYPE,
        num_gpus_per_node=NUM_GPUS,
        dir_dst=str(MODEL_DIR),
        hf_checkpoint=str(MODEL_DIR / MODEL_NAME),
        megatron_path=str(MEGATRON_PATH),
    )


def execute():
    config_path = RUN_DIR / "code-golf.toml"
    config_path.write_text('[taskset]\nid = "code-golf-v1"\n')
    dump_dir = RUN_DIR / "dump"

    train_args = " ".join(
        [
            f"--hf-checkpoint {MODEL_DIR}/{MODEL_NAME}",
            "--sglang-tokenizer-path Qwen/Qwen3-0.6B",
            f"--ref-load {MODEL_DIR}/{MODEL_NAME}_torch_dist",
            f"--verifiers-config {config_path}",
            "--num-rollout 1",
            "--rollout-batch-size 3",
            "--n-samples-per-prompt 4",
            "--over-sampling-batch-size 3",
            "--rollout-max-response-len 512",
            "--rollout-max-context-len 2048",
            "--rollout-temperature 0.8",
            "--global-batch-size 12",
            "--balance-data",
            "--advantage-estimator grpo",
            "--entropy-coef 0.0",
            "--eps-clip 0.2",
            "--eps-clip-high 0.28",
            "--optimizer adam",
            "--lr 1e-6",
            "--lr-decay-style constant",
            "--weight-decay 0.1",
            "--adam-beta1 0.9",
            "--adam-beta2 0.98",
            "--no-gradient-accumulation-fusion",
            "--rollout-num-gpus-per-engine 1",
            "--sglang-mem-fraction-static 0.6",
            "--sglang-enable-metrics",
            "--tensor-model-parallel-size 1",
            "--pipeline-model-parallel-size 1",
            "--context-parallel-size 1",
            "--use-dynamic-batch-size",
            "--max-tokens-per-gpu 4096",
            "--actor-num-nodes 1",
            f"--actor-num-gpus-per-node {NUM_GPUS}",
            "--colocate",
            f"--dump-details {dump_dir}",
            U.get_default_wandb_args(__file__),
        ]
    )

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=NUM_GPUS,
        megatron_model_type=MODEL_TYPE,
        extra_env_vars={"MILES_EXPERIMENTAL_ROLLOUT_REFACTOR": "0"},
        megatron_path=str(MEGATRON_PATH),
    )
    verify(dump_dir)


def verify(dump_dir: Path):
    samples = torch.load(dump_dir / "rollout_data" / "0.pt", weights_only=False)["samples"]

    assert len(samples) == 12
    assert Counter(sample["group_index"] for sample in samples) == {0: 4, 1: 4, 2: 4}
    assert {sample["status"] for sample in samples} <= {"completed", "truncated"}
    assert any(sample["status"] == "completed" for sample in samples)
    assert all(math.isfinite(sample["reward"]) for sample in samples)
    assert all(
        len(sample["rollout_log_probs"]) == len(sample["loss_mask"]) == sample["response_length"] for sample in samples
    )
    assert all(math.isfinite(value) for sample in samples for value in sample["rollout_log_probs"])

    for sample in samples:
        metadata = sample["metadata"]["verifiers"]
        assert set(metadata["rewards"]) == {"correct", "fastest", "most_concise"}
        assert metadata["metrics"]["passed"] in {0.0, 1.0}
        assert math.isfinite(metadata["metrics"]["latency"])
        assert "error" not in metadata

    assert any(sample["metadata"]["verifiers"]["metrics"]["passed"] for sample in samples)
    for group_index in range(3):
        group = [sample for sample in samples if sample["group_index"] == group_index]
        assert sum(sample["metadata"]["verifiers"]["rewards"]["fastest"] for sample in group) == pytest.approx(0.5)


if __name__ == "__main__":
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute()
