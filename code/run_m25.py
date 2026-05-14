#!/usr/bin/env python3
"""Ensure this repo's `miles` package wins over site-packages or /root/miles copies."""
import os
import sys
from pathlib import Path
_MILES_REPO = "/home/yangchengyi/data/miles"
import miles

print(f"{miles.__file__=}")
import datetime
import inspect
import subprocess
from contextlib import contextmanager

import miles.utils.external_utils.command_utils as U

DATA_ROOT = Path(os.environ.get("MILES_DATA_ROOT", "/home/yangchengyi/data"))
MILES_ROOT = DATA_ROOT / "miles"
MODEL_ROOT = DATA_ROOT / "models"
CKPT_ROOT = DATA_ROOT / "ckpts"
DATASET_ROOT = DATA_ROOT / "datasets"
TB_ROOT = DATA_ROOT / "tb"
LOG_ROOT = DATA_ROOT / "logs"

MODEL_NAME = "MiniMax-M2.5"
MODEL_TYPE = "minimax-m2.5"
NUM_GPUS = 8
NUM_NODES = 8
INTERVAL = 50
PROJECT_NAME = "minmax-dev"
VARIANT = f"{MODEL_NAME}_minmax_moe_test"


def ensure_required_dirs():
    required_dirs = [MILES_ROOT, MODEL_ROOT, CKPT_ROOT, DATASET_ROOT, TB_ROOT, LOG_ROOT]
    for path in required_dirs:
        path.mkdir(parents=True, exist_ok=True)


@contextmanager
def tee_to_log(log_path: Path):
    # Redirect process stdout/stderr to `tee`, so subprocess output is also mirrored to file.
    saved_stdout = os.dup(1)
    saved_stderr = os.dup(2)
    tee_proc = subprocess.Popen(
        ["tee", "-a", str(log_path)],
        stdin=subprocess.PIPE,
        stdout=saved_stdout,
        stderr=saved_stderr,
    )
    try:
        assert tee_proc.stdin is not None
        os.dup2(tee_proc.stdin.fileno(), 1)
        os.dup2(tee_proc.stdin.fileno(), 2)
        yield
    finally:
        os.dup2(saved_stdout, 1)
        os.dup2(saved_stderr, 2)
        os.close(saved_stdout)
        os.close(saved_stderr)
        if tee_proc.stdin is not None:
            tee_proc.stdin.close()
        tee_proc.wait()


def prepare():
    # Only convert when torch_dist is missing.
    torch_dist_dir = MODEL_ROOT / f"{MODEL_NAME}_torch_dist"
    tracker = torch_dist_dir / "latest_checkpointed_iteration.txt"
    if torch_dist_dir.exists() and tracker.exists() and tracker.read_text().strip() == "release":
        print(f"Skip checkpoint conversion, found existing torch_dist: {torch_dist_dir}")
        return
    print(f"Convert checkpoint from {MODEL_ROOT / MODEL_NAME} to {torch_dist_dir}")
    U.convert_checkpoint(
        model_name=MODEL_NAME,
        megatron_model_type=MODEL_TYPE,
        num_gpus_per_node=NUM_GPUS,
        hf_checkpoint=str(MODEL_ROOT / MODEL_NAME),
        dir_dst=str(MODEL_ROOT),
    )


def execute():
    model_dir = MODEL_ROOT
    ckpt_dir = CKPT_ROOT
    data_dir = DATASET_ROOT
    eval_path = data_dir / "qwen-cot"
    eval_files = sorted(eval_path.glob("*.jsonl"))
    if not eval_files:
        raise FileNotFoundError(f"No eval jsonl files found under {eval_path}")

    eval_prompt_data_args = " ".join(
        f"{eval_file.stem} {eval_file}" for eval_file in eval_files
    )

    ckpt_args = (
        f"--hf-checkpoint {model_dir}/{MODEL_NAME} "
        f"--ref-load {model_dir}/{MODEL_NAME}_torch_dist "
        f"--load {ckpt_dir}/{VARIANT} "
        f"--save {ckpt_dir}/{VARIANT} "
        f"--save-interval {INTERVAL} "
    )

    rollout_args = (
        f"--prompt-data {data_dir}/deepmath-103k_miles.jsonl "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        # "--apply-chat-template-kwargs '{\"enable_thinking\":false}' "
        "--rollout-shuffle "
        "--rm-type deepmath "
        "--reward-key score "
        "--num-rollout 3000 "
        "--rollout-batch-size 32 "
        "--n-samples-per-prompt 8 "
        "--rollout-max-response-len 4096 "
        "--rollout-temperature 1 "
        "--global-batch-size 256 "
        "--balance-data "
    )

    eval_args = (
        f"--eval-interval {INTERVAL} "
        "--n-samples-per-eval-prompt 4 "
        "--eval-max-response-len 8192 "
        "--eval-top-p 1 "
        f"--eval-prompt-data {eval_prompt_data_args} "
    )

    perf_args = (
        "--tensor-model-parallel-size 2 "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 4 "
        "--context-parallel-size 2 "
        "--expert-model-parallel-size 8 "
        "--expert-tensor-parallel-size 1 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 4096 "
    )

    grpo_args = (
        "--advantage-estimator gspo "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type low_var_kl "
        "--kl-coef 0.00 "
        "--entropy-coef 0.00 "
        "--eps-clip 4e-4 "
    )

    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
        "--optimizer-cpu-offload "
        "--overlap-cpu-optimizer-d2h-h2d "
        "--use-precision-aware-optimizer "
    )

    wandb_key = os.environ.get("WANDB_API_KEY", "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")
    wandb_args = (
        "--use-wandb "
        f"--wandb-project {PROJECT_NAME} "
        f"--wandb-group {VARIANT} "
        f"{f'--wandb-key {wandb_key} ' if wandb_key else ''}"
    )

    tb_args = (
        "--use-tensorboard "
        f"--tb-dir {TB_ROOT} "
        f"--tb-project-name {PROJECT_NAME} "
        f"--tb-experiment-name {VARIANT} "
    )

    # SGLang: minimum deployable engine is 4xH100 with tp_size=4 -> rollout-num-gpus-per-engine 4.
    sglang_args = (
        "--rollout-num-gpus-per-engine 4 "
        "--sglang-mem-fraction-static 0.60 "
        "--sglang-ep-size 4 "
        "--sglang-cuda-graph-bs 1 2 4 8 16 32 48 64 "
        "--sglang-max-running-requests 256 "
        "--sglang-chunked-prefill-size 8192 "
        "--sglang-server-concurrency 256 "
    )

    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        "--moe-token-dispatcher-type flex "
        "--moe-enable-deepep "
        f"--actor-num-nodes {NUM_NODES} "
        "--actor-num-gpus-per-node 8 "
        "--colocate "
    )

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{wandb_args} "
        f"{tb_args} "
        f"{perf_args} "
        f"{eval_args} "
        f"{sglang_args} "
        f"{misc_args} "
    )

    # Keep shell behavior: external Ray + submit to existing dashboard by default.
    os.environ.setdefault("MILES_SCRIPT_EXTERNAL_RAY", "true")
    master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("RAY_ADDRESS", f"http://{master_addr}:8265")

    # Older miles installs may not have working_dir / skip_cleanup on execute_train.
    execute_train_kw: dict = {
        "train_args": train_args,
        "num_gpus_per_node": NUM_GPUS,
        "megatron_model_type": MODEL_TYPE,
    }
    _sig = inspect.signature(U.execute_train)
    if "working_dir" in _sig.parameters:
        execute_train_kw["working_dir"] = str(MILES_ROOT)
    if "skip_cleanup" in _sig.parameters:
        execute_train_kw["skip_cleanup"] = True
    U.execute_train(**execute_train_kw)


if __name__ == "__main__":
    ensure_required_dirs()
    tz_utc8 = datetime.timezone(datetime.timedelta(hours=8))
    log_path = LOG_ROOT / f"{MODEL_NAME}_dapo_miles_{datetime.datetime.now(tz_utc8).strftime('%Y%m%d_%H%M%S')}.log"
    with tee_to_log(log_path):
        print(f"Logging to: {log_path}")
        for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
            os.environ.pop(proxy_var, None)
        prepare()
        execute()
