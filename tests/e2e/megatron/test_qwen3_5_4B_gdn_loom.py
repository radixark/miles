"""E2E: Qwen3.5-4B full-parameter GRPO with the deterministic GDN backend under head-sharded TP.

Runs ``--linear-attention-backend loom`` (the generated deterministic chunked GDN kernels in
``miles_plugins/models/gdn_chunk_train``) with ``--tensor-model-parallel-size 2 --sequence-parallel``,
so the unified GDN core shards the linear-attention heads across TP instead of replicating the GDN
compute, and the Megatron -> HF weight update goes through the head-interleaved ``in_proj_qkv`` /
``conv1d`` converters.  ``MILES_GDN_BACKEND=fla`` runs the same recipe on the FLA backend for an
A/B reference.  Needs Blackwell (SM100a/SM103a) GPUs for the loom backend.

Paths default to the CI layout (``/root/models``, ``/root/datasets``, ``/root``); set
``MILES_E2E_ROOT`` to relocate all three under one directory (out-of-CI runs on scratch storage).
"""

import os

from tests.ci.ci_register import register_cuda_ci

import miles.utils.external_utils.command_utils as U

register_cuda_ci(
    est_time=1800,
    suite="stage-c-4-gpu-b200",
    labels=["megatron"],
    hardware=["blackwell"],
)

MODEL_NAME = "Qwen3.5-4B"
MODEL_TYPE = "qwen3.5-4B"
NUM_GPUS = 4
BACKEND = os.environ.get("MILES_GDN_BACKEND", "loom")
ROOT = os.environ.get("MILES_E2E_ROOT")
MODEL_DIR = f"{ROOT}/models" if ROOT else "/root/models"
DATA_DIR = f"{ROOT}/datasets" if ROOT else "/root/datasets"
CKPT_DIR = ROOT if ROOT else "/root"
MEGATRON_PATH = os.environ.get("MILES_E2E_MEGATRON_PATH", "/root/Megatron-LM")
NUM_ROLLOUT = int(os.environ.get("MILES_E2E_NUM_ROLLOUT", "3"))


def prepare():
    U.exec_command_cpu(f"mkdir -p {MODEL_DIR} {DATA_DIR}")
    U.exec_command_cpu(f"hf download Qwen/{MODEL_NAME} --local-dir {MODEL_DIR}/{MODEL_NAME}")
    U.hf_download_dataset("zhuzilin/dapo-math-17k", data_dir=DATA_DIR)
    U.hf_download_dataset("zhuzilin/aime-2024", data_dir=DATA_DIR)
    U.convert_checkpoint(
        model_name=MODEL_NAME,
        megatron_model_type=MODEL_TYPE,
        num_gpus_per_node=NUM_GPUS,
        dir_dst=CKPT_DIR,
        hf_checkpoint=f"{MODEL_DIR}/{MODEL_NAME}",
        megatron_path=MEGATRON_PATH,
    )


def execute():
    ckpt_args = f"--hf-checkpoint {MODEL_DIR}/{MODEL_NAME}/ " f"--ref-load {CKPT_DIR}/{MODEL_NAME}_torch_dist "

    rollout_args = (
        f"--prompt-data {DATA_DIR}/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type deepscaler "
        f"--num-rollout {NUM_ROLLOUT} "
        "--rollout-batch-size 8 "
        "--n-samples-per-prompt 8 "
        "--rollout-max-response-len 4096 "
        "--rollout-temperature 0.8 "
        "--global-batch-size 32 "
        "--balance-data "
    )

    eval_args = (
        f"--eval-prompt-data aime24 {DATA_DIR}/aime-2024/aime-2024.jsonl "
        "--n-samples-per-eval-prompt 1 "
        "--eval-max-response-len 16384 "
        "--eval-top-k 1 "
    )

    perf_args = (
        "--tensor-model-parallel-size 2 "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 16384 "
    )

    grpo_args = (
        "--advantage-estimator grpo "
        "--use-kl-loss "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type low_var_kl "
        "--entropy-coef 0.00 "
        "--eps-clip 0.2 "
        "--eps-clip-high 0.28 "
    )

    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
    )

    sglang_args = (
        "--rollout-num-gpus-per-engine 2 "
        "--sglang-mem-fraction-static 0.7 "
        "--sglang-max-running-requests 256 "
    )

    ci_args = "--ci-test "

    misc_args = (
        # default dropout in megatron is 0.1
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        # should be good for model performance
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        "--actor-num-nodes 1 "
        f"--actor-num-gpus-per-node {NUM_GPUS} "
        "--colocate "
        f"--linear-attention-backend {BACKEND} "
    )

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{U.get_default_wandb_args(__file__, run_name_prefix=f'gdn-{BACKEND}')} "
        f"{perf_args} "
        f"{eval_args} "
        f"{sglang_args} "
        f"{ci_args} "
        f"{misc_args} "
    )

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=NUM_GPUS,
        megatron_model_type=MODEL_TYPE,
        megatron_path=MEGATRON_PATH,
    )


if __name__ == "__main__":
    stage = os.environ.get("MILES_E2E_STAGE", "all")
    if stage in ("prepare", "all"):
        prepare()
    if stage in ("train", "all"):
        for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
            os.environ.pop(proxy_var, None)
        execute()
