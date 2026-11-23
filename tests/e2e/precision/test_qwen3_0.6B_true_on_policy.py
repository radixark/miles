"""Qwen3 dense TOP gate: exact trainer/rollout log-probs over two GRPO steps, and per-layer gradients against HF.

The trainer runs as TOP recipes do (``TOP_RECIPE_TRAIN_ARGS``): rollout log-probs are the old policy, with no
forward-only pass, and full activation recompute.

Forward: ``--ci-test`` with ``--true-on-policy-mode`` fails the run on the first training-forward log-prob unequal to
the rollout's. Step 1 scores the initial weights; step 2 scores weights trained and synced to the engine.
Temperature 0.8 runs the BF16 temperature boundary on both sides.

Backward: rollout 0 captures four decoder layers, and ``miles.utils.debug_utils.top_vjp`` judges them after the run:
each attention and MLP module's backward against HF's at TOP's own module inputs (the contract), and each layer against
HF in FP32, bounded by HF BF16's own distance from it (the quality).

Layout: the trainer runs two pipeline stages of 14 layers. Layers 13 and 14 sit on either side of the stage boundary,
so the verdict also checks that the packed branch/residual pair and its gradient cross it bit for bit. The gate has one
phase, so it has no cross-layout comparison; the verdict's references are layout-free.
"""

import os

from tests.ci.ci_register import register_cuda_ci

from miles.true_on_policy.launch import TOP_RECIPE_TRAIN_ARGS, top_launch
from miles.utils.debug_utils.top_capture import capture_args, fail_on
from miles.utils.debug_utils.top_vjp import vjp_verdict
from miles.utils.external_utils import command_utils

# est_time is an upper bound until a /rerun-test run measures it.
register_cuda_ci(
    est_time=1440,
    suite="stage-c-4-gpu-h200",
    labels=["megatron", "precision"],
    hardware=["hopper", "blackwell"],
    num_gpus=4,
)

MODEL_NAME = "Qwen3-0.6B"
MODEL_TYPE = "qwen3-0.6B"
MODEL_DIR = f"/root/models/{MODEL_NAME}"
NUM_GPUS = 4
DUMP_DIR = "/root/dumps/qwen3_0.6B_true_on_policy"
STAGE_LAYERS = (14, 14)
PIPELINE_STAGES = len(STAGE_LAYERS)
# First and last layer of each stage; every layer runs the same ops.
VJP_LAYERS = (0, 13, 14, 27)


def prepare():
    U = command_utils.default_config().create_backend()
    U.exec_command_cpu("mkdir -p /root/models /root/datasets")
    U.exec_command_cpu(f"hf download Qwen/{MODEL_NAME} --local-dir {MODEL_DIR}")
    U.hf_download_dataset("zhuzilin/dapo-math-17k")
    U.convert_checkpoint(
        model_name=MODEL_NAME, megatron_model_type=MODEL_TYPE, num_gpus_per_node=NUM_GPUS, dir_dst="/root/models"
    )


def execute():
    U = command_utils.default_config().create_backend()
    launch = top_launch(model_type="qwen3", engine_gpus=1)

    ckpt_args = f"--hf-checkpoint {MODEL_DIR}/ --ref-load {MODEL_DIR}_torch_dist "

    rollout_args = (
        "--prompt-data /root/datasets/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type deepscaler "
        "--num-rollout 2 "
        "--rollout-batch-size 4 "
        "--n-samples-per-prompt 2 "
        "--rollout-max-response-len 1024 "
        "--rollout-temperature 0.8 "
        "--global-batch-size 8 "
    )

    # TOP admits Qwen3 dense at trainer TP1/CP1 and any PP, without sequence parallelism or selective recompute.
    perf_args = (
        "--tensor-model-parallel-size 1 "
        f"--pipeline-model-parallel-size {PIPELINE_STAGES} "
        "--context-parallel-size 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 16384 "
    )

    grpo_args = (
        "--advantage-estimator grpo "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type k1 "
        "--kl-coef 0.00 "
        "--entropy-coef 0.00 "
        "--eps-clip 0.2 "
    )

    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
    )

    sglang_args = "--rollout-num-gpus 2 --rollout-num-gpus-per-engine 1 --sglang-mem-fraction-static 0.7 "

    misc_args = (
        "--attention-dropout 0.0 --hidden-dropout 0.0 --actor-num-nodes 1 "
        f"--actor-num-gpus-per-node {PIPELINE_STAGES} "
    )

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{command_utils.get_default_wandb_args(__file__)} "
        f"{perf_args} "
        f"{sglang_args} "
        "--ci-test "
        f"{misc_args} "
        f"{capture_args(dump_dir=DUMP_DIR, layers=VJP_LAYERS, stage_layers=STAGE_LAYERS)} "
        f"{' '.join(TOP_RECIPE_TRAIN_ARGS)} "
        f"{' '.join(launch.train_args)} "
    )

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=NUM_GPUS,
        megatron_model_type=MODEL_TYPE,
        extra_env_vars=launch.env_vars,
    )


def judge_backward() -> dict:
    verdict = vjp_verdict(
        family="Qwen3 dense",
        model_path=MODEL_DIR,
        dump_dir=DUMP_DIR,
        layers=VJP_LAYERS,
        stage_layers=STAGE_LAYERS,
    )
    fail_on(verdict=verdict["failures"])
    return dict(verdict=verdict)


if __name__ == "__main__":
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute()
    judge_backward()
