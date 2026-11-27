"""Qwen GDN dense TOP gate with images: exact log-probs of image-conditioned samples at trainer CP1, all-gather CP2,
PP2 and PP4.

Qwen3.8-27B (the Qwen3.5 dense architecture) cut to 4 layers, with its whole 27-layer vision tower. One 1-GPU engine
serves prompts with drawn images (``write_image_prompts``: one or two images per prompt at eight sizes, and one
text-only prompt) through SGLang's token-space processor. The trainer rebuilds each sample's pixels, placeholder
placement and mRoPE positions with the same processor, encodes the images with the frozen tower on serving's TOP
tower program, and writes the rows over the placeholder embeddings; full attention rotates by the mRoPE positions.

Phase ``cp1`` trains on one GPU; phase ``cp2`` on two under all-gather CP, where an image's rows are split between
the ranks at 64-row grid points. Both run ``--ci-test`` with ``--true-on-policy-mode``, which fails on the first
unequal log-prob over two GRPO steps; the second step scores rollouts of the updated weights. Step 1 of both phases
scores the same rollouts from the same weights, which is checked, and its gradient norm must agree across them.

Phase ``pp2`` trains on two pipeline stages: the first embeds and runs the tower, every stage rotates by the samples'
mRoPE positions, the last scores. Phase ``pp4`` puts one layer on each of four stages, so every layer boundary of the
slice (GDN to GDN, GDN to full attention) is a stage boundary. All four score the same step-1 rollouts and must agree
on the step-1 gradient norm.

Phase ``video`` trains at all-gather CP2 on drawn clips (``write_video_prompts``: one or two clips per prompt, a
drawing and a clip together, and a text-only prompt): each clip's sampled frames form temporal patch groups, each its
own attention sequence in the tower, and its tokens take the model's video positions. It is judged by its log-probs.

The tower is frozen, so nothing judges a tower backward; the text model's backward verdicts are the text gates'.
"""

import math
import os

import torch

from tests.ci.ci_register import register_cuda_ci
from tests.e2e.precision._qwen_gdn_common import (
    DECODE_KV_SPLIT,
    grpo_args,
    slice_checkpoint,
    vision_args,
    write_image_prompts,
    write_video_prompts,
)

from miles.true_on_policy.launch import top_launch
from miles.utils.debug_utils.top_capture import ROLLOUT_DATA, check_same_rollouts, probe_reward
from miles.utils.external_utils import command_utils

# Nightly, as the other TOP gates. est_time is an upper bound until a /rerun-test run measures it.
register_cuda_ci(
    est_time=2700,
    suite="stage-c-8-gpu-h200",
    labels=["megatron", "precision"],
    hardware=["hopper", "blackwell"],
    num_gpus=8,
    nightly=True,
)

HF_REPO = "Qwen/Qwen3.8-27B"
SOURCE = "/root/models/Qwen3.8-27B"
MODEL_DIR = "/root/models/Qwen3.8-27B-4layer"
MODEL_TYPE = "qwen3.8-27B-4layer"
DATA_DIR = "/root/datasets/top_vision_drawings"
VIDEO_DATA_DIR = "/root/datasets/top_vision_clips"
NUM_GPUS = 8
DUMP_DIR = "/root/dumps/qwen3_8_27B_4layer_vision_true_on_policy"
GRAD_NORM_DIR = "/root/grad_norm/qwen3_8_27B_4layer_vision_true_on_policy"
# Each phase's trainer context-parallel and pipeline sizes and its data (dense, DP1).
PHASES = {
    "cp1": dict(cp=1, pp=1, video=False),
    "cp2": dict(cp=2, pp=1, video=False),
    "pp2": dict(cp=1, pp=2, video=False),
    "pp4": dict(cp=1, pp=4, video=False),
    "video": dict(cp=2, pp=1, video=True),
}
# The phases that train on the same rollouts.
COMPARED = ("cp1", "cp2", "pp2", "pp4")
# Step-1 gradient norms of the two phases; they differ only in reduction order.
GRAD_NORM_RTOL = 1e-3


def prepare():
    U = command_utils.default_config().create_backend()
    U.exec_command_cpu("mkdir -p /root/models /root/datasets")
    U.exec_command_cpu(f"hf download {HF_REPO} --local-dir {SOURCE}")
    slice_checkpoint(SOURCE, MODEL_DIR, num_layers=4)
    write_prompts()


def write_prompts():
    write_image_prompts(DATA_DIR)
    write_video_prompts(VIDEO_DATA_DIR)


def execute(phase: str):
    cp, pp, video = (PHASES[phase][key] for key in ("cp", "pp", "video"))
    U = command_utils.default_config().create_backend()
    launch = top_launch(model_type="qwen3_5", engine_gpus=1)
    os.makedirs(f"{GRAD_NORM_DIR}/{phase}", exist_ok=True)

    perf_args = (
        "--tensor-model-parallel-size 1 "
        f"--pipeline-model-parallel-size {pp} "
        f"--context-parallel-size {cp} "
        f"{'--allgather-cp ' if cp > 1 else ''}"
    )
    train_args = (
        f"{grpo_args(model_dir=MODEL_DIR, prompt_data=None)} "
        f"{vision_args(data_dir=VIDEO_DATA_DIR, video=True) if video else vision_args(data_dir=DATA_DIR)} "
        f"{perf_args} "
        "--rollout-num-gpus 1 --rollout-num-gpus-per-engine 1 "
        f"--actor-num-nodes 1 --actor-num-gpus-per-node {cp * pp} "
        f"--ci-save-grad-norm {GRAD_NORM_DIR}/{phase}/{{role}}_{{rollout_id}}_{{step_id}}.pt "
        f"--save-debug-rollout-data {DUMP_DIR}/{phase}/{ROLLOUT_DATA}/{{rollout_id}}.pt "
        # Unclipped gradients under the probe reward, as the capturing gates: step 1 of both phases sees one loss.
        "--clip-grad 0 "
        f"--custom-rm-path {probe_reward.__module__}.{probe_reward.__name__} "
        f"{command_utils.get_default_wandb_args(__file__)} "
        f"{' '.join(launch.train_args)} "
    )

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=NUM_GPUS,
        megatron_model_type=MODEL_TYPE,
        extra_env_vars={**launch.env_vars, "SGLANG_TOP_DECODE_KV_SPLIT": DECODE_KV_SPLIT},
    )


def assert_backward_matches():
    norms = {
        # The last pipeline stage saves its norm on its own device; read every norm on the CPU.
        phase: float(torch.load(f"{GRAD_NORM_DIR}/{phase}/actor_0_0.pt", map_location="cpu", weights_only=False))
        for phase in COMPARED
    }
    print(f"GRAD_NORMS {norms}", flush=True)
    assert norms["cp1"] > 0 and all(
        math.isclose(norm, norms["cp1"], rel_tol=GRAD_NORM_RTOL) for norm in norms.values()
    ), f"step-1 grad norms {norms}"


def judge_backward() -> dict:
    """The compared phases scored the same image rollouts, and their step-1 gradient norms agree."""
    check_same_rollouts({phase: f"{DUMP_DIR}/{phase}" for phase in COMPARED})
    assert_backward_matches()
    return {}


if __name__ == "__main__":
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    for phase in PHASES:
        execute(phase)
    judge_backward()
