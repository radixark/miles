"""Qwen GDN MoE TOP gate with images: exact log-probs of image-conditioned samples from an attention-DP engine, at
trainer CP1 and all-gather CP4, and with the engine speculating.

Qwen3.5-35B-A3B cut to 4 layers (three GDN layers, one full-attention layer, 256 routed experts with a gated shared
expert), with its whole 27-layer vision tower. One 4-GPU engine serves with attention DP4 and EP4 over DeepEP: each
DP rank encodes the images of its own requests with an unsharded tower. The prompts are drawn images
(``write_image_prompts``) through SGLang's token-space processor, as in the dense image gate
(``test_qwen3_8_27B_4layer_vision_true_on_policy``), which describes the trainer side.

Phase ``cp1`` trains at CP1 x EP4 (DP4), phase ``cp4`` at CP4 x EP4 under all-gather CP, where an image's rows are
split between ranks at 64-row grid points. Both run ``--ci-test`` with ``--true-on-policy-mode``, which fails on the
first unequal log-prob over two GRPO steps. Step 1 of both scores the same rollouts from the same weights, which is
checked, and its gradient norm must agree across them. Phase ``spec`` trains at CP1 with the engine speculating
through the checkpoint's MTP layer (the chain recipe (3, 1, 4)): the tower runs at prefill only, and verify makes
every emitted token draft-independent, so the log-probs must stay exact. The slice's draft is almost never accepted,
so this phase checks verify on image requests, not accepted drafts. Phase ``video`` trains at CP4 x EP4 on drawn
clips (``write_video_prompts``), judged by its log-probs. Phase ``pp2`` trains on two pipeline stages at EP2 (the
experts of a stage's layers across the stage's two data-parallel ranks); it scores the same rollouts as ``cp1`` and
``cp4`` and must agree on the step-1 gradient norm.
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
    est_time=3600,
    suite="stage-c-8-gpu-h200",
    labels=["megatron", "precision"],
    hardware=["hopper", "blackwell"],
    num_gpus=8,
    nightly=True,
)

HF_REPO = "Qwen/Qwen3.5-35B-A3B"
SOURCE = "/root/models/Qwen3.5-35B-A3B"
MODEL_DIR = "/root/models/Qwen3.5-35B-A3B-4layer"
# The slice with its MTP head, which the ``spec`` phase's engine drafts with.
MTP_MODEL_DIR = "/root/models/Qwen3.5-35B-A3B-4layer-mtp"
# The 4-layer args without --moe-router-dtype, which TOP refuses.
MODEL_TYPE = "qwen3.5-35B-A3B-4layer-top"
DATA_DIR = "/root/datasets/top_vision_drawings"
VIDEO_DATA_DIR = "/root/datasets/top_vision_clips"
NUM_GPUS = 8
TRAINER_GPUS = ENGINE_GPUS = 4
DUMP_DIR = "/root/dumps/qwen3_5_35B_A3B_4layer_vision_true_on_policy"
GRAD_NORM_DIR = "/root/grad_norm/qwen3_5_35B_A3B_4layer_vision_true_on_policy"
# Each phase's trainer context-parallel and pipeline sizes on the 4 trainer GPUs; expert parallelism spans the
# remaining data-parallel and context-parallel ranks of a stage (EP4 at PP1, EP2 at PP2).
PHASES = {
    "cp1": dict(cp=1, pp=1),
    "cp4": dict(cp=4, pp=1),
    "spec": dict(cp=1, pp=1),
    "video": dict(cp=4, pp=1),
    "pp2": dict(cp=1, pp=2),
}
# The phases that train on the same rollouts.
COMPARED = ("cp1", "cp4", "pp2")
# Step-1 gradient norms of the compared phases; they differ only in reduction order.
GRAD_NORM_RTOL = 1e-3
# The engine's chain speculation in phase ``spec``: (steps, top-k, draft tokens), the cookbook MTP recipe.
SPEC_RECIPE = (3, 1, 4)


def prepare():
    U = command_utils.default_config().create_backend()
    U.exec_command_cpu("mkdir -p /root/models /root/datasets")
    U.exec_command_cpu(f"hf download {HF_REPO} --local-dir {SOURCE}")
    slice_checkpoint(SOURCE, MODEL_DIR, num_layers=4)
    slice_checkpoint(SOURCE, MTP_MODEL_DIR, num_layers=4, keep_mtp=True)
    write_prompts()


def write_prompts():
    write_image_prompts(DATA_DIR)
    write_video_prompts(VIDEO_DATA_DIR)


def execute(phase: str):
    cp, pp = PHASES[phase]["cp"], PHASES[phase]["pp"]
    U = command_utils.default_config().create_backend()
    launch = top_launch(model_type="qwen3_5_moe", engine_gpus=ENGINE_GPUS)
    os.makedirs(f"{GRAD_NORM_DIR}/{phase}", exist_ok=True)

    perf_args = (
        "--tensor-model-parallel-size 1 "
        f"--pipeline-model-parallel-size {pp} "
        f"--context-parallel-size {cp} "
        f"{'--allgather-cp ' if cp > 1 else ''}"
        f"--expert-model-parallel-size {TRAINER_GPUS // pp} "
        "--expert-tensor-parallel-size 1 "
    )
    steps, topk, draft_tokens = SPEC_RECIPE
    spec_args = (
        "--sglang-speculative-algorithm NEXTN "
        f"--sglang-speculative-num-steps {steps} "
        f"--sglang-speculative-eagle-topk {topk} "
        f"--sglang-speculative-num-draft-tokens {draft_tokens} "
        # The draft worker allocates its own KV pool beside the target's: leave it room (grpo_args asks for 0.7).
        "--sglang-mem-fraction-static 0.4 "
        if phase == "spec"
        else ""
    )
    train_args = (
        f"{grpo_args(model_dir=MTP_MODEL_DIR if phase == 'spec' else MODEL_DIR, prompt_data=None)} "
        f"{vision_args(data_dir=VIDEO_DATA_DIR, video=True) if phase == 'video' else vision_args(data_dir=DATA_DIR)} "
        f"{perf_args} "
        f"--rollout-num-gpus {ENGINE_GPUS} --rollout-num-gpus-per-engine {ENGINE_GPUS} "
        # DeepEP's low-latency decode dispatches at most 128 tokens per rank; capture decode graphs below that
        # (a verify step scores 4 rows a request).
        f"--sglang-cuda-graph-max-bs-decode {16 if phase == 'spec' else 32} "
        f"{spec_args}"
        f"--actor-num-nodes 1 --actor-num-gpus-per-node {TRAINER_GPUS} "
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
