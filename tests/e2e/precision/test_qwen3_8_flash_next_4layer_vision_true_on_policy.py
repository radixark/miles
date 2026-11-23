"""Qwen3.8-Flash-Next TOP gate with images: exact log-probs of image-conditioned samples at trainer CP1 x EP4
against an attention-DP4 x EP4 engine, plain and with the engine retracting and speculating.

The slice of the text gate (``test_qwen3_8_flash_next_4layer_true_on_policy``: three GDN layers and one QSA layer
under hyper-connections, the PLE n-gram layer, its MTP layer) with the checkpoint's whole 27-layer vision tower.
The prompts are drawn images through SGLang's token-space processor (``write_image_prompts(long=True)``): every image
prompt opens with 2,264 image tokens, so QSA selects among image keys in prefill, and one prompt is text only.

Images reach three places the text gate does not. The frozen tower's rows replace the placeholder embeddings, as
in the Qwen GDN image gates. QSA rotates its index queries and compressed keys by the model's mRoPE positions, each
rotary lane at its own axis, while it counts key groups and blocks by token index; a compressed key rotates at its
group's first member, which may be an image token. The PLE layer hashes the id serving's language model reads at a
placeholder (the last vocabulary id), not the placeholder token.

Phase ``ep4`` runs ``--ci-test`` with ``--true-on-policy-mode``, which fails on the first unequal log-prob over two
GRPO steps. Phase ``spec`` repeats it while the engine retracts running requests every 64 decode steps and speculates
through the checkpoint's MTP layer (EAGLE (3, 1, 4)): a request restores its state from the radix cache and prefills
its decoded tokens again, and verify compresses each completed group at the RoPE position its decode would use. The
slice's draft is accepted almost never, so this checks verify and retraction on image requests, not accepted drafts.
Step 1 of both phases scores the same rollouts from the same weights, which is checked, and its gradient norm must
agree across them. Phase ``video`` runs as ``ep4`` on drawn clips behind the same 2,264 image tokens
(``write_video_prompts(long=True)``), so QSA selects among image and video keys; it is judged by its log-probs.
"""

import math
import os

import torch

from tests.ci.ci_register import register_cuda_ci
from tests.e2e.precision._qwen_gdn_common import (
    grpo_args,
    slice_checkpoint,
    vision_args,
    write_image_prompts,
    write_video_prompts,
)

from miles.true_on_policy.launch import top_launch
from miles.utils.debug_utils.top_capture import ROLLOUT_DATA, check_same_rollouts, probe_reward
from miles.utils.external_utils import command_utils

# Nightly, as the other family gates. est_time is an upper bound until a /rerun-test run measures it.
register_cuda_ci(
    est_time=3600,
    suite="stage-c-8-gpu-h200",
    labels=["megatron", "precision"],
    hardware=["hopper", "blackwell"],
    num_gpus=8,
    nightly=True,
)

HF_REPO = "Qwen/Qwen3.8-Flash-Next"
SOURCE = "/root/models/Qwen3.8-Flash-Next"
# The first four layers and the MTP layer, which only the speculating phase's engine runs.
MODEL_DIR = "/root/models/Qwen3.8-Flash-Next-4layer-mtp"
# The 4-layer args without --moe-router-dtype, which TOP refuses.
MODEL_TYPE = "qwen3.8-flash-next-4layer-top"
DATA_DIR = "/root/datasets/top_vision_drawings_long"
VIDEO_DATA_DIR = "/root/datasets/top_vision_clips_long"
NUM_GPUS = 8
TRAINER_GPUS = ENGINE_GPUS = 4
DUMP_DIR = "/root/dumps/qwen3_8_flash_next_4layer_vision_true_on_policy"
GRAD_NORM_DIR = "/root/grad_norm/qwen3_8_flash_next_4layer_vision_true_on_policy"
# (the engine retracts running requests, the engine speculates) of each phase.
PHASES = {"ep4": (False, False), "spec": (True, True), "video": (False, False)}
# The phases that train on the same rollouts.
COMPARED = ("ep4", "spec")
RETRACT_INTERVAL = 64
# EAGLE on the checkpoint's MTP layer: three draft steps of one chain, the root and three drafts verified together.
MTP_RECIPE = (3, 1, 4)
# Step-1 gradient norms of the two phases; they differ only in reduction order.
GRAD_NORM_RTOL = 1e-3


def prepare():
    U = command_utils.default_config().create_backend()
    U.exec_command_cpu("mkdir -p /root/models /root/datasets")
    U.exec_command_cpu(f"hf download {HF_REPO} --local-dir {SOURCE}")
    slice_checkpoint(SOURCE, MODEL_DIR, num_layers=4, keep_mtp=True)
    write_prompts()


def write_prompts():
    write_image_prompts(DATA_DIR, long=True)
    write_video_prompts(VIDEO_DATA_DIR, long=True)


def execute(phase: str):
    retract, speculate = PHASES[phase]
    U = command_utils.default_config().create_backend()
    launch = top_launch(model_type="qwen4_exp", engine_gpus=ENGINE_GPUS)
    os.makedirs(f"{GRAD_NORM_DIR}/{phase}", exist_ok=True)

    perf_args = (
        "--tensor-model-parallel-size 1 "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        f"--expert-model-parallel-size {TRAINER_GPUS} "
        "--expert-tensor-parallel-size 1 "
    )
    steps, topk, draft_tokens = MTP_RECIPE
    spec_args = (
        "--sglang-speculative-algorithm NEXTN "
        f"--sglang-speculative-num-steps {steps} --sglang-speculative-eagle-topk {topk} "
        f"--sglang-speculative-num-draft-tokens {draft_tokens} "
        if speculate
        else ""
    )
    train_args = (
        f"{grpo_args(model_dir=MODEL_DIR, prompt_data=None)} "
        f"{vision_args(data_dir=VIDEO_DATA_DIR, video=True) if phase == 'video' else vision_args(data_dir=DATA_DIR)} "
        f"{perf_args} "
        f"--rollout-num-gpus {ENGINE_GPUS} --rollout-num-gpus-per-engine {ENGINE_GPUS} "
        # DeepEP's low-latency decode dispatches at most 128 tokens per rank; capture decode graphs below that
        # (a verify step scores 4 rows a request).
        f"--sglang-cuda-graph-max-bs-decode {16 if speculate else 32} "
        f"{spec_args}"
        f"--actor-num-nodes 1 --actor-num-gpus-per-node {TRAINER_GPUS} "
        # The engines keep the vision tower and the frozen host n-gram table; the trainer syncs neither.
        "--check-weight-update-skip-list visual. ple_embedding. "
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
        extra_env_vars={
            **launch.env_vars,
            **({"SGLANG_TEST_RETRACT": "1", "SGLANG_TEST_RETRACT_INTERVAL": str(RETRACT_INTERVAL)} if retract else {}),
        },
    )


def assert_backward_matches():
    spec, plain = (
        torch.load(f"{GRAD_NORM_DIR}/{phase}/actor_0_0.pt", weights_only=False) for phase in ("spec", "ep4")
    )
    assert plain > 0 and math.isclose(spec, plain, rel_tol=GRAD_NORM_RTOL), f"step-1 grad norm spec {spec} vs {plain}"


def judge_backward() -> dict:
    """The two phases scored the same image rollouts, and their step-1 gradient norms agree."""
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
