"""Qwen GDN MoE TOP gate: exact log-probs at trainer all-gather CP4 and CP1, the CP4 backward against CP1's per tensor,
and the CP1 backward against its contract and an FP32 reference; then the same at CP1 with forced retraction, and
agentic sessions at CP4.

Qwen3.5-35B-A3B cut to 4 layers (three GDN layers, one full-attention layer, 256 routed experts with a gated shared
expert). One 4-GPU engine serves with attention DP4 and EP4 over DeepEP; radix reuse is on, so the 2 samples of a
prompt restore their prompt's GDN state from a 64-token grid point; the decode KV split is lowered to 512 keys so
1,024-token responses cross it, and every response crosses GDN chunk boundaries in decode.

Phase ``cp4`` trains at CP4 x EP4 on 4 GPUs: each GDN layer hands its FP32 state and convolution inputs from rank to
rank, full attention all-gathers its keys. Phase ``cp1`` trains at CP1 x EP4 (DP4). Both capture the GDN layer 2 and
the full-attention layer 3, and run ``--ci-test`` with ``--true-on-policy-mode``, which fails on the first unequal
log-prob over two GRPO steps. Step 1 of both phases scores the same rollouts from the same weights, which is checked
first; then ``miles.utils.debug_utils.top_layout`` compares the CP4 capture against CP1's per tensor (the state
gradient handed back across ranks, the CP chunk-boundary rows), and the step-1 gradient norm must agree across the two
(the uncaptured layers). Last, ``miles.utils.debug_utils.top_vjp`` judges the CP1 capture: each module's backward
against HF's at TOP's own module inputs and routing (the contract), and each layer against HF in FP32 on its PyTorch
GDN path (the quality).

Phase ``retract`` trains at CP1 like ``cp1``, but the engine retracts running requests every 64 decode steps, so a
request resumes from the radix cache and prefills its decoded tokens again: its GDN outputs must still be the chunk
prefill's over the whole sample, which the trainer runs once. Its rollouts differ from the other phases', so it is
judged alone: exact log-probs, and the verdict on its capture.

Phase ``multi_turn`` trains at CP4 on agentic sessions (``top_multi_turn``: three turns per sample, a tool message
between turns, each GRPO group pinned to one DP rank) with forced retraction. A later turn restores its previous turn's
decoded state from the radix tree and prefills on from it, so the phase checks exact log-probs across turns and that
later turns hit their previous reply. Every other group ends its earlier turns on a stop token at a track boundary, so
the next turn restores the decode snapshot one interval before that end, taken while the boundary step after the last
token was still in flight.
"""

import math
import os
from pathlib import Path

import torch

from tests.ci.ci_register import register_cuda_ci
from tests.e2e.precision._qwen_gdn_common import DECODE_KV_SPLIT, grpo_args, slice_checkpoint

from miles.true_on_policy.launch import top_launch
from miles.utils.debug_utils import top_multi_turn
from miles.utils.debug_utils.top_capture import capture_args, check_same_rollouts, fail_on, probe_reward
from miles.utils.debug_utils.top_layout import layout_verdict
from miles.utils.debug_utils.top_vjp import vjp_verdict
from miles.utils.external_utils import command_utils

# Nightly, as the GLM and DeepSeek-V4 gates. est_time is an upper bound until a /rerun-test run measures it.
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
# The 4-layer args without --moe-router-dtype, which TOP refuses.
MODEL_TYPE = "qwen3.5-35B-A3B-4layer-top"
NUM_GPUS = 8
TRAINER_GPUS = ENGINE_GPUS = 4
DUMP_DIR = "/root/dumps/qwen3_5_35B_A3B_4layer_true_on_policy"
GRAD_NORM_DIR = "/root/grad_norm/qwen3_5_35B_A3B_4layer_true_on_policy"
# A GDN layer and the full-attention layer after it (the handoff between them is checked bit for bit).
VJP_LAYERS = (2, 3)
# Step-1 gradient norms of the CP4 and CP1 phases; they differ only in reduction order.
GRAD_NORM_RTOL = 1e-3
# Context parallel size of each phase; CP4 has DP1, CP1 has DP4 (expert parallelism spans the DP ranks).
PHASES = {"cp4": 4, "cp1": 1, "retract": 1, "multi_turn": 4}
# The phases that train on the same rollouts and are compared per tensor.
COMPARED = ("cp4", "cp1")
# Phases ``retract`` and ``multi_turn`` retract running requests every this many decode steps.
RETRACT_INTERVAL = 64
MULTI_TURN_DIR = "/root/dumps/qwen3_5_35B_A3B_4layer_multi_turn"
# Share of later turns whose radix hit reaches their previous reply's last page; arbitrary margin for a forced
# retraction or an eviction.
MULTI_TURN_HIT_RATE = 0.9


def prepare():
    U = command_utils.default_config().create_backend()
    U.exec_command_cpu("mkdir -p /root/models /root/datasets")
    U.exec_command_cpu(f"hf download {HF_REPO} --local-dir {SOURCE}")
    U.hf_download_dataset("zhuzilin/dapo-math-17k")
    slice_checkpoint(SOURCE, MODEL_DIR, num_layers=4)


def execute(phase: str):
    cp = PHASES[phase]
    U = command_utils.default_config().create_backend()
    launch = top_launch(model_type="qwen3_5_moe", engine_gpus=ENGINE_GPUS)
    os.makedirs(f"{GRAD_NORM_DIR}/{phase}", exist_ok=True)

    perf_args = (
        "--tensor-model-parallel-size 1 "
        "--pipeline-model-parallel-size 1 "
        f"--context-parallel-size {cp} "
        f"{'--allgather-cp ' if cp > 1 else ''}"
        f"--expert-model-parallel-size {TRAINER_GPUS} "
        "--expert-tensor-parallel-size 1 "
    )
    # The judged phases capture the same layers, with the probe reward and unclipped gradients: step 1 sees the same
    # loss. The agentic phase is judged by its log-probs and radix hits only, still under the probe reward (the gate
    # args name no rule-based reward).
    if phase == "multi_turn":
        capture = (
            f"--custom-generate-function-path {top_multi_turn.__name__}.generate "
            f"--save-debug-rollout-data {MULTI_TURN_DIR}/{{rollout_id}}.pt "
            f"--custom-rm-path {probe_reward.__module__}.{probe_reward.__name__} "
        )
    else:
        capture = capture_args(dump_dir=f"{DUMP_DIR}/{phase}", layers=VJP_LAYERS)
    train_args = (
        f"{grpo_args(model_dir=MODEL_DIR)} "
        f"{perf_args} "
        f"--rollout-num-gpus {ENGINE_GPUS} --rollout-num-gpus-per-engine {ENGINE_GPUS} "
        # DeepEP's low-latency decode dispatches at most 128 tokens per rank; capture decode graphs below that.
        "--sglang-cuda-graph-max-bs-decode 32 "
        f"--actor-num-nodes 1 --actor-num-gpus-per-node {TRAINER_GPUS} "
        f"--ci-save-grad-norm {GRAD_NORM_DIR}/{phase}/{{role}}_{{rollout_id}}_{{step_id}}.pt "
        f"{command_utils.get_default_wandb_args(__file__)} "
        f"{capture} "
        f"{' '.join(launch.train_args)} "
    )

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=NUM_GPUS,
        megatron_model_type=MODEL_TYPE,
        extra_env_vars={
            **launch.env_vars,
            "SGLANG_TOP_DECODE_KV_SPLIT": DECODE_KV_SPLIT,
            **(
                {"SGLANG_TEST_RETRACT": "1", "SGLANG_TEST_RETRACT_INTERVAL": str(RETRACT_INTERVAL)}
                if phase in ("retract", "multi_turn")
                else {}
            ),
        },
    )


def assert_cp_backward_matches():
    cp4, cp1 = (torch.load(f"{GRAD_NORM_DIR}/{phase}/actor_0_0.pt", weights_only=False) for phase in ("cp4", "cp1"))
    assert cp4 > 0 and math.isclose(cp4, cp1, rel_tol=GRAD_NORM_RTOL), f"step-1 grad norm CP4 {cp4} vs CP1 {cp1}"


def assert_multi_turn_hits():
    train_args = top_launch(model_type="qwen3_5_moe", engine_gpus=ENGINE_GPUS).train_args
    page_size = int(train_args[train_args.index("--sglang-page-size") + 1])
    track_interval = int(train_args[train_args.index("--sglang-mamba-track-interval") + 1])
    samples = [
        sample
        for path in sorted(Path(MULTI_TURN_DIR).glob("*.pt"))
        for sample in torch.load(path, weights_only=False)["samples"]
    ]
    report = top_multi_turn.turn_hit_report(samples, page_size=page_size, track_interval=track_interval)
    print(f"MULTI_TURN_HITS {report}", flush=True)
    assert report["later_turns"] and report["rate"] >= MULTI_TURN_HIT_RATE, report
    assert report["boundary_turns"], report


def judge_backward() -> dict:
    """Rollout identity first, then the layout check and the verdicts of ``cp1`` and ``retract`` (all reported before
    any fails), then the step-1 gradient norms."""
    dump_dirs = {phase: f"{DUMP_DIR}/{phase}" for phase in COMPARED}
    check_same_rollouts(dump_dirs)
    layout = layout_verdict(model_path=MODEL_DIR, dump_dirs=dump_dirs, layers=VJP_LAYERS)
    verdict = vjp_verdict(family="Qwen GDN MoE", model_path=MODEL_DIR, dump_dir=dump_dirs["cp1"], layers=VJP_LAYERS)
    retract = vjp_verdict(
        family="Qwen GDN MoE", model_path=MODEL_DIR, dump_dir=f"{DUMP_DIR}/retract", layers=VJP_LAYERS
    )
    fail_on(layout=layout["failures"], verdict=verdict["failures"], retract=retract["failures"])
    assert_cp_backward_matches()
    return dict(layout=layout, verdict=verdict, retract=retract)


if __name__ == "__main__":
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    for phase in PHASES:
        execute(phase)
    judge_backward()
    assert_multi_turn_hits()
