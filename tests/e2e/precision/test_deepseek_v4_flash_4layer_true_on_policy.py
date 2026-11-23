"""DeepSeek-V4 TOP gate (block-FP8): exact log-probs at trainer CP1 x EP4, all-gather CP2 x EP4 and PP2 x EP2, their
backward against CP1's per tensor, and the CP1 backward against its contract and an FP32 reference.

DeepSeek-V4-Flash-FP8 cut to 4 layers (layers 0-3; the checkpoint's MTP layer is not served): compress ratios [0, 0, 4,
128], so one SWA-only pair, one C4 layer with its indexer and one C128 layer; layers 0-2 route by token hash and layer
3 by sqrtsoftplus score; 256 routed experts with a shared expert, mHC over 4 residual streams. Weights are E4M3 with
UE8M0 128x128 block scales, FP8 experts and a BF16 wo_a, as the released checkpoint; the trainer installs BF16 masters
behind serving's FP8 bytes straight from the HF checkpoint. One 4-GPU engine serves with attention DP4 and EP4 over
DeepEP on an FP8 KV cache. Responses run to 3,072 tokens, past the C4 indexer's 512-entry top-k over 4x-compressed
keys, so the selection is sparse in prefill, decode and the trainer's scoring.

Phase ``cp1`` trains at CP1 x EP4 (DP4), ``cp2`` at all-gather CP2 x EP4 (DP2) and ``pp2`` at PP2 x EP2 (DP2) with
layers 0-2 and 3 on the two stages (Megatron keeps every hash-routed layer on the embedding's stage). No phase
speculates: on a slice the checkpoint's MTP draft is almost always rejected, so speculation is exercised by full-model
runs, not by this gate. Every phase captures layers 2 (C4, hash routing) and 3
(C128, score routing). Every phase fits GPU memory as the DAPO-4k recipe does: one sample per micro-batch, MoE token
chunks, and Adam state streamed from node-local disk.

All phases run ``--ci-test`` with ``--true-on-policy-mode``, which fails on the first unequal log-prob over two GRPO
steps, and run as TOP recipes do (``TOP_RECIPE_TRAIN_ARGS``). Step 1 of every phase scores the same rollouts from the
same weights, which is checked first; then ``miles.utils.debug_utils.top_layout`` compares the captures of ``cp2`` and
``pp2`` against ``cp1`` per tensor (CP attention backward, local-row rotary, the PP cotangent handoff and mHC across
stages), and the step-1 gradient norm must agree across the three layouts (the uncaptured layers). Last,
``miles.utils.debug_utils.top_vjp`` judges the ``cp1`` capture in DeepSeek's inference Block: each module's backward
against its declared surrogate at TOP's own module inputs, at the served weights with TOP's routing and index selection
and the KV rows rounded as the FP8 cache stores them, and each layer against the unquantized FP32 reference at the
trainer's masters, bounded by the FP8-STE reference's own distance from it.
"""

import math
import os

import torch

from tests.ci.ci_register import register_cuda_ci

from miles.true_on_policy.launch import TOP_RECIPE_TRAIN_ARGS, top_launch
from miles.utils.debug_utils.top_capture import capture_args, check_same_rollouts, fail_on
from miles.utils.debug_utils.top_layout import layout_verdict
from miles.utils.debug_utils.top_vjp import vjp_verdict
from miles.utils.external_utils import command_utils

# Nightly, as the GLM gate: three 8-GPU phases and the VJP. est_time is an upper bound until a /rerun-test measures it.
register_cuda_ci(
    est_time=3600,
    suite="stage-c-8-gpu-h200",
    labels=["megatron", "precision"],
    hardware=["hopper"],
    num_gpus=8,
    nightly=True,
)

HF_REPO = "Pinaster/DeepSeek-V4-Flash-FP8_4layer-mtp"
MODEL_NAME = "DeepSeek-V4-Flash-FP8_4layer-mtp"
MODEL_DIR = f"/root/models/{MODEL_NAME}"
# The reference of the VJP: every FP8 weight as the BF16 weight serving computes with at the first rollout.
REFERENCE_DIR = f"{MODEL_DIR}-served"
# Every FP8 weight as the trainer's BF16 master: the VJP's unquantized reference and its contract-bias report.
MASTERS_DIR = f"{MODEL_DIR}-masters"
MODEL_TYPE = "deepseek-v4-flash-4layer-top"
NUM_GPUS = 8
TRAINER_GPUS = ENGINE_GPUS = 4
DUMP_DIR = "/root/dumps/deepseek_v4_flash_4layer_true_on_policy"
GRAD_NORM_DIR = "/root/grad_norm/deepseek_v4_flash_4layer_true_on_policy"
# The C4 layer (its own indexer, hash routing) and the C128 layer (score routing).
VJP_LAYERS = (2, 3)
# Past the C4 indexer's 512 selected entries of 4x-compressed keys, so its selection is sparse.
RESPONSE_LEN = 3072
# The DAPO-4k recipe's micro-batch budget allows a 512-token prompt beside the response.
PROMPT_BUDGET = 512
# Node-local disk for the streamed optimizer state.
OPTIMIZER_STATE_DIR = "/root/optimizer_state/deepseek_v4_flash_4layer_true_on_policy"
# Step-1 gradient norms of the phases; they differ only in reduction order.
GRAD_NORM_RTOL = 1e-3
# (context parallel, pipeline stages, expert parallel); EP spans the DP and CP ranks of a stage.
PHASES = {"cp1": (1, 1, 4), "cp2": (2, 1, 4), "pp2": (1, 2, 2)}
# The layers of each pipeline stage under PP2: Megatron keeps every hash-routed layer (0-2) on the embedding's stage.
PP_STAGE_LAYERS = (3, 1)


def prepare():
    U = command_utils.default_config().create_backend()
    U.exec_command_cpu("mkdir -p /root/models /root/datasets")
    U.exec_command_cpu(f"hf download {HF_REPO} --local-dir {MODEL_DIR}")
    U.hf_download_dataset("zhuzilin/dapo-math-17k")
    for weights, save_dir in (("served", REFERENCE_DIR), ("masters", MASTERS_DIR)):
        U.exec_command_gpu_once(
            f"python tools/convert_fp8_to_bf16_reference.py --weights {weights} --model-dir {MODEL_DIR} "
            f"--save-dir {save_dir}",
            save_dir,
        )


def execute(phase: str):
    cp, pp, ep = PHASES[phase]
    U = command_utils.default_config().create_backend()
    launch = top_launch(model_type="deepseek_v4", engine_gpus=ENGINE_GPUS, fp8=True)
    os.makedirs(f"{GRAD_NORM_DIR}/{phase}", exist_ok=True)

    # The model provider installs the masters from --hf-checkpoint and skips the HF --ref-load.
    ckpt_args = f"--hf-checkpoint {MODEL_DIR}/ --ref-load {MODEL_DIR}/ "

    rollout_args = (
        "--prompt-data /root/datasets/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--num-rollout 2 "
        "--rollout-batch-size 4 "
        "--n-samples-per-prompt 2 "
        f"--rollout-max-response-len {RESPONSE_LEN} "
        "--rollout-temperature 0.8 "
        "--global-batch-size 8 "
    )

    first, last = PP_STAGE_LAYERS
    stages = f"--decoder-first-pipeline-num-layers {first} --decoder-last-pipeline-num-layers {last} "
    perf_args = (
        "--tensor-model-parallel-size 1 "
        f"--pipeline-model-parallel-size {pp} "
        f"{stages if pp > 1 else ''}"
        f"--context-parallel-size {cp} "
        f"{'--allgather-cp ' if cp > 1 else ''}"
        f"--expert-model-parallel-size {ep} "
        "--expert-tensor-parallel-size 1 "
        # GPU memory as the DAPO-4k recipe fits it: one sample per micro-batch and a 2 GiB margin.
        "--use-dynamic-batch-size "
        f"--max-tokens-per-gpu {-(-(RESPONSE_LEN + PROMPT_BUDGET) // cp)} "
        "--train-memory-margin-bytes 2147483648 "
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
        # Step 2 must read FP8 bytes the optimizer moved.
        "--lr 1e-5 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
        # Adam over FP32 masters and BF16 moments streamed per bucket from node-local disk, as the GLM gate.
        "--stream-optimizer-state-to-disk "
        "--stream-optimizer-state-moment-dtype bf16 "
        f"--offload-train-disk-dir {OPTIMIZER_STATE_DIR}/{phase} "
    )

    sglang_args = (
        f"--rollout-num-gpus {ENGINE_GPUS} --rollout-num-gpus-per-engine {ENGINE_GPUS} "
        "--sglang-mem-fraction-static 0.7 "
        "--sglang-context-length 8192 "
        # DeepEP's low-latency decode dispatches at most 128 tokens per rank; capture decode graphs below that.
        "--sglang-cuda-graph-max-bs-decode 32 "
    )

    # Every phase captures the same layers, with the probe reward and unclipped gradients: step 1 sees the same loss.
    capture = capture_args(
        dump_dir=f"{DUMP_DIR}/{phase}", layers=VJP_LAYERS, stage_layers=PP_STAGE_LAYERS if pp > 1 else None
    )

    misc_args = (
        "--attention-dropout 0.0 --hidden-dropout 0.0 "
        f"--actor-num-nodes 1 --actor-num-gpus-per-node {TRAINER_GPUS} "
        f"--ci-save-grad-norm {GRAD_NORM_DIR}/{phase}/{{role}}_{{rollout_id}}_{{step_id}}.pt "
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
        f"{capture} "
        f"{' '.join(TOP_RECIPE_TRAIN_ARGS)} "
        f"{' '.join(launch.train_args)} "
    )

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=NUM_GPUS,
        megatron_model_type=MODEL_TYPE,
        # The recipe's MoE token chunks: a hot EP rank's rows are bounded per chunk (row-wise kernels, same bytes).
        extra_env_vars={**launch.env_vars, "TOP_MOE_CHUNKS": "2"},
    )


def assert_backward_matches():
    norms = {phase: torch.load(f"{GRAD_NORM_DIR}/{phase}/actor_0_0.pt", weights_only=False) for phase in PHASES}
    reference = norms["cp1"]
    assert reference > 0, f"step-1 grad norm {norms}"
    for norm in norms.values():
        assert math.isclose(norm, reference, rel_tol=GRAD_NORM_RTOL), f"step-1 grad norms {norms}"


def judge_backward() -> dict:
    """Rollout identity first, then the verdict and the layout check (both reported before either fails; the layout
    check judges the near-cancelling mHC biases against the verdict's token sums), then the step-1 gradient norms."""
    dump_dirs = {phase: f"{DUMP_DIR}/{phase}" for phase in PHASES}
    check_same_rollouts(dump_dirs)
    verdict = vjp_verdict(
        family="DeepSeek-V4",
        model_path=REFERENCE_DIR,
        dump_dir=dump_dirs["cp1"],
        layers=VJP_LAYERS,
        kv_cache_dtype=top_launch(model_type="deepseek_v4", engine_gpus=ENGINE_GPUS, fp8=True).kv_cache_dtype,
        masters_path=MASTERS_DIR,
    )
    layout = layout_verdict(
        model_path=MODEL_DIR,
        dump_dirs=dump_dirs,
        layers=VJP_LAYERS,
        stage_layers={"pp2": PP_STAGE_LAYERS},
        token_sums=verdict["token_sums"],
    )
    fail_on(layout=layout["failures"], verdict=verdict["failures"])
    assert_backward_matches()
    return dict(layout=layout, verdict=verdict)


if __name__ == "__main__":
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    for phase in PHASES:
        execute(phase)
    judge_backward()
