"""GLM TOP gate (block-FP8): exact log-probs at trainer all-gather CP2 x EP4, CP1 x EP4 and PP2 x EP2 (the FP8 experts
across pipeline stages), the CP2 and PP2 backward against CP1's per tensor, and the CP1 backward against its contract
and an FP32 reference.

GLM-5.2-FP8 cut to 5 layers: 3 dense and 2 MoE layers, 256 routed experts with a shared expert; layers 0-2 run their
own DSA indexer and layers 3-4 reuse layer 2's index selection. Weights are E4M3 with FP32 128x128 block scales, as
the released checkpoint; the trainer installs BF16 masters behind serving's FP8 bytes straight from the HF checkpoint.
One 4-GPU engine serves with attention DP4 and EP4 over DeepEP on an FP8 KV cache. Responses run to 3,072 tokens, so
most rows have more causal keys than the 2,048-key index top-k and the DSA selection is sparse in prefill, decode and
the trainer's scoring. Step 1 scores the sync quantizer's bytes, not the checkpoint's (Miles syncs before the first
rollout), and step 2 those of one optimizer step at lr 1e-5, large enough to move FP8 bytes; both read the trainer's
FP8 caches rebuilt from the masters with serving's sync quantizer.

Phase ``cp2`` trains at CP2 x EP4 on 4 GPUs, ``cp1`` at CP1 x EP4 (DP4), and ``pp2`` at PP2 x EP2 (DP2) with layers
0-1 on the first stage and 2-4 on the second, so each stage starts on a layer that computes its own index selection.
Every phase captures layer 2 (its own indexer, dense MLP) and layer 3 (layer 2's selection, MoE). Every phase fits GPU
memory as the DAPO-4k recipe does: one sample per micro-batch, MoE token chunks, and Adam state streamed from
node-local disk.

All phases run ``--ci-test`` with ``--true-on-policy-mode``, which fails on the first unequal log-prob over two GRPO
steps, and run as TOP recipes do (``TOP_RECIPE_TRAIN_ARGS``). Step 1 of every phase scores the same rollouts from the
same weights, which is checked first; then ``miles.utils.debug_utils.top_layout`` compares the captures of ``cp2`` and
``pp2`` against ``cp1`` per tensor, and the step-1 gradient norm must agree across the three layouts (the uncaptured
layers). Last, ``miles.utils.debug_utils.top_vjp`` judges the ``cp1`` capture: each module's backward against its
declared surrogate at TOP's own module inputs, at the served weights with TOP's routing and index selection and the
MLA latent rounded as the FP8 cache stores it, and each layer against the unquantized FP32 reference at the trainer's
masters, bounded by the FP8-STE reference's own distance from it.
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

# Nightly: three phases, then the VJP, run above the regular runs' 2,400 s cap; est_time is an upper bound.
register_cuda_ci(
    est_time=3600,
    suite="stage-c-8-gpu-h200",
    labels=["megatron", "precision"],
    hardware=["hopper"],
    num_gpus=8,
    nightly=True,
)

HF_REPO = "Pinaster/GLM-5.2-FP8_5layer"
MODEL_NAME = "GLM-5.2-FP8_5layer"
MODEL_DIR = f"/root/models/{MODEL_NAME}"
# The HF reference of the VJP: every FP8 weight as the BF16 weight serving computes with at the first rollout.
REFERENCE_DIR = f"{MODEL_DIR}-served"
# Every FP8 weight as the trainer's BF16 master: the VJP's unquantized reference and its contract-bias report.
MASTERS_DIR = f"{MODEL_DIR}-masters"
MODEL_TYPE = "glm5.2-744B-A40B_5layer-top"
NUM_GPUS = 8
TRAINER_GPUS = ENGINE_GPUS = 4
DUMP_DIR = "/root/dumps/glm5_2_5layer_true_on_policy"
GRAD_NORM_DIR = "/root/grad_norm/glm5_2_5layer_true_on_policy"
# The last layer with its own indexer and the first that reuses its selection (the handoff is checked bit for bit).
VJP_LAYERS = (2, 3)
# Longer than the 2,048-key index top-k, so the DSA selection is sparse.
RESPONSE_LEN = 3072
# The DAPO-4k recipe's micro-batch budget allows a 512-token prompt beside the response.
PROMPT_BUDGET = 512
# Node-local disk for the streamed optimizer state.
OPTIMIZER_STATE_DIR = "/root/optimizer_state/glm5_2_5layer_true_on_policy"
# Step-1 gradient norms of the phases; they differ only in reduction order.
GRAD_NORM_RTOL = 1e-3
# (context parallel, pipeline stages, expert parallel); EP spans the DP and CP ranks of a stage.
PHASES = {"cp2": (2, 1, 4), "cp1": (1, 1, 4), "pp2": (1, 2, 2)}
# The layers of each pipeline stage under PP2.
PP_STAGE_LAYERS = (2, 3)


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
    launch = top_launch(model_type="glm_moe_dsa", engine_gpus=ENGINE_GPUS, fp8=True)
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
        # Step 2 must read FP8 bytes the optimizer moved; at 1e-6 one step leaves nearly all of them in place.
        "--lr 1e-5 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
        # Adam state as the DAPO-4k recipe fits it: GPU Adam over FP32 masters and BF16 moments streamed per bucket
        # from node-local disk. With unsharded expert state (EP spans every trainer GPU) the FP8 bytes beside the BF16
        # masters leave no room for it on the GPU.
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
    """Rollout identity first, then the layout check and the verdict (both reported before either fails), then the
    step-1 gradient norms."""
    dump_dirs = {phase: f"{DUMP_DIR}/{phase}" for phase in PHASES}
    check_same_rollouts(dump_dirs)
    layout = layout_verdict(
        model_path=MODEL_DIR, dump_dirs=dump_dirs, layers=VJP_LAYERS, stage_layers={"pp2": PP_STAGE_LAYERS}
    )
    verdict = vjp_verdict(
        family="GLM",
        model_path=REFERENCE_DIR,
        dump_dir=dump_dirs["cp1"],
        layers=VJP_LAYERS,
        kv_cache_dtype=top_launch(model_type="glm_moe_dsa", engine_gpus=ENGINE_GPUS, fp8=True).kv_cache_dtype,
        masters_path=MASTERS_DIR,
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
