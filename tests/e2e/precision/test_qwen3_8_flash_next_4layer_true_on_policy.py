"""Qwen3.8-Flash-Next (HF ``qwen4_exp``) TOP gate: exact log-probs at trainer CP1 x EP4 against an engine with
attention DP4 and EP4 over DeepEP, over two GRPO steps (plain, with forced retraction, and speculating), and the
backward of the plain phase against transformers' ``qwen4_exp``.

Qwen3.8-Flash-Next cut to 4 layers plus its MTP layer: three GDN layers and one QSA full-attention layer under 4-branch
hyper-connections, the PLE n-gram layer at layer 1, 512 routed experts (top-10) with a gated shared expert. 512 experts
need EP >= 4 on both sides. Responses run to 3,072 tokens: QSA's block top-k keeps 512 blocks of 4 keys, so it only
selects among keys past 2,048 tokens, in prefill, decode and the trainer's scoring alike; every response also crosses
GDN chunk boundaries in decode. The radix cache is on; the gate's rollouts record no cached prompt tokens, so retraction
is what restores from it.

Phase ``ep4`` runs ``--ci-test`` with ``--true-on-policy-mode``, which fails on the first unequal log-prob over two
GRPO steps (step 2 scores rollouts of the synced step-1 weights), and captures layers 1 (GDN under the PLE layer), 2
(GDN) and 3 (QSA) for ``miles.utils.debug_utils.top_vjp``: each module's backward against the reference's at TOP's own
module inputs, routing and index selection (the contract: GDN, QSA attention, the MoE block, and each
hyper-connection's read and write gates), and each layer against the reference in FP32 (the quality, reported; the
PLE layer's own parameters appear there only). Phase ``retract`` repeats the run while the engine retracts running requests every 64 decode steps, so a request restores its
GDN state, PLE convolution state and QSA keys from the radix cache at a 64-token grid point and prefills its decoded
tokens again. Phase ``spec`` adds EAGLE (3, 1, 4) with the checkpoint's MTP layer as the draft to the retracting
engine: verify scores the root and three drafts per request in one forward and draws every emitted token from the
target, so the rollouts are still those of plain decode. The slice's draft is a 48-layer model's head on a 4-layer
model and is accepted almost never, so this phase checks the verify path and that rejected drafts leave no state
behind; accepted drafts are qualified on the full checkpoint (serving's speculating ids and log-probs equal the
non-speculating run's; no CI job).

Sampling depends on the seed and position only, so step 1 of every phase scores the same rollouts from the same
weights, which is checked, and the step-1 gradient norms must agree (the backward's reduction order is not pinned,
so not bit for bit).
"""

import math
import os

import torch

from tests.ci.ci_register import register_cuda_ci
from tests.e2e.precision._qwen_gdn_common import grpo_args, slice_checkpoint

from miles.true_on_policy.launch import top_launch
from miles.utils.debug_utils.top_capture import ROLLOUT_DATA, capture_args, check_same_rollouts, fail_on, probe_reward
from miles.utils.debug_utils.top_vjp import vjp_verdict
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
NUM_GPUS = 8
TRAINER_GPUS = ENGINE_GPUS = 4
DUMP_DIR = "/root/dumps/qwen3_8_flash_next_4layer_true_on_policy"
GRAD_NORM_DIR = "/root/grad_norm/qwen3_8_flash_next_4layer_true_on_policy"
# The GDN layer under the PLE layer, the GDN layer after it and the QSA layer (the handoffs between them are checked
# bit for bit).
VJP_LAYERS = (1, 2, 3)
# The phase whose capture the verdict judges.
JUDGED = "ep4"
# Past QSA's 2,048-key budget, so its block top-k is selective for the later rows of every sample.
RESPONSE_LEN = 3072
# (the engine retracts running requests, the engine speculates) of each phase.
PHASES = {"ep4": (False, False), "retract": (True, False), "spec": (True, True)}
RETRACT_INTERVAL = 64
# EAGLE on the checkpoint's MTP layer: three draft steps of one chain, the root and three drafts verified together.
# The recipe is passed explicitly: TOP admits the serving layout before SGLang fills in NEXTN's defaults.
MTP_RECIPE = (3, 1, 4)
NUM_ROLLOUTS = 2
# Step-1 gradient norms of the two phases; they differ only in reduction order (the Qwen GDN MoE gate's tolerance).
GRAD_NORM_RTOL = 1e-3


def prepare():
    U = command_utils.default_config().create_backend()
    U.exec_command_cpu("mkdir -p /root/models /root/datasets")
    U.exec_command_cpu(f"hf download {HF_REPO} --local-dir {SOURCE}")
    U.hf_download_dataset("zhuzilin/dapo-math-17k")
    slice_checkpoint(SOURCE, MODEL_DIR, num_layers=4, keep_mtp=True)


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
    # The judged phase captures its layers; every phase runs under the probe reward with unclipped gradients and saves
    # its rollouts, so step 1 sees the same loss and the reported gradient norms are those of a nonzero one.
    if phase == JUDGED:
        capture = capture_args(dump_dir=f"{DUMP_DIR}/{phase}", layers=VJP_LAYERS)
    else:
        capture = (
            f"--save-debug-rollout-data {DUMP_DIR}/{phase}/{ROLLOUT_DATA}/{{rollout_id}}.pt "
            "--clip-grad 0 "
            f"--custom-rm-path {probe_reward.__module__}.{probe_reward.__name__} "
        )
    train_args = (
        f"{grpo_args(model_dir=MODEL_DIR)} "
        f"--rollout-max-response-len {RESPONSE_LEN} "
        f"{perf_args} "
        f"--rollout-num-gpus {ENGINE_GPUS} --rollout-num-gpus-per-engine {ENGINE_GPUS} "
        # DeepEP's low-latency decode dispatches at most 128 tokens per rank; capture decode graphs below that
        # (a verify step scores 4 rows a request).
        f"--sglang-cuda-graph-max-bs-decode {16 if speculate else 32} "
        f"{_speculation_args() if speculate else ''}"
        f"--actor-num-nodes 1 --actor-num-gpus-per-node {TRAINER_GPUS} "
        # The engines keep the vision tower and the frozen host n-gram table; the trainer syncs neither.
        "--check-weight-update-skip-list visual. ple_embedding. "
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
            **({"SGLANG_TEST_RETRACT": "1", "SGLANG_TEST_RETRACT_INTERVAL": str(RETRACT_INTERVAL)} if retract else {}),
        },
    )


def _speculation_args() -> str:
    steps, topk, draft_tokens = MTP_RECIPE
    return (
        "--sglang-speculative-algorithm NEXTN "
        f"--sglang-speculative-num-steps {steps} --sglang-speculative-eagle-topk {topk} "
        f"--sglang-speculative-num-draft-tokens {draft_tokens} "
    )


def judge_backward() -> dict:
    """Rollout identity first; then the verdict on the judged phase's capture and every phase's step gradient norms
    (all reported before any fails). The norms must be finite and nonzero (a zero norm means the probe reward's loss
    reached no parameter), and step 1's must agree across the phases, which train on the same rollouts."""
    check_same_rollouts({phase: f"{DUMP_DIR}/{phase}" for phase in PHASES})
    verdict = vjp_verdict(
        family="Qwen3.8-Flash-Next", model_path=MODEL_DIR, dump_dir=f"{DUMP_DIR}/{JUDGED}", layers=VJP_LAYERS
    )
    report = dict(verdict=verdict)
    for phase in PHASES:
        norms = [
            float(torch.load(f"{GRAD_NORM_DIR}/{phase}/actor_{rollout}_0.pt", weights_only=False))
            for rollout in range(NUM_ROLLOUTS)
        ]
        print(f"GRAD_NORMS {phase} {norms}", flush=True)
        failures = [f"{phase} step {i}: grad norm {norm}" for i, norm in enumerate(norms) if not _usable(norm)]
        report[f"grad_norm_{phase}"] = dict(norms=norms, failures=failures)
    norm_failures = [failure for phase in PHASES for failure in report[f"grad_norm_{phase}"]["failures"]]
    fail_on(verdict=verdict["failures"], grad_norms=norm_failures)
    first, *rest = (report[f"grad_norm_{phase}"]["norms"][0] for phase in PHASES)
    assert all(math.isclose(norm, first, rel_tol=GRAD_NORM_RTOL) for norm in rest), f"step-1 grad norms: {report}"
    return report


def _usable(norm: float) -> bool:
    return math.isfinite(norm) and norm > 0


if __name__ == "__main__":
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    for phase in PHASES:
        execute(phase)
    judge_backward()
