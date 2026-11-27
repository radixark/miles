"""GLM-5.3-Flash TOP gate: exact log-probs over two GRPO steps, plain, with forced retraction and in agentic sessions
that reuse the radix cache, and the backward of the plain phase against transformers' ``glm5_next``.

GLM-5.3-Flash cut to 4 layers (``CharyZeng/GLM-5.3-Flash-4layer``: KDA, DSA, KDA, DSA; layer 0 a dense clamped MLP,
layers 1-3 the 288-expert sigmoid MoE with a shared expert; mHC hyper-connections on every sublayer), BF16. One
4-GPU engine serves with attention DP4 and EP4 over DeepEP and radix reuse; the trainer runs TP1 x EP4 on the
other 4 GPUs. Responses run to 3,072 tokens: the DSA indexer keeps 512 pools of 4 keys plus the open tail, so it only
selects among keys past 2,048 tokens, in prefill, decode and the trainer's scoring alike; every response also crosses
KDA chunk boundaries in decode (the row engine closes a chunk every 64 tokens), and prompts end off the 64-token grid
(the extend runs its tail through the row engine).

Phase ``ep4`` runs ``--ci-test`` with ``--true-on-policy-mode``, which fails on the first unequal log-prob over the
two steps (step 2 scores rollouts of the synced step-1 weights), and captures all four layers for
``miles.utils.debug_utils.top_vjp``: each module's backward against the reference's at TOP's own module inputs,
routing and index selection (the contract: KDA, DSA attention, the dense MLP and the MoE block, and each mHC site's
read), and each layer against the reference in FP32 (the quality, reported). Phase ``retract`` repeats it while the
engine retracts running requests every 64 decode steps, so a request prefills its decoded tokens again from the
64-token grid and its KDA outputs must still be the chunk prefill's over the whole sample.

Phase ``multi_turn`` runs agentic sessions (``top_multi_turn``: three turns per sample, a tool message between turns,
each GRPO group pinned to one DP rank) with forced retraction. A later turn restores its previous turn's decoded KDA
state, KV and pooled index keys from the radix tree and prefills on from there, so the phase checks exact log-probs
across turns and that later turns hit their previous reply. Every other group ends its earlier turns on a stop token
at a track boundary, so the next turn restores the decode snapshot one interval before that end.

Sampling depends on the seed and position only, so step 1 of both phases scores the same rollouts from the same
weights, which is checked, and the step-1 gradient norms must agree (the backward's reduction order is not pinned,
so not bit for bit).
"""

import math
import os
from pathlib import Path

import torch

from tests.ci.ci_register import register_cuda_ci
from tests.e2e.precision._qwen_gdn_common import grpo_args

from miles.true_on_policy.launch import top_launch
from miles.utils.debug_utils import top_multi_turn
from miles.utils.debug_utils.top_capture import ROLLOUT_DATA, capture_args, check_same_rollouts, fail_on, probe_reward
from miles.utils.debug_utils.top_vjp import vjp_verdict
from miles.utils.external_utils import command_utils

# Nightly, as the other TOP gates. est_time is an upper bound until a /rerun-test run measures it.
register_cuda_ci(
    est_time=3600,
    suite="stage-c-8-gpu-h200",
    labels=["megatron", "precision"],
    hardware=["hopper"],
    num_gpus=8,
    nightly=True,
)

HF_REPO = "CharyZeng/GLM-5.3-Flash-4layer"
# The slice ships no processor config; the full model's applies (as run_glm5_3_flash_lora does).
PROCESSOR_REPO = "zai-org/GLM-5.3-Flash"
MODEL_DIR = "/root/models/GLM-5.3-Flash-4layer"
# The 4-layer args without --moe-router-dtype, which TOP refuses.
MODEL_TYPE = "glm5.3-flash-4layer-top"
NUM_GPUS = 8
TRAINER_GPUS = ENGINE_GPUS = 4
DUMP_DIR = "/root/dumps/glm5_3_flash_4layer_true_on_policy"
GRAD_NORM_DIR = "/root/grad_norm/glm5_3_flash_4layer_true_on_policy"
# Every layer: KDA under the dense MLP, DSA, KDA and DSA under the MoE block (the handoffs between them are checked bit
# for bit).
VJP_LAYERS = (0, 1, 2, 3)
# The phase whose capture the verdict judges.
JUDGED = "ep4"
# Past DSA's 2,048-key budget, so its pooled top-k is selective for the later rows of every sample.
RESPONSE_LEN = 3072
# Engine retraction per phase.
PHASES = {"ep4": False, "retract": True, "multi_turn": True}
# The phases that train on the same step-1 rollouts.
COMPARED = ("ep4", "retract")
RETRACT_INTERVAL = 64
MULTI_TURN_DIR = "/root/dumps/glm5_3_flash_4layer_multi_turn"
# Share of later turns whose radix hit reaches their previous reply's last page; arbitrary margin for a forced
# retraction or an eviction (the Qwen GDN gate's).
MULTI_TURN_HIT_RATE = 0.9
NUM_ROLLOUTS = 2
# Step-1 gradient norms of the two phases; they differ only in reduction order (the Qwen GDN MoE gate's tolerance).
GRAD_NORM_RTOL = 1e-3


def prepare():
    U = command_utils.default_config().create_backend()
    U.exec_command_cpu("mkdir -p /root/models /root/datasets")
    U.exec_command_cpu(f"hf download {HF_REPO} --local-dir {MODEL_DIR}")
    U.exec_command_cpu(f"hf download {PROCESSOR_REPO} processor_config.json --local-dir {MODEL_DIR}")
    U.hf_download_dataset("zhuzilin/dapo-math-17k")


def execute(phase: str, *, dump_dir: str = DUMP_DIR):
    U = command_utils.default_config().create_backend()
    launch = top_launch(model_type="glm5_next", engine_gpus=ENGINE_GPUS)
    os.makedirs(f"{GRAD_NORM_DIR}/{phase}", exist_ok=True)
    perf_args = (
        "--tensor-model-parallel-size 1 "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        f"--expert-model-parallel-size {TRAINER_GPUS} "
        "--expert-tensor-parallel-size 1 "
    )
    # The judged phase captures its layers; every phase runs under the probe reward with unclipped gradients and saves
    # its rollouts, so step 1 sees the same loss and the reported gradient norms are those of a nonzero one. A 4-layer
    # slice earns no rule-based reward; the probe reward also gives step 1 nonzero advantages, so step 2 scores updated,
    # synced weights.
    if phase == JUDGED:
        capture = capture_args(dump_dir=f"{dump_dir}/{phase}", layers=VJP_LAYERS)
    elif phase == "multi_turn":
        capture = (
            f"--custom-generate-function-path {top_multi_turn.__name__}.generate "
            f"--save-debug-rollout-data {MULTI_TURN_DIR}/{{rollout_id}}.pt "
            f"--custom-rm-path {probe_reward.__module__}.{probe_reward.__name__} "
        )
    else:
        capture = (
            f"--save-debug-rollout-data {dump_dir}/{phase}/{ROLLOUT_DATA}/{{rollout_id}}.pt "
            "--clip-grad 0 "
            f"--custom-rm-path {probe_reward.__module__}.{probe_reward.__name__} "
        )
    train_args = (
        f"{grpo_args(model_dir=MODEL_DIR)} "
        f"--rollout-max-response-len {RESPONSE_LEN} "
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
    retract = PHASES[phase]
    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=NUM_GPUS,
        megatron_model_type=MODEL_TYPE,
        extra_env_vars={
            **launch.env_vars,
            **({"SGLANG_TEST_RETRACT": "1", "SGLANG_TEST_RETRACT_INTERVAL": str(RETRACT_INTERVAL)} if retract else {}),
        },
    )


def judge_backward(*, dump_dir: str = DUMP_DIR) -> dict:
    """Rollout identity first; then the verdict on the judged phase's capture and every phase's step gradient norms
    (all reported before any fails). The norms must be finite and nonzero (a zero norm means the probe reward's loss
    reached no parameter), and step 1's must agree across the phases, which train on the same rollouts."""
    check_same_rollouts({phase: f"{dump_dir}/{phase}" for phase in COMPARED})
    verdict = vjp_verdict(
        family="GLM-5.3-Flash", model_path=MODEL_DIR, dump_dir=f"{dump_dir}/{JUDGED}", layers=VJP_LAYERS
    )
    report = dict(verdict=verdict)
    for phase in COMPARED:
        norms = [
            float(torch.load(f"{GRAD_NORM_DIR}/{phase}/actor_{rollout}_0.pt", weights_only=False))
            for rollout in range(NUM_ROLLOUTS)
        ]
        print(f"GRAD_NORMS {phase} {norms}", flush=True)
        failures = [f"{phase} step {i}: grad norm {norm}" for i, norm in enumerate(norms) if not _usable(norm)]
        report[f"grad_norm_{phase}"] = dict(norms=norms, failures=failures)
    norm_failures = [failure for phase in COMPARED for failure in report[f"grad_norm_{phase}"]["failures"]]
    fail_on(verdict=verdict["failures"], grad_norms=norm_failures)
    first, *rest = (report[f"grad_norm_{phase}"]["norms"][0] for phase in COMPARED)
    assert all(math.isclose(norm, first, rel_tol=GRAD_NORM_RTOL) for norm in rest), f"step-1 grad norms: {report}"
    return report


def assert_multi_turn_hits(*, multi_turn_dir: str = MULTI_TURN_DIR) -> dict:
    train_args = top_launch(model_type="glm5_next", engine_gpus=ENGINE_GPUS).train_args
    page_size = int(train_args[train_args.index("--sglang-page-size") + 1])
    track_interval = int(train_args[train_args.index("--sglang-mamba-track-interval") + 1])
    samples = [
        sample
        for path in sorted(Path(multi_turn_dir).glob("*.pt"))
        for sample in torch.load(path, weights_only=False)["samples"]
    ]
    report = top_multi_turn.turn_hit_report(samples, page_size=page_size, track_interval=track_interval)
    print(f"MULTI_TURN_HITS {report}", flush=True)
    assert report["later_turns"] and report["rate"] >= MULTI_TURN_HIT_RATE, report
    assert report["boundary_turns"], report
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
    assert_multi_turn_hits()
