"""DeepSeek-V4.1 TOP gate (release FP8): exact log-probs at trainer CP1 x EP4 against an engine with attention DP4 and EP4
over DeepEP, over two GRPO steps, and per-layer gradients against DeepSeek's own inference code.

Vehicle: DeepSeek-V4.1-Flash cut to 8 layers, original [0, 1, 2, 3, 20, 21, 24, 25] renumbered 0-7, with routed experts
0-63 of each layer and of each of the 3 DSpark stages: compress ratios [0, 0, 2, 2, 1, 1, 1, 1], so two SWA-only layers
(layer 1 with Engram), a ratio-2 KV and index source (2) and its consumer (3), a ratio-1 KV, index and candidate source
(4) and its consumer (5), an index-only source reading layer 4's keys under its candidates (6) and its consumer (7).
Dense weights are the release's E4M3 with UE8M0 32x32 block scales, experts its MXFP4, served cast to 128x128 FP8
(lossless); the trainer installs BF16 masters behind serving's bytes straight from the HF checkpoint. Responses run to
3,072 tokens, past the indexer's 512-entry top-k over ratio-1 keys, so selection is sparse in prefill, decode and the
trainer's scoring. The experts are cut to 64 per layer because 384 per layer is ~109B trainable parameters at 8 layers;
the draft keeps no more, since its DeepEP calls map into the target's buffer layout, which a wider layer would overrun.

Phase ``ep4`` trains at CP1 x EP4 (DP4) on 4 GPUs against a 4-GPU engine speculating with the checkpoint's DSpark draft
at block size 5 (recipe (1, 1, 6)), ``--ci-test`` with ``--true-on-policy-mode`` failing on the first unequal log-prob,
so verify scores and draws every emitted token. The slice's draft reads target layers no 40-layer model feeds it and
accepts almost nothing here; the accepted-draft path is qualified on the full checkpoint (serving's decode log-probs
equal a prefill re-score, and the tokens equal non-speculative decode's). V4.1 layers are not recomputed (the TOP block
refuses it), so the run takes ``--use-rollout-logprobs`` from the TOP recipe without its full recompute. It captures
layers 1 (Engram, SWA only), 2 (ratio-2 KV and index source), 4 (ratio-1 KV, index and candidate source) and 6 (index-
only, reading layer 4's latent) for ``miles.utils.debug_utils.top_vjp``, which replays each in DeepSeek's inference
Block at the served weights with the KV rounded as DeepSeek's kernels store it (the math TOP's backward
differentiates), at TOP's routing, index selection and engram rows, with the mHC pre and source latents carried across
layers as TOP carried them, beside the unquantized FP32 reference at the trainer's masters. The V4.1 reference declares
no contract modules yet, so the verdict reports those replays and judges the capture checks (each layer's cotangent,
the handoff between consecutive captured layers, the PP stage boundary included).

Phases ``pp2_ratio2`` and ``pp2_ratio1`` repeat it at trainer PP2 x EP2 (DP2), with the stage boundary after layer 2
(layer 3 reads the ratio-2 source's latent, cache and selection from the first stage) and after layer 4 (layers 5-7
read the ratio-1 source's latent, cache, index K and candidates): what crosses a boundary travels in the pipeline
activation (miles_plugins.top.dsv41_pipeline), and each phase's capture gets its own verdict.

Phase ``radix`` repeats ``ep4`` on agentic sessions (``top_multi_turn``: three turns per sample, a tool message between
turns, each GRPO group pinned to one DP rank) with the engine retracting running requests every 64 decode steps. A
later turn, and a retracted request, resumes from a radix hit on whole 256-token pages and prefills on from it: its
ratio-2 pair ring, compressed KV, index keys and Engram history must give the rows a cold prefill gives, which the
trainer runs once. The phase checks exact log-probs, its own verdict, and that later turns hit their previous reply.
Every other group ends its earlier turns on a 256-token boundary (the probe's interval is SGLang's track-interval
default, which no V4.1 state reads), so the next turn's hit ends one page before the previous reply does.
"""

import os

from tests.ci.ci_register import register_cuda_ci

from miles.true_on_policy.launch import top_launch
from miles.utils.debug_utils import top_multi_turn
from miles.utils.debug_utils.top_capture import capture_args, fail_on, rollout_samples
from miles.utils.debug_utils.top_vjp import vjp_verdict
from miles.utils.external_utils import command_utils

# est_time is an upper bound until a /rerun-test measures it.
register_cuda_ci(
    est_time=9600,
    suite="stage-c-8-gpu-h200",
    labels=["megatron", "precision"],
    hardware=["hopper"],
    num_gpus=8,
    nightly=True,
)

MODEL_NAME = "DeepSeek-V4.1-Flash_8layer-e64-dspark-e64"
MODEL_DIR = f"/root/models/{MODEL_NAME}"
# The VJP's reference: every quantized weight as the BF16 weight serving computes with at the first rollout.
REFERENCE_DIR = f"{MODEL_DIR}-served"
# Every quantized weight as the trainer's BF16 master, for the VJP's contract-bias report.
MASTERS_DIR = f"{MODEL_DIR}-masters"
DUMP_DIR = "/root/dumps/deepseek_v4_1_flash_8layer_true_on_policy"
# Engram + SWA only (1), the ratio-2 KV and index source (2), the ratio-1 KV, index and candidate source (4), and an
# index-only layer reading layer 4's latent under its candidates (6).
VJP_LAYERS = (1, 2, 4, 6)
MODEL_TYPE = "deepseek-v4.1-flash-8layer-e64-top"
NUM_LAYERS = 8
NUM_GPUS = 8
TRAINER_GPUS = ENGINE_GPUS = 4
RESPONSE_LEN = 3072
PROMPT_BUDGET = 512
CONTEXT_LEN = 8192
# (context parallel, pipeline stages, expert parallel, speculate, first stage's layers or None for an even split)
PHASES = {
    "ep4": (1, 1, 4, True, None),
    # Trainer PP2 x EP2: the stage boundary after the ratio-2 source (layer 3 reads layer 2's latent, cache and
    # selection), and after the ratio-1 source (layers 5-7 read layer 4's latent, cache, index K and candidates).
    "pp2_ratio2": (1, 2, 2, True, 3),
    "pp2_ratio1": (1, 2, 2, True, 5),
    # ep4's layout on multi-turn sessions with forced retraction: rows resumed from radix hits.
    "radix": (1, 1, 4, True, None),
}
# Phase ``radix`` retracts running requests every this many decode steps.
RETRACT_INTERVAL = 64
# V4.1's radix page (SGLang's DeepSeek-V4 page size); hits are whole pages of it.
PAGE_SIZE = 256
# Share of later turns whose radix hit reaches their previous reply's last page; arbitrary margin for a forced
# retraction or an eviction.
MULTI_TURN_HIT_RATE = 0.9
# DSpark at block size 5 on the checkpoint's own draft stages: one step of one chain, the anchor and five drafts
# verified as one static chain. The recipe is passed explicitly: TOP admits the serving layout before SGLang resolves
# DSpark's draft tokens from the block size.
DSPARK_BLOCK_SIZE = 5
DSPARK_RECIPE = (1, 1, DSPARK_BLOCK_SIZE + 1)


def stage_layers(phase):
    """The layers of each pipeline stage, for the capture and the verdict (None without PP)."""
    _, pp, _, _, first = PHASES[phase]
    if pp == 1:
        return None
    if pp != 2 or first is None:
        raise NotImplementedError("the gate's PP phases are two stages with an explicit first stage")
    return (first, NUM_LAYERS - first)


def train_args(phase, *, model_dir, prompt_data, grad_norm_dir, optimizer_state_dir, dump_dir):
    """The phase's train.py arguments and extra environment, for CI (execute) and local runs alike."""
    cp, pp, ep, speculate, first = PHASES[phase]
    launch = top_launch(model_type="deepseek_v41", engine_gpus=ENGINE_GPUS, fp8=True)
    ckpt_args = f"--hf-checkpoint {model_dir}/ --ref-load {model_dir}/ "
    rollout_args = (
        f"--prompt-data {prompt_data} "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--num-rollout 2 "
        "--rollout-batch-size 4 "
        "--n-samples-per-prompt 4 "
        f"--rollout-max-response-len {RESPONSE_LEN} "
        "--rollout-temperature 0.8 "
        "--global-batch-size 16 "
    )
    perf_args = (
        "--tensor-model-parallel-size 1 "
        f"--pipeline-model-parallel-size {pp} "
        f"{f'--decoder-first-pipeline-num-layers {first} ' if first else ''}"
        f"--context-parallel-size {cp} "
        f"--expert-model-parallel-size {ep} "
        "--expert-tensor-parallel-size 1 "
        "--use-dynamic-batch-size "
        f"--max-tokens-per-gpu {RESPONSE_LEN + PROMPT_BUDGET} "
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
        "--stream-optimizer-state-to-disk "
        "--stream-optimizer-state-moment-dtype bf16 "
        f"--offload-train-disk-dir {optimizer_state_dir}/{phase} "
    )
    sglang_args = (
        f"--rollout-num-gpus {ENGINE_GPUS} --rollout-num-gpus-per-engine {ENGINE_GPUS} "
        "--sglang-mem-fraction-static 0.7 "
        f"--sglang-context-length {CONTEXT_LEN} "
        # validate_dsv41: each DP rank's prefill chunk covers the context, so no prefill is split.
        f"--sglang-chunked-prefill-size {CONTEXT_LEN * ENGINE_GPUS} "
        # DeepEP's low-latency decode dispatches at most 128 tokens per rank: a verify step scores 6 rows a request.
        f"--sglang-cuda-graph-max-bs-decode {16 if speculate else 32} "
    )
    if speculate:
        steps, topk, draft = DSPARK_RECIPE
        sglang_args += (
            f"--sglang-speculative-algorithm DSPARK --sglang-speculative-dspark-block-size {DSPARK_BLOCK_SIZE} "
            f"--sglang-speculative-num-steps {steps} --sglang-speculative-eagle-topk {topk} "
            f"--sglang-speculative-num-draft-tokens {draft} "
        )
    misc_args = (
        "--attention-dropout 0.0 --hidden-dropout 0.0 "
        f"--actor-num-nodes 1 --actor-num-gpus-per-node {TRAINER_GPUS} "
        f"--ci-save-grad-norm {grad_norm_dir}/{phase}/{{role}}_{{rollout_id}}_{{step_id}}.pt "
        # The probe reward, unclipped gradients and the per-layer capture for top_vjp.
        f"{capture_args(dump_dir=dump_dir, layers=VJP_LAYERS, stage_layers=stage_layers(phase))}"
        "--use-rollout-logprobs "
        # The equality checker scrambles every engine tensor and expects the sync to rewrite it; V4.1's frozen
        # serving state (the 98 GB host engram tables, the untrained indexers, the DSpark stages) is never synced.
        # Step 2's log-prob equality is what checks the synced weights.
        "--ci-disable-weight-update-checker "
    )
    env = dict(launch.env_vars)
    if phase == "radix":
        misc_args += f"--custom-generate-function-path {top_multi_turn.__name__}.generate "
        env |= {"SGLANG_TEST_RETRACT": "1", "SGLANG_TEST_RETRACT_INTERVAL": str(RETRACT_INTERVAL)}
    args = (
        f"{ckpt_args} {rollout_args} {optimizer_args} {grpo_args} {perf_args} {sglang_args} "
        f"--ci-test {misc_args} {' '.join(launch.train_args)} "
    )
    return args, env


def assert_turn_hits(dump_dir):
    samples = [sample for rollout in (0, 1) for sample in rollout_samples(dump_dir, rollout)]
    report = top_multi_turn.turn_hit_report(samples, page_size=PAGE_SIZE, track_interval=PAGE_SIZE)
    print(f"MULTI_TURN_HITS {report}", flush=True)
    assert report["later_turns"] and report["rate"] >= MULTI_TURN_HIT_RATE, report
    assert report["boundary_turns"], report


def prepare():
    U = command_utils.default_config().create_backend()
    for weights, save_dir in (("served", REFERENCE_DIR), ("masters", MASTERS_DIR)):
        U.exec_command_gpu_once(
            f"python tools/convert_dsv41_to_bf16_reference.py --weights {weights} --model-dir {MODEL_DIR} "
            f"--save-dir {save_dir}",
            save_dir,
        )


def execute(phase: str):
    U = command_utils.default_config().create_backend()
    args, env = train_args(
        phase,
        model_dir=MODEL_DIR,
        prompt_data="/root/datasets/dapo-math-17k/dapo-math-17k.jsonl",
        grad_norm_dir="/root/grad_norm/deepseek_v4_1_flash_8layer_true_on_policy",
        optimizer_state_dir="/root/optimizer_state/deepseek_v4_1_flash_8layer_true_on_policy",
        dump_dir=f"{DUMP_DIR}/{phase}",
    )
    os.makedirs("/root/grad_norm/deepseek_v4_1_flash_8layer_true_on_policy", exist_ok=True)
    U.execute_train(
        train_args=f"{args} {command_utils.get_default_wandb_args(__file__)}",
        num_gpus_per_node=NUM_GPUS,
        megatron_model_type=MODEL_TYPE,
        extra_env_vars=env,
    )


if __name__ == "__main__":
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    prepare()
    for phase in PHASES:
        execute(phase)
        verdict = vjp_verdict(
            family="DeepSeek-V4.1",
            model_path=REFERENCE_DIR,
            dump_dir=f"{DUMP_DIR}/{phase}",
            layers=VJP_LAYERS,
            kv_cache_dtype=top_launch(model_type="deepseek_v41", engine_gpus=ENGINE_GPUS, fp8=True).kv_cache_dtype,
            masters_path=MASTERS_DIR,
            stage_layers=stage_layers(phase),
        )
        fail_on(verdict=verdict["failures"])
        if phase == "radix":
            assert_turn_hits(f"{DUMP_DIR}/{phase}")
