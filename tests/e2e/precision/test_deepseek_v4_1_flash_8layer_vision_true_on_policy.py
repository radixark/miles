"""DeepSeek-V4.1 TOP gate with images: exact log-probs of image-conditioned samples from an attention-DP4 x EP4 engine
over DeepEP, at trainer EP4 and PP2 x EP2, and with the engine speculating.

Vehicle: the text gate's 8-layer, 64-expert slice (``test_deepseek_v4_1_flash_8layer_true_on_policy``) with the
release's whole vision tower, aligner, image boundary embeddings and every kept layer's VL routing bias added back
(``derive_vision_slice``). The prompts are drawn images (``write_image_prompts``): the release's renderer turns each
into one ``<｜deepseek_image｜>``, SGLang's token-space processor expands it to the image's whole span and preprocesses
on the CPU, and each trainer rank rebuilds the same BF16 patches. On both sides the frozen tower runs the shared TOP
tower program (``sglang.srt.true_on_policy.vision``) and its span rows replace the span's token embeddings; every row
routes through the VL-biased gate (image rows by the VL bias), and Engram leaves image rows' streams as they came.

Phase ``ep4`` trains at CP1 x EP4 (DP4) against an engine without speculation; phase ``pp2`` trains on two pipeline
stages at EP2, the boundary after the ratio-1 source (layer 4), where only the first stage writes image rows. Both run
``--ci-test`` with ``--true-on-policy-mode``, which fails on the first unequal log-prob over two GRPO steps; they score
the same rollouts, which is checked, and their step-1 gradient norms must agree. Phase ``spec`` trains at EP4 with the
engine speculating through the checkpoint's DSpark draft (recipe (1, 1, 6)): the tower runs at prefill only and verify
makes every emitted token draft-independent, so the log-probs must stay exact. The per-layer backward is the text
gate's: images change which experts a row takes and which rows Engram gates, not the differentiated program.
"""

import json
import math
import os
import re
from collections import defaultdict

import torch

from tests.ci.ci_register import register_cuda_ci
from tests.e2e.precision._qwen_gdn_common import vision_args, write_image_prompts

from miles.true_on_policy.launch import top_launch
from miles.utils.debug_utils.top_capture import ROLLOUT_DATA, check_same_rollouts, probe_reward
from miles.utils.external_utils import command_utils

# Nightly, as the other TOP gates. est_time is an upper bound until a /rerun-test run measures it.
register_cuda_ci(
    est_time=7200,
    suite="stage-c-8-gpu-h200",
    labels=["megatron", "precision"],
    hardware=["hopper"],
    num_gpus=8,
    nightly=True,
)

FULL_MODEL_DIR = "/root/models/DeepSeek-V4.1-Flash"
TEXT_MODEL_DIR = "/root/models/DeepSeek-V4.1-Flash_8layer-e64-dspark-e64"
MODEL_DIR = f"{TEXT_MODEL_DIR}-vision"
MODEL_TYPE = "deepseek-v4.1-flash-8layer-e64-top"
DATA_DIR = "/root/datasets/top_vision_drawings"
DUMP_DIR = "/root/dumps/deepseek_v4_1_flash_8layer_vision_true_on_policy"
GRAD_NORM_DIR = "/root/grad_norm/deepseek_v4_1_flash_8layer_vision_true_on_policy"
NUM_GPUS = 8
TRAINER_GPUS = ENGINE_GPUS = 4
# The text vehicle's kept layers, in its order, and its routed experts per layer (cut_v41_slice).
KEPT_LAYERS = (0, 1, 2, 3, 20, 21, 24, 25)
EXPERTS = 64
RESPONSE_LEN = 1024
# Two drawings of up to 1,024 span tokens each, and the prompt text.
PROMPT_BUDGET = 2560
CONTEXT_LEN = 8192
# Pipeline stages, the first stage's layers (None without PP) and whether the engine speculates.
PHASES = {
    "ep4": dict(pp=1, first=None, speculate=False),
    "pp2": dict(pp=2, first=5, speculate=False),
    "spec": dict(pp=1, first=None, speculate=True),
}
# The phases that train on the same rollouts.
COMPARED = ("ep4", "pp2")
# Step-1 gradient norms of the compared phases; they differ only in reduction order.
GRAD_NORM_RTOL = 1e-3
DSPARK_BLOCK_SIZE = 5
DSPARK_RECIPE = (1, 1, DSPARK_BLOCK_SIZE + 1)


def derive_vision_slice(full: str, text: str, dst: str) -> None:
    """``dst``: the text vehicle (shards hard-linked) plus one shard of the release's ``vision.*``, ``aligner.*`` and
    ``image_*`` tensors and the VL bias of each kept layer and DSpark stage, cut to the vehicle's experts; its config
    gains the release's ``vision_config``."""
    from safetensors import safe_open
    from safetensors.torch import save_file

    def target(name):
        if name.startswith(("vision.", "aligner.", "image_")) or re.fullmatch(r"mtp\.\d+\.ffn\.gate\.bias_vl", name):
            return name
        m = re.fullmatch(r"layers\.(\d+)\.ffn\.gate\.bias_vl", name)
        if m and int(m.group(1)) in KEPT_LAYERS:
            return f"layers.{KEPT_LAYERS.index(int(m.group(1)))}.ffn.gate.bias_vl"
        return None

    os.makedirs(dst, exist_ok=False)
    weight_map = dict(json.load(open(f"{text}/model.safetensors.index.json"))["weight_map"])
    for name in os.listdir(text):
        if name not in ("model.safetensors.index.json", "config.json") and os.path.isfile(f"{text}/{name}"):
            os.link(f"{text}/{name}", f"{dst}/{name}")
    by_file = defaultdict(list)
    for name, file in json.load(open(f"{full}/model.safetensors.index.json"))["weight_map"].items():
        if (t := target(name)) is not None:
            by_file[file].append((name, t))
    tensors = {}
    for file, names in sorted(by_file.items()):
        with safe_open(f"{full}/{file}", "pt") as f:
            for name, t in names:
                value = f.get_tensor(name)
                tensors[t] = value[:EXPERTS].contiguous() if t.endswith("bias_vl") else value
    save_file(tensors, f"{dst}/model-vision.safetensors", metadata={"format": "pt"})
    weight_map |= {name: "model-vision.safetensors" for name in tensors}
    total = sum(os.path.getsize(f"{dst}/{f}") for f in set(weight_map.values()))
    json.dump(
        {"metadata": {"total_size": total}, "weight_map": dict(sorted(weight_map.items()))},
        open(f"{dst}/model.safetensors.index.json", "w"),
        indent=2,
    )
    config = json.load(open(f"{text}/config.json"))
    config["vision_config"] = json.load(open(f"{full}/config.json"))["vision_config"]
    json.dump(config, open(f"{dst}/config.json", "w"), indent=2)


def prepare():
    if not os.path.exists(MODEL_DIR):
        derive_vision_slice(FULL_MODEL_DIR, TEXT_MODEL_DIR, MODEL_DIR)
    write_prompts()


def write_prompts():
    write_image_prompts(DATA_DIR)


def execute(phase: str):
    pp, first, speculate = PHASES[phase]["pp"], PHASES[phase]["first"], PHASES[phase]["speculate"]
    U = command_utils.default_config().create_backend()
    launch = top_launch(model_type="deepseek_v41", engine_gpus=ENGINE_GPUS, fp8=True)
    os.makedirs(f"{GRAD_NORM_DIR}/{phase}", exist_ok=True)
    rollout_args = (
        f"--hf-checkpoint {MODEL_DIR}/ --ref-load {MODEL_DIR}/ "
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
        f"{vision_args(data_dir=DATA_DIR)}"
    )
    perf_args = (
        "--tensor-model-parallel-size 1 "
        f"--pipeline-model-parallel-size {pp} "
        f"{f'--decoder-first-pipeline-num-layers {first} ' if first else ''}"
        "--context-parallel-size 1 "
        f"--expert-model-parallel-size {TRAINER_GPUS // pp} "
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
        "--lr 1e-5 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
        "--stream-optimizer-state-to-disk "
        "--stream-optimizer-state-moment-dtype bf16 "
        f"--offload-train-disk-dir {DUMP_DIR}/optimizer_state/{phase} "
    )
    sglang_args = (
        f"--rollout-num-gpus {ENGINE_GPUS} --rollout-num-gpus-per-engine {ENGINE_GPUS} "
        "--sglang-mem-fraction-static 0.7 "
        f"--sglang-context-length {CONTEXT_LEN} "
        # validate_dsv41: each DP rank's prefill chunk covers the context, so no prefill is split.
        f"--sglang-chunked-prefill-size {CONTEXT_LEN * ENGINE_GPUS} "
        # DeepEP's low-latency decode dispatches at most 128 tokens per rank: a verify step scores 6 rows a request.
        f"--sglang-cuda-graph-max-bs-decode {16 if speculate else 32} "
        # validate_dsv41_radix: radix reuse with the vision tower loaded is not qualified.
        "--sglang-disable-radix-cache "
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
        f"--ci-save-grad-norm {GRAD_NORM_DIR}/{phase}/{{role}}_{{rollout_id}}_{{step_id}}.pt "
        f"--save-debug-rollout-data {DUMP_DIR}/{phase}/{ROLLOUT_DATA}/{{rollout_id}}.pt "
        # Unclipped gradients under the probe reward: step 1 of the compared phases sees one loss.
        "--clip-grad 0 "
        f"--custom-rm-path {probe_reward.__module__}.{probe_reward.__name__} "
        "--use-rollout-logprobs "
        # As the text gate: V4.1's frozen serving state is never synced; step 2's log-probs check the synced weights.
        "--ci-disable-weight-update-checker "
    )
    U.execute_train(
        train_args=(
            f"{rollout_args} {optimizer_args} {grpo_args} {perf_args} {sglang_args} --ci-test {misc_args} "
            f"{command_utils.get_default_wandb_args(__file__)} {' '.join(launch.train_args)} "
        ),
        num_gpus_per_node=NUM_GPUS,
        megatron_model_type=MODEL_TYPE,
        extra_env_vars=dict(launch.env_vars),
    )


def assert_backward_matches():
    norms = {
        # The last pipeline stage saves its norm on its own device; read every norm on the CPU.
        phase: float(torch.load(f"{GRAD_NORM_DIR}/{phase}/actor_0_0.pt", map_location="cpu", weights_only=False))
        for phase in COMPARED
    }
    print(f"GRAD_NORMS {norms}", flush=True)
    assert norms["ep4"] > 0 and all(
        math.isclose(norm, norms["ep4"], rel_tol=GRAD_NORM_RTOL) for norm in norms.values()
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
