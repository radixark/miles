#!/usr/bin/env bash
# ============================================================================
#  run_glm5_5layer_likefull_2node.sh
#    One-key launcher: GLM-5.2 *5-layer* LoRA RL with the SAME recipe as the
#    FULL 744B model — differing ONLY in PARALLELISM (2 nodes; single-node
#    TP=EP=GPUS_PER_NODE / PP1 / CP1; rollout engine 2 GPU).
#
#    Thin wrapper: exports the full-model recipe env, then delegates to the
#    unified scripts/run_glm5_lora_multinode.sh (which already routes the
#    5-layer prune through the FULL LoRA/rollout config — run_glm5_lora.py
#    hardcodes _is_full=True for ALL GLM-5 models: MLP-target drop, MoE-LoRA,
#    dp-attention / nsa, no reasoning/tool parser). So the only thing this
#    script changes vs the toy default are the recipe knobs set below; the
#    parallelism stays small (the intended exception you asked for).
#
#    WHY: validate the full GLM-5.2 LoRA *recipe + rollout* path at small scale
#    (the 5-layer prune loads in ~seconds vs ~15 min for 744B) before the
#    64-GPU production run, WITHOUT changing recipe semantics.
#
#  ── what matches FULL (recipe) ────────────────────────────────────────────
#    TASK=dapo-math, SEQ=8192, RECOMPUTE=on, SGLANG_MEM_FRACTION=0.70,
#    FP8_ROLLOUT=off (bf16), LORA_RANK=16, WANDB=on, alltoall MoE dispatcher
#    (enforced in code), + everything already shared (target-drop, MoE-LoRA,
#    BACKEND=megatron-bridge, SGLANG_LORA_BACKEND=triton, RESP_LEN, batches…).
#  ── what stays SMALL (parallelism — the exception) ─────────────────────────
#    NODES=2, GPUS_PER_NODE=8 (=TP=EP), PP1/CP1, no EP32 override,
#    rollout-num-gpus-per-engine=2  (all from the underlying *5layer* branch).
#
#  USAGE (same 3-step flow as run_glm5_lora_multinode.sh):
#    # 1) HEAD node — start Ray head, wait for the worker:
#    HEAD_IP=<ip> bash scripts/run_glm5_5layer_likefull_2node.sh head
#    # 2) the 2nd node — join:
#    HEAD_IP=<ip> bash scripts/run_glm5_5layer_likefull_2node.sh worker
#    # 3) HEAD node, once `ray status` shows 2 nodes — launch
#    #    (export WANDB_API_KEY first to match full's WANDB=on, OR set WANDB=offline):
#    HEAD_IP=<ip> WANDB_API_KEY=... bash scripts/run_glm5_5layer_likefull_2node.sh launch
#
#    Override any knob inline, e.g.:
#      GPUS_PER_NODE=4   # the validated toy used 4/node (this is the TP=EP knob)
#      WANDB=offline     # keyless local wandb
#      TASK=gsm8k SEQ= RECOMPUTE=off   # quick keyless smoke instead of the full recipe
# ============================================================================
set -euo pipefail

ROLE="${1:-}"
if [[ "$ROLE" != "head" && "$ROLE" != "worker" && "$ROLE" != "launch" ]]; then
  echo "usage: $0 {head|worker|launch}" >&2
  exit 2
fi

# ── MODEL + PARALLELISM: the ONLY things kept small vs the full run ─────────
export MODEL="GLM-5.2_5layer"                       # 5-layer prune (3 dense + 2 MoE); forced
export NODES="${NODES:-2}"                           # 2-node test (override via NODES=)
export GPUS_PER_NODE="${GPUS_PER_NODE:-8}"           # = TP = EP (single-node 5layer layout); toy validated at 4
# Deliberately NOT set: PARALLEL_EXTRA / ROLLOUT_GPUS_PER_ENGINE. The underlying
# script's *5layer* branch keeps PP1/CP1, EP=GPUS_PER_NODE (no EP32 override) and
# rollout-num-gpus-per-engine=2 — exactly the parallelism exception requested.

# ── RECIPE: aligned to the FULL model (this is what "make it like full" means) ──
export TASK="${TASK:-dapo-math}"                            # full uses dapo-math (toy example used gsm8k)
export SEQ="${SEQ:-8192}"                                   # full seq window
export RECOMPUTE="${RECOMPUTE:-on}"                         # effectively required at seq 8192
export SGLANG_MEM_FRACTION="${SGLANG_MEM_FRACTION:-0.70}"   # full rollout mem-frac (5layer auto would be 0.5)
export FP8_ROLLOUT="${FP8_ROLLOUT:-off}"                    # full default = bf16 rollout
export LORA_RANK="${LORA_RANK:-16}"                         # same as full
export WANDB="${WANDB:-on}"                                 # full default ON (needs WANDB_API_KEY; or WANDB=offline)
# DAPO dynamic sampling is ON by default in the underlying script when TASK=dapo-math.
# alltoall MoE dispatcher needs NO flag here: bridge_lora_helpers.py hardcodes
# provider.moe_token_dispatcher_type="alltoall" for the LoRA actor, so 5layer is alltoall regardless.

# Delegate to the unified launcher (it cd's to the miles root and reads the env above).
cd "$(dirname "$0")"
exec bash run_glm5_lora_multinode.sh "$ROLE"
