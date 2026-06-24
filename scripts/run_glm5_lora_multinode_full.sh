#!/usr/bin/env bash
# ============================================================================
#  run_glm5_lora_multinode_full.sh
#    PRESET wrapper around scripts/run_glm5_lora_multinode.sh that pins the
#    VALIDATED full-scale GLM-5.2 (744B, 78L) LoRA-RL e2e recipe on 8 nodes x
#    8 H200 = 64 GPU. This is the exact config used to bring the e2e up
#    (rollout -> train -> save) on 2026-06-24; every knob below is a default
#    you can still override from the environment.
#
#    It does NOT reimplement anything -- it just exports the validated knobs
#    and delegates to run_glm5_lora_multinode.sh {head|worker|launch}.
#
#  ─────────────────────────────────────────────────────────────────────────
#  WHY THESE DEFAULTS (hard-won; see the per-knob notes):
#   * BACKEND=slime  -> the FUSED TileLang DSA backend. The unfused
#       (megatron-bridge) backend OOMs on GPU at seq>=~8192: its mcore DSA
#       indexer `_compute_index_scores` materialises a [s,b,h,t] fp32 score
#       tensor (O(seq^2), ~8 GiB at seq 8192) and CP>1 is rejected
#       ("Currently context parallelism is not supported by DSAttention") and
#       expandable_segments is incompatible with torch_memory_saver. The fused
#       lighting_indexer never materialises that tensor -> no GPU OOM.
#   * R3=off  -> drops --use-rollout-routing-replay AND (slime) the DSA
#       --use-rollout-indexer-replay. The indexer replay makes sglang allocate
#       an IndexerTopkCapturer HOST pinned buffer sized (max_total_num_tokens,
#       num_layers, index_topk=2048) int32 ~= 78-128 GB/rank * 8 ranks/node ->
#       blows the ~1.78 TB pod cgroup -> RolloutManager host-OOM (SIGTERM) ->
#       cluster-wide sglang crash. With R3 off the capturer is never built and
#       --sglang-max-total-tokens can stay at the sglang default for full
#       rollout throughput. Trade-off: rollout<->train is off-policy in the
#       sparse-attn / MoE-routing dims -- fine for e2e bring-up; set R3=on for a
#       correctness-faithful RL run (then ALSO cap --sglang-max-total-tokens,
#       e.g. SGLANG_MAX_TOTAL_TOKENS via PARALLEL_EXTRA, so the host buffer fits).
#   * SEQ=1024 / RESP_LEN=512  -> fast e2e bring-up (short responses). Bump to
#       SEQ=8192 RESP_LEN=7168 for the real run (fits host once R3=off).
#   * EP32 / PP1 / CP1  -> the only validated full-model layout. PP>1 trips the
#       GLM-5.2 cross-layer-DSA build assert (a PP stage must START on a compute
#       layer, freq=4 index-share group); CP>1 is unsupported by DSAttention.
#   * --lora-base-cpu-backup is OFF (default) -- its sglang host mirror
#       (~372 GB/node) is extra host pressure and not needed.
#
#  OPS GOTCHAS learned during bring-up (NOT encoded here -- do them by hand):
#   * If a worker pod OOM-restarts, its CONTAINER-LOCAL pip editable installs
#     revert to the image (sglang -> /sgl-workspace/sglang, miles -> /root/miles)
#     even though /personal/* persists on the PVC. Symptom: actor import fails
#     with "cannot import name 'ParallelismContext' from sglang...". FIX: re-run
#       pip install -e /personal/sglang/python -e /personal/miles \
#                      -e /personal/Megatron-Bridge --no-build-isolation --no-deps
#     on the affected ranks, THEN relaunch.
#   * If the Ray cluster has < 64 GPU (dead raylets from earlier crashes), the
#     64-GPU PACK placement group hangs forever at "Creating placement group".
#     Reform a CLEAN cluster (ray stop --force on all ranks, then head + workers)
#     before launching. Verify `ray status` shows /64.0 GPU and 8 ALIVE nodes.
#  ─────────────────────────────────────────────────────────────────────────
#
#  USAGE (same head/worker/launch flow as run_glm5_lora_multinode.sh):
#    # HEAD:
#    HEAD_IP=<ip> bash scripts/run_glm5_lora_multinode_full.sh head
#    # each of the other 7 workers:
#    HEAD_IP=<ip> bash scripts/run_glm5_lora_multinode_full.sh worker
#    # HEAD, once `ray status` shows 8 nodes / 64 GPU:
#    export WANDB_API_KEY=...                 # NEVER hardcode the key
#    HEAD_IP=<ip> bash scripts/run_glm5_lora_multinode_full.sh launch
#
#    # real run (long context) instead of fast e2e:
#    SEQ=8192 RESP_LEN=7168 R3=on \
#      HEAD_IP=<ip> bash scripts/run_glm5_lora_multinode_full.sh launch
# ============================================================================
set -euo pipefail

ROLE="${1:-}"
if [[ "$ROLE" != "head" && "$ROLE" != "worker" && "$ROLE" != "launch" ]]; then
  echo "usage: $0 {head|worker|launch}   (see header)" >&2
  exit 2
fi

# ----- VALIDATED full-scale GLM-5.2 LoRA-RL e2e defaults (all overridable) -----
export MODEL="${MODEL:-GLM-5.2}"                 # full 78-layer 744B
export NODES="${NODES:-8}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
export BACKEND="${BACKEND:-slime}"               # FUSED DSA (see header)
export R3="${R3:-off}"                           # drop replay -> no host capturer buffer
export LORA_RANK="${LORA_RANK:-16}"
export SEQ="${SEQ:-4096}"                        # dapo-math sweet spot (rollout-only raw_reward sweep:
                                                 #   1024=0.0, 2048=0.125, 4096=0.25, 8192=0.3125 -- diminishing
                                                 #   returns past 4096 at 2x rollout time). Bump SEQ=8192 for max signal.
export RESP_LEN="${RESP_LEN:-3584}"              # response budget within seq 4096 (leaves ~512 for the prompt)
export TASK="${TASK:-dapo-math}"
export NUM_ROLLOUT="${NUM_ROLLOUT:-50}"
export HF_CHECKPOINT="${HF_CHECKPOINT:-/cluster-storage/models/${MODEL}}"
# TP8 x DP8 / PP1 / CP1 / EP32 -- the only validated full-model layout.
export PARALLEL_EXTRA="${PARALLEL_EXTRA:---pipeline-model-parallel-size 1 --context-parallel-size 1 --expert-model-parallel-size 32}"
# wandb: default ON; KEY MUST come from the env (run_glm5_lora_multinode.sh
# fails fast if WANDB=on and WANDB_API_KEY is unset). Never hardcode the key here.
export WANDB="${WANDB:-on}"
export WANDB_PROJECT="${WANDB_PROJECT:-miles-glm52-8node}"

# ----- editable installs (SELF-HEAL the pod-restart-wipes-install gotcha) -----
# A worker pod that OOM-restarts loses its CONTAINER-LOCAL pip editable installs
# and reverts to the image's (sglang -> /sgl-workspace/sglang, miles -> /root/miles),
# even though /personal/* persists on the PVC. That makes the actor import fail with
# "cannot import name 'ParallelismContext' from sglang...". So before bringing a node
# up we (idempotently) re-point the editable installs at /personal/*. These are
# already built in-place on the PVC, so --no-build-isolation --no-deps just rewrites
# the .pth (fast, no recompile). Skip with SKIP_INSTALL=1.
PERSONAL_DIR="${PERSONAL_DIR:-/personal}"
SKIP_INSTALL="${SKIP_INSTALL:-0}"
ensure_editable_installs() {
  echo "[full] ensuring editable installs point at ${PERSONAL_DIR} (sglang/miles/Megatron-Bridge)..."
  pip install \
    -e "${PERSONAL_DIR}/sglang/python" \
    -e "${PERSONAL_DIR}/miles" \
    -e "${PERSONAL_DIR}/Megatron-Bridge" \
    --no-build-isolation --no-deps 2>&1 | grep -iE "Successfully installed|error" | tail -3 || true
  echo "[full] sglang -> $(pip show sglang 2>/dev/null | grep -i 'Editable project location' || echo '?')"
}
# Run on every role (head/worker bring-up needs it; harmless+fast on launch).
if [[ "$SKIP_INSTALL" != "1" ]]; then
  ensure_editable_installs
fi

echo "[full] role=${ROLE} MODEL=${MODEL} NODES=${NODES}x${GPUS_PER_NODE} BACKEND=${BACKEND} R3=${R3} SEQ=${SEQ} RESP_LEN=${RESP_LEN} WANDB=${WANDB}"
echo "[full] PARALLEL_EXTRA=${PARALLEL_EXTRA}"

exec bash "$(dirname "$0")/run_glm5_lora_multinode.sh" "${ROLE}"
