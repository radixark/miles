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
#   * R3=on (default)  -> adds ONLY --use-rollout-routing-replay (rollout MoE top-8
#       replayed in training for on-policy parity; cheap sglang routed-experts capturer
#       ~0.5 GB/rank). It does NOT add --use-rollout-indexer-replay: that DSA indexer
#       top-k replay is a DEBUG aid (the slime kernel recomputes the top-k, so training
#       does not need it) and it was the killer -- it makes sglang allocate an
#       IndexerTopkCapturer HOST pinned buffer (max_total_num_tokens, num_layers, 2048)
#       int32 ~= 78-128 GB/rank * 8 ranks/node -> blows the ~1.78 TB pod cgroup ->
#       RolloutManager host-OOM (SIGTERM) -> cluster-wide sglang crash. With indexer
#       replay gone, --sglang-max-total-tokens can stay at the sglang default for full
#       rollout throughput. (R3=off drops routing replay too -> fully off-policy; only
#       for quick mechanics bring-up.)
#   * RESP_LEN=4096 (dapo) -> reward/throughput sweet spot (rollout-only raw_reward sweep:
#       1024=0.0, 2048=0.125, 4096=0.25, 8192=0.3125; >4096 = 2x time for +0.06). SEQ is left
#       UNSET (window auto-derives from model config); set SEQ=<n> only to cap memory for colocate.
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
#    # longer responses / cap the window for colocate memory:
#    RESP_LEN=8192 SEQ=10240 R3=on \
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
export R3="${R3:-on}"                             # R3 ON = rollout ROUTING replay only (on-policy MoE
                                                  # parity, cheap ~0.5GB/rank). The DSA INDEXER replay is
                                                  # NOT added (it's debug-only + was the ~78-128GB/rank host
                                                  # buffer that OOM'd the pod) -- see run_glm5_lora.py r3_args.
# --lora-base-cpu-backup ON is REQUIRED for correct colocate LoRA, not just a perf knob. With it OFF,
# skip_base_sync=False -> megatron RE-SHIPS the full base to sglang every update_weights, and the bridge
# export mis-maps the GLM-5.2 slime base -> sglang serves a CORRUPTED base -> rollout!=train policy ->
# degenerate rollout (reward 0). PROVEN on the 5-layer A/B: train_rollout_kl 1.04 (off) -> 0.0004 (on).
# ON => skip_base_sync=True, no re-ship, sglang serves its own correctly-loaded base + a host mirror
# (~372 GB/node) across pause/resume. Host fits now that R3 is routing-only (no indexer host buffer).
export LORA_BASE_CPU_BACKUP="${LORA_BASE_CPU_BACKUP:-on}"
export LORA_RANK="${LORA_RANK:-16}"
export TASK="${TASK:-dapo-math}"                 # dataset: dapo-math | gsm8k
# RESP_LEN (--rollout-max-response-len) default per TASK. SEQ is left UNSET on purpose, so
# --seq-length / --rollout-max-context-len are omitted and the total window auto-derives from the
# model config (set SEQ=<n> in the env only to cap window/memory for colocate).
#   dapo-math -> 4096  (long-CoT competition math; >2048 seq makes the GLM-5.2 DSA indexer go SPARSE)
#   gsm8k     -> 256   (short-answer grade-school math; dense DSA -> fast smoke)
if [[ "$TASK" == "gsm8k" ]]; then
  export RESP_LEN="${RESP_LEN:-256}"
else
  export RESP_LEN="${RESP_LEN:-4096}"
fi
# Dynamic sampling (check_reward_nonzero_std filter) DEFAULT OFF here. At the current reward /
# truncation levels (seq 4096 dapo: raw_reward ~0.25, ~75% truncated -> many groups are all-zero
# = zero-std), the filter rejects most groups and RESAMPLES indefinitely, so rollout 0 never
# fills a batch and training never starts. Off => rollout uses the fixed batch and proceeds to
# train every step (GRPO still learns from the mixed-reward groups). Re-enable (DAPO_DYNAMIC_SAMPLING=on)
# once rewards are dense enough (longer seq / a partially-trained policy) that zero-std groups are rare.
export DAPO_DYNAMIC_SAMPLING="${DAPO_DYNAMIC_SAMPLING:-off}"
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

echo "[full] role=${ROLE} MODEL=${MODEL} NODES=${NODES}x${GPUS_PER_NODE} BACKEND=${BACKEND} R3=${R3} SEQ=${SEQ:-<auto>} RESP_LEN=${RESP_LEN} WANDB=${WANDB}"
echo "[full] PARALLEL_EXTRA=${PARALLEL_EXTRA}"

exec bash "$(dirname "$0")/run_glm5_lora_multinode.sh" "${ROLE}"
