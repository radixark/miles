#!/usr/bin/env bash
# ============================================================================
#  run_glm5_lora_multinode_full_model.sh
#    Multi-node GLM-5.2 *FULL* (glm5.2-744B-A40B, 78 layers) LoRA RL launcher.
#    A thin wrapper around scripts/run_glm5_lora.py that drives the multi-node
#    flow, derived 1:1 from scripts/run_glm5_lora_multinode.sh.
#
#    Validate first with NODES=4 (smoke), then run NODES=8 (production). NODES
#    is the single top-of-file knob you flip between the two (default 8).
#
#  ─────────────────────────────────────────────────────────────────────────
#  ⚠ READ THIS BEFORE THE FIRST RUN — three things are UNVALIDATED at 744B:
#
#   1. PARALLELISM. run_glm5_lora.py:_get_parallel_config is HARD-CODED
#      single-node: TP=EP=GPUS_PER_NODE, PP=1, CP=1. That fits the 5-layer toy
#      but the FULL 78-layer / 256-expert / 744B model very likely will NOT fit
#      with PP=1/CP=1 — the canonical full recipe (run_glm5_744b_a40b.py, >=16
#      nodes) uses PP=4 + CP=2 + EP=32 + --decoder-last-pipeline-num-layers 18
#      + --allgather-cp. This wrapper exposes a PARALLEL_EXTRA hook so you can
#      override PP/CP/EP/decoder-split via --extra-args (argparse last-wins
#      beats the auto-emitted --pipeline-model-parallel-size 1 etc.), but the
#      exact 8-node (64-GPU) layout for the full model is NOT pre-validated in
#      this repo. Confirm a fitting layout before trusting the defaults.
#      NOTE: PP>1 may trip the GLM-5.2 cross-layer-DSA build-time PP-split
#      assert; CP>1 + --allgather-cp is the safer scaling lever.
#
#   2. SEQ=8192. There is NO --seq-length flag on run_glm5_lora.py. We wire it
#      via --extra-args "--seq-length 8192 --rollout-max-context-len 8192" plus
#      the script flag --rollout-max-response-len. With static micro-batch-size
#      1 (forced by the DSA bshd path; --use-dynamic-batch-size is OFF-LIMITS
#      here) an 8192 window on the full model is memory-heavy — recompute is
#      effectively required (see RECOMPUTE knob, on by default).
#
#   3. ROLLOUT. sglang does NOT yet serve the GLM-5.2 cross-layer subset-indexer
#      checkpoint, so a live colocate rollout->train loop is BLOCKED for full
#      GLM-5.2. If your cluster's sglang still lacks GLM-5.2 serving, set
#      TRAIN_ONLY=on with DUMP_ROLLOUT=<path to a GLM-5.1 rollout dump .pt> to
#      run the train-only replay path instead. Verify sglang GLM-5.2 support on
#      the target cluster before committing to a live 8-node run.
#  ─────────────────────────────────────────────────────────────────────────
#
#  WHY THIS WRAPPER EXISTS (same as run_glm5_lora_multinode.sh)
#    run_glm5_lora.py is single-node by design: _train() hardcodes
#    `--actor-num-nodes 1`, and U.execute_train only does a local
#    `ray start --head`. This wrapper (a) forms a Ray cluster across the nodes
#    MANUALLY (head + workers), and (b) submits with MILES_SCRIPT_EXTERNAL_RAY=1
#    so miles reuses that cluster, overriding --actor-num-nodes via --extra-args
#    (argparse last-wins: extra_args is appended after misc_args).
#
#  ⚠ CRITICAL — set GPUS_PER_NODE to the *REAL* per-node GPU count.
#    miles' rollout addr-allocator uses --num-gpus-per-node to place sglang
#    engines per node; if it disagrees with reality, worker-node engines block
#    on a 600s cross-node TCPStore timeout and die. We pass GPUS_PER_NODE in
#    BOTH the script flag AND --extra-args for this reason.
#
#  PREREQUISITES
#    * Shared FS for code + checkpoints (e.g. /personal) reachable by all nodes.
#    * The FULL GLM-5.2 HF checkpoint present at HF_CHECKPOINT on EVERY node
#      (huge ~744B; `prepare` downloads zai-org/GLM-5.2 head-node-only).
#    * dapo-math dataset prepared: see PREREQ COMMANDS below.
#    * Editable installs on every node (sglang, Megatron-Bridge `bridge` branch,
#      miles). Inter-node TCP connectivity; inter-node NIC (NCCL_IFNAME).
#    * WANDB_API_KEY exported (this script FAILS FAST if unset & WANDB!=off).
#
#  USAGE  (run from a miles checkout; this script cd's to the miles root)
#    # ---- SMOKE TEST FIRST: NODES=4 ----
#    # 1) HEAD node — start Ray head, wait for workers:
#    HEAD_IP=10.220.51.62 GPUS_PER_NODE=8 NODES=4 \
#      bash scripts/run_glm5_lora_multinode_full_model.sh head
#    # 2) EACH of the other 3 nodes — join:
#    HEAD_IP=10.220.51.62 GPUS_PER_NODE=8 \
#      bash scripts/run_glm5_lora_multinode_full_model.sh worker
#    # 3) HEAD node, once `ray status` shows 4 nodes — launch:
#    export WANDB_API_KEY=...                     # REQUIRED (or `wandb login` then export)
#    HEAD_IP=10.220.51.62 GPUS_PER_NODE=8 NODES=4 \
#      bash scripts/run_glm5_lora_multinode_full_model.sh launch
#
#    # ---- PRODUCTION: NODES=8 ---- (same 3 steps, NODES=8, 7 worker nodes)
#
#    # Monitor (head):
#    RAY_ADDRESS=http://$HEAD_IP:8265 ray job logs   $JOB_ID --follow
#    RAY_ADDRESS=http://$HEAD_IP:8265 ray job status $JOB_ID
# ============================================================================
set -euo pipefail

ROLE="${1:-}"
if [[ "$ROLE" != "head" && "$ROLE" != "worker" && "$ROLE" != "launch" ]]; then
  echo "usage: $0 {head|worker|launch}   (see header for the full flow)" >&2
  exit 2
fi

# cd to the miles repo root so `scripts/run_glm5_lora.py` resolves.
cd "$(dirname "$0")/.."

# ============================================================================
#  TOP-OF-FILE KNOBS — the things you actually flip between runs.
# ============================================================================
NODES="${NODES:-8}"                            # node count: 4 for the smoke test, 8 for production
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"            # REAL GPUs per node (==> TP=EP=this intra-node)
MODEL="${MODEL:-GLM-5.2}"                       # FULL model -> megatron_model_type glm5.2-744B-A40B (78 layers)
BACKEND="${BACKEND:-megatron-bridge}"          # unfused DSA path (--dsa-attention-backend); "slime" = fused
LORA_RANK="${LORA_RANK:-16}"                    # LoRA rank (alpha=32, dropout=0.0 set by run_glm5_lora.py)
SEQ="${SEQ:-8192}"                              # target sequence window (see SEQ note below; not a native flag)
TASK="${TASK:-dapo-math}"                       # dataset: dapo-math | gsm8k
# ----------------------------------------------------------------------------

# Back-compat alias: the base wrapper used NUM_NODES; keep both in sync.
NUM_NODES="${NUM_NODES:-$NODES}"

# ----- secondary knobs -----
RAY_PORT="${RAY_PORT:-6379}"                   # Ray GCS port (head)
DASH_PORT="${DASH_PORT:-8265}"                 # Ray dashboard / job-submit port (head)
NUM_ROLLOUT="${NUM_ROLLOUT:-50}"               # == number of train steps
SAVE_INTERVAL="${SAVE_INTERVAL:-10}"           # keep ~NUM_ROLLOUT/SAVE_INTERVAL adapters
RESP_LEN="${RESP_LEN:-7168}"                   # --rollout-max-response-len; response budget within SEQ
DAPO_DYNAMIC_SAMPLING="${DAPO_DYNAMIC_SAMPLING:-on}"  # on for a REAL model (toy resamples forever); off for gsm8k smoke
HF_CHECKPOINT="${HF_CHECKPOINT:-/cluster-storage/models/${MODEL}}"  # LOCAL dir, not a repo id
DATA_DIR="${DATA_DIR:-/personal/datasets}"     # persistent (NOT /root)
HF_HOME="${HF_HOME:-/cluster-storage/models}"
JOB_ID="${JOB_ID:-glm5_lora_full_mn_$(date +%y%m%d-%H%M%S)}"

# ----- recompute (effectively required for full model @ seq 8192) -----
#   These are NOT run_glm5_lora.py flags; they ride in --extra-args (miles passthrough).
#   Set RECOMPUTE=off to drop them.
RECOMPUTE="${RECOMPUTE:-on}"
RECOMPUTE_ARGS="--recompute-granularity full --recompute-method uniform --recompute-num-layers 1"

# ----- full-model parallelism override hook (see header risk #1) -----
#   _get_parallel_config emits a SINGLE-NODE layout TP=EP=GPUS_PER_NODE, PP=1, CP=1
#   regardless of node count (run_glm5_lora.py:192-215). For the FULL 744B model the
#   default PP=1 forces ALL 78 layers' frozen MoE experts onto every GPU (~161 GiB/GPU)
#   => guaranteed OOM on 141 GiB H200. PP>1 across nodes is MANDATORY. We OVERRIDE
#   PP/EP/CP via --extra-args (argparse last-wins). Layouts below are the Phase-2
#   memory-analysis recommendation (~24 GiB base/GPU for full 744B LoRA); ESTIMATE,
#   confirm against the first-launch allocator peak.  world = TP(=GPUS_PER_NODE)*PP*DP ;
#   EP must be <= DP*TP.  If PARALLEL_EXTRA is set in the env it is used verbatim.
PARALLEL_EXTRA="${PARALLEL_EXTRA:-}"
if [[ -z "$PARALLEL_EXTRA" ]]; then
  _world=$(( NODES * GPUS_PER_NODE ))
  # PP-FREE layout for GLM-5.2 cross-layer DSA. GLM-5.2 shares the DSA index across a
  # freq=4 group (1 compute layer + skip layers reuse its top-k). PP>1 can split such a
  # group across pipeline stages -> the skip layer loses its index source (assert / wrong
  # result), because Megatron PP only passes hidden states between stages, not the shared
  # index tensor. So we shard the frozen MoE experts via EP ONLY and keep PP=1, CP=1.
  #   world = TP(=GPUS_PER_NODE) x DP ; EP <= DP*TP ; experts/GPU ~= 1289GiB/EP.
  if [[ "$GPUS_PER_NODE" -eq 8 && "$NODES" -eq 8 ]]; then
    # 64 GPU: TP8 x DP8, PP1, CP1, EP32 -> experts ~40 + nonexpert ~12 = ~52 GiB/GPU.
    PARALLEL_EXTRA="--pipeline-model-parallel-size 1 --context-parallel-size 1 --expert-model-parallel-size 32"
  elif [[ "$GPUS_PER_NODE" -eq 8 && "$NODES" -eq 4 ]]; then
    # 32 GPU: TP8 x DP4, PP1, CP1, EP32 (=DP*TP, full expert shard) -> ~52 GiB/GPU base.
    # TIGHT under colocate sglang (mem-frac 0.5 ~70 GiB): if OOM, lower --sglang-mem-fraction-static.
    PARALLEL_EXTRA="--pipeline-model-parallel-size 1 --context-parallel-size 1 --expert-model-parallel-size 32"
  else
    echo "[warn] no auto PARALLEL_EXTRA for NODES=$NODES x GPUS_PER_NODE=$GPUS_PER_NODE (world=$_world):"
    echo "[warn]   single-node default EP=GPUS_PER_NODE OOMs the full model. Set PARALLEL_EXTRA: keep"
    echo "[warn]   PP=1 CP=1; set EP large (>=16, <=DP*TP) so the frozen MoE experts fit per GPU."
  fi
  [[ -n "$PARALLEL_EXTRA" ]] && echo "[parallel] auto PARALLEL_EXTRA (PP-free for cross-layer; validate at first launch): $PARALLEL_EXTRA"
fi
# If experts still OOM, raise EP (more expert sharding) FIRST. Only consider PP>1 if you have
# confirmed the cross-layer freq=4 group boundaries align to pipeline stages (else index sharing
# breaks). CP>1 is not needed at SEQ=8192 (activations are tiny under full recompute).

# ----- train-only fallback (see header risk #3: sglang may not serve GLM-5.2) -----
#   TRAIN_ONLY=on + DUMP_ROLLOUT=<path/to/rollout_data/0.pt> replays a dumped
#   rollout instead of doing live sglang generation.
TRAIN_ONLY="${TRAIN_ONLY:-off}"
DUMP_ROLLOUT="${DUMP_ROLLOUT:-}"

# ----- wandb (default ON; FAIL FAST if key missing) -----
#   WANDB=on      -> online logging. run_glm5_lora.py auto-adds
#                    --use-wandb --wandb-project miles-run_glm5_lora --wandb-group <run_id>
#                    --wandb-key <WANDB_API_KEY> --disable-wandb-random-suffix.
#                    REQUIRES WANDB_API_KEY in the env (NEVER hardcode it here).
#   WANDB=offline -> logs locally only (--wandb-mode offline).
#   WANDB=off     -> --no-enable-wandb (no wandb at all).
WANDB="${WANDB:-on}"
WANDB_API_KEY="${WANDB_API_KEY:-}"             # read from env / `wandb login`; do NOT commit it
WANDB_TEAM="${WANDB_TEAM:-}"                   # optional entity/team -> --wandb-team
WANDB_PROJECT="${WANDB_PROJECT:-}"             # optional override
WANDB_GROUP="${WANDB_GROUP:-}"                 # optional override

# Inter-node NIC for NCCL/GLOO (auto-detect iface in HEAD_IP's /24; override via NCCL_IFNAME).
HEAD_IP="${HEAD_IP:?set HEAD_IP to the head node IP on the inter-node NIC}"
if [[ -z "${NCCL_IFNAME:-}" ]]; then
  HEAD_PREFIX="${HEAD_IP%.*}."
  NCCL_IFNAME="$(ip -o -4 addr show 2>/dev/null | awk -v p="$HEAD_PREFIX" '$4 ~ ("^" p){print $2; exit}' || true)"
  if [[ -z "$NCCL_IFNAME" ]]; then
    NCCL_IFNAME="$(ifconfig 2>/dev/null | awk -v p="$HEAD_PREFIX" '/^[a-zA-Z]/{ifc=$1} $0 ~ ("inet "p){sub(/:$/,"",ifc); print ifc; exit}' || true)"
  fi
  NCCL_IFNAME="${NCCL_IFNAME:-eth0}"
fi

case "$ROLE" in
  # --------------------------------------------------------------------------
  head)
    echo "[head] (re)starting Ray head on ${HEAD_IP}:${RAY_PORT} with ${GPUS_PER_NODE} GPUs"
    echo "[head] config: MODEL=${MODEL} NODES=${NUM_NODES} GPUS_PER_NODE=${GPUS_PER_NODE} (world_size=$((GPUS_PER_NODE * NUM_NODES)))"
    ray stop --force >/dev/null 2>&1 || true
    sleep 3
    ray start --head \
      --node-ip-address "${HEAD_IP}" --port "${RAY_PORT}" \
      --dashboard-host 0.0.0.0 --dashboard-port "${DASH_PORT}" \
      --num-gpus "${GPUS_PER_NODE}" --disable-usage-stats
    echo
    echo "[head] now run THIS on each of the other $((NUM_NODES - 1)) node(s):"
    echo "         HEAD_IP=${HEAD_IP} GPUS_PER_NODE=${GPUS_PER_NODE} bash scripts/run_glm5_lora_multinode_full_model.sh worker"
    echo "[head] waiting for ${NUM_NODES} nodes to join (Ctrl-C to stop waiting)..."
    export RAY_ADDRESS="${HEAD_IP}:${RAY_PORT}"
    while :; do
      n="$(ray status 2>/dev/null | grep -cE '^ 1 node_' || true)"
      echo "[head]   nodes joined: ${n}/${NUM_NODES}"
      [[ "${n}" -ge "${NUM_NODES}" ]] && break
      sleep 5
    done
    echo "[head] cluster ready (${NUM_NODES} nodes). Next, on the head node run:"
    echo "         HEAD_IP=${HEAD_IP} GPUS_PER_NODE=${GPUS_PER_NODE} NODES=${NUM_NODES} MODEL=${MODEL} bash scripts/run_glm5_lora_multinode_full_model.sh launch"
    ;;

  # --------------------------------------------------------------------------
  worker)
    echo "[worker] (re)joining Ray head ${HEAD_IP}:${RAY_PORT} with ${GPUS_PER_NODE} GPUs"
    ray stop --force >/dev/null 2>&1 || true
    sleep 3
    ray start --address="${HEAD_IP}:${RAY_PORT}" --num-gpus "${GPUS_PER_NODE}"
    echo "[worker] joined. (Verify on head: RAY_ADDRESS=${HEAD_IP}:${RAY_PORT} ray status)"
    ;;

  # --------------------------------------------------------------------------
  launch)
    echo "============================================================"
    echo "[launch] GLM-5.2 FULL LoRA RL"
    echo "[launch]   MODEL=${MODEL}  BACKEND=${BACKEND}  LORA_RANK=${LORA_RANK}"
    echo "[launch]   NODES=${NUM_NODES} x ${GPUS_PER_NODE} GPU  (world_size=$((GPUS_PER_NODE * NUM_NODES)))"
    echo "[launch]   TASK=${TASK}  SEQ=${SEQ}  RESP_LEN=${RESP_LEN}  steps=${NUM_ROLLOUT}"
    echo "[launch]   HF_CHECKPOINT=${HF_CHECKPOINT}"
    echo "[launch]   JOB_ID=${JOB_ID}  NCCL/GLOO iface=${NCCL_IFNAME}"
    echo "============================================================"

    # ----- wandb: FAIL FAST when on/offline but key is missing (online) -----
    WANDB_SCRIPT_FLAG=""
    WANDB_EXTRA=""
    if [[ "$WANDB" == "off" ]]; then
      WANDB_SCRIPT_FLAG="--no-enable-wandb"
      echo "[launch] wandb: OFF"
    else
      if [[ "$WANDB" != "offline" && -z "$WANDB_API_KEY" ]]; then
        echo "[launch] FATAL: WANDB=${WANDB} but WANDB_API_KEY is unset." >&2
        echo "         miles would SILENTLY skip wandb. Export the key first (NEVER hardcode it):" >&2
        echo "           export WANDB_API_KEY=...        # or: wandb login && export WANDB_API_KEY=\$(...)" >&2
        echo "         Or set WANDB=offline (local-only, no key) / WANDB=off (disable)." >&2
        exit 3
      fi
      export WANDB_API_KEY                                  # get_default_wandb_args reads os.environ
      [[ "$WANDB" == "offline" ]] && WANDB_EXTRA+=" --wandb-mode offline"
      [[ -n "$WANDB_TEAM"    ]] && WANDB_EXTRA+=" --wandb-team ${WANDB_TEAM}"
      [[ -n "$WANDB_PROJECT" ]] && WANDB_EXTRA+=" --wandb-project ${WANDB_PROJECT}"
      [[ -n "$WANDB_GROUP"   ]] && WANDB_EXTRA+=" --wandb-group ${WANDB_GROUP}"
      echo "[launch] wandb: ${WANDB} (project auto=miles-run_glm5_lora unless overridden; key SET)"
    fi

    # ----- task / dataset flags -----
    #  --task wires --prompt-data/--input-key/--rm-type math automatically.
    #  --rollout-max-response-len is the ONLY length flag run_glm5_lora.py exposes.
    TASK_FLAGS="--task ${TASK} --rollout-max-response-len ${RESP_LEN}"
    if [[ "$TASK" == "dapo-math" && "$DAPO_DYNAMIC_SAMPLING" == "on" ]]; then
      TASK_FLAGS+=" --dapo-dynamic-sampling"
    fi
    echo "[launch] task flags: ${TASK_FLAGS}"

    # ----- sequence length (NO native --seq-length flag) -----
    #  run_glm5_lora.py has no seq-length / context-length flag. We thread the
    #  total window through miles passthrough args via --extra-args:
    #    --seq-length             -> megatron max_position_embeddings (defaults 4096 if unset)
    #    --rollout-max-context-len-> sglang inference context cap (auto prompt-len = SEQ-1)
    SEQ_EXTRA="--seq-length ${SEQ} --rollout-max-context-len ${SEQ}"

    # ----- recompute -----
    RC_EXTRA=""
    [[ "$RECOMPUTE" == "on" ]] && RC_EXTRA=" ${RECOMPUTE_ARGS}"

    # ----- train-only / dumped-rollout fallback (sglang GLM-5.2 gap) -----
    TRAINONLY_EXTRA=""
    if [[ "$TRAIN_ONLY" == "on" ]]; then
      [[ -z "$DUMP_ROLLOUT" ]] && { echo "[launch] FATAL: TRAIN_ONLY=on requires DUMP_ROLLOUT=<rollout_data/0.pt>." >&2; exit 4; }
      TRAINONLY_EXTRA=" --debug-train-only --load-debug-rollout-data ${DUMP_ROLLOUT}"
      echo "[launch] TRAIN-ONLY replay from ${DUMP_ROLLOUT} (no live sglang rollout)"
    fi

    # ----- assemble --extra-args (miles/train.py passthrough; argparse last-wins) -----
    #  --actor-num-nodes overrides the hardcoded 1; --num-gpus-per-node MUST be
    #  repeated here (rollout addr-allocator reads it) — see header CRITICAL note.
    EXTRA_ARGS="--actor-num-nodes ${NUM_NODES} --num-gpus-per-node ${GPUS_PER_NODE}"
    EXTRA_ARGS+=" --save-interval ${SAVE_INTERVAL}"
    EXTRA_ARGS+=" ${SEQ_EXTRA}"
    EXTRA_ARGS+="${RC_EXTRA}"
    [[ -n "$PARALLEL_EXTRA" ]] && EXTRA_ARGS+=" ${PARALLEL_EXTRA}"
    EXTRA_ARGS+="${TRAINONLY_EXTRA}"
    EXTRA_ARGS+="${WANDB_EXTRA}"
    echo "[launch] --extra-args: ${EXTRA_ARGS}"

    # ----- launch-time env -----
    export HF_HOME PYTHONUNBUFFERED=1
    export NCCL_SOCKET_IFNAME="${NCCL_IFNAME}" GLOO_SOCKET_IFNAME="${NCCL_IFNAME}"
    export MILES_SCRIPT_EXTERNAL_RAY=1                       # reuse the manually-formed cluster
    export RAY_ADDRESS="http://${HEAD_IP}:${DASH_PORT}"      # job-submit endpoint (dashboard)
    export MILES_RAY_SUBMIT_NO_WAIT=1                        # detached submit (survives WS drops)
    export MILES_RAY_SUBMISSION_ID="${JOB_ID}"

    # NOTE: --num-gpus-per-node appears BOTH as a script flag (drives TP=EP via
    # _get_parallel_config + --actor-num-gpus-per-node) AND inside --extra-args.
    python scripts/run_glm5_lora.py train \
      --model-name "${MODEL}" \
      --hf-checkpoint "${HF_CHECKPOINT}" \
      --dsa-attention-backend "${BACKEND}" \
      --lora-rank "${LORA_RANK}" \
      --num-gpus-per-node "${GPUS_PER_NODE}" \
      --num-rollout "${NUM_ROLLOUT}" \
      --data-dir "${DATA_DIR}" \
      ${TASK_FLAGS} \
      ${WANDB_SCRIPT_FLAG} \
      --extra-args "${EXTRA_ARGS}"

    echo
    echo "[launch] submitted job '${JOB_ID}'. Monitor with:"
    echo "    RAY_ADDRESS=http://${HEAD_IP}:${DASH_PORT} ray job status ${JOB_ID}"
    echo "    RAY_ADDRESS=http://${HEAD_IP}:${DASH_PORT} ray job logs   ${JOB_ID} --follow"
    echo "[launch] verify in logs: world_size=$((GPUS_PER_NODE * NUM_NODES)), actor-num-nodes=${NUM_NODES}, TP=${GPUS_PER_NODE}, qkv-format=bshd."
    ;;
esac
