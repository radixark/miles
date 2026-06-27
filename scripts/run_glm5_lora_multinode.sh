#!/usr/bin/env bash
# ============================================================================
#  run_glm5_lora_multinode.sh
#    UNIFIED multi-node GLM-5.x LoRA RL launcher — a thin wrapper around
#    scripts/run_glm5_lora.py that drives the *multi-node* flow for BOTH the
#    5-layer toy AND the FULL 78-layer GLM-5.2 (744B) model.
#
#    ⚠ DEFAULTS CHANGED (2026-06-24): this script now defaults to the FULL
#      GLM-5.2 / 8-node / 64-GPU production config (per util/plan_glm52_8node_full.md
#      condition C0 — 8-node training launches from THIS path). The validated
#      5-layer toy recipe is still reachable via env knobs (see TOY RECIPE below).
#
#    VALIDATED 2026-06-23 (toy): GLM-5.2_5layer, 2 nodes x 4xH200, unfused
#    (--dsa-attention-backend megatron-bridge), 50 steps -> Ray job SUCCEEDED.
#    Topology: world_size=8, TP4/EP4/PP1 + DP2, qkv-format bshd; 5 adapters saved.
#
#  ─────────────────────────────────────────────────────────────────────────
#  ⚠ READ BEFORE THE FIRST FULL-MODEL RUN — three things are UNVALIDATED at 744B:
#   1. PARALLELISM. run_glm5_lora.py:_get_parallel_config is HARD-CODED single-node
#      (TP=EP=GPUS_PER_NODE, PP=1, CP=1). That fits the toy, but the FULL 78-layer
#      model with PP=1 forces ALL frozen MoE experts onto every GPU -> OOM. We
#      OVERRIDE via PARALLEL_EXTRA (--extra-args, argparse last-wins). The auto
#      layout for 8 nodes is TP8 x DP8 / PP1 / CP1 / EP32 (~52 GiB/GPU base).
#      EP/CP choice criterion = "whatever launches clean, no OOM/no assert":
#      raise EP first; then CP>1 (+--allgather-cp); AVOID PP>1 (trips GLM-5.2
#      cross-layer-DSA build-time PP-split assert — freq=4 shared-index group).
#   2. SEQ=8192. No native --seq-length flag; wired via --extra-args
#      "--seq-length / --rollout-max-context-len" + --rollout-max-response-len.
#      Static micro-batch-size 1 (DSA bshd path) => recompute effectively required
#      (RECOMPUTE on by default).
#   3. ROLLOUT. sglang may not yet serve the GLM-5.2 cross-layer subset-indexer
#      checkpoint. If so, set TRAIN_ONLY=on DUMP_ROLLOUT=<rollout_data/0.pt> to
#      replay a dumped rollout instead of live colocate generation.
#  ─────────────────────────────────────────────────────────────────────────
#
#  WHY THIS WRAPPER EXISTS
#    run_glm5_lora.py is single-node by design: _train() hardcodes
#    `--actor-num-nodes 1`, and U.execute_train only does a local `ray start
#    --head`. This wrapper (a) forms a Ray cluster across nodes MANUALLY (head +
#    workers), and (b) submits with MILES_SCRIPT_EXTERNAL_RAY=1 so miles reuses
#    that cluster, overriding --actor-num-nodes via --extra-args (last-wins).
#
#  ⚠ CRITICAL — set GPUS_PER_NODE to the *REAL* per-node GPU count.
#    miles' rollout addr-allocator uses --num-gpus-per-node to place sglang
#    engines; if it disagrees with reality, worker-node engines block on a 600s
#    cross-node TCPStore timeout and die. We pass GPUS_PER_NODE in BOTH the
#    script flag AND --extra-args.
#
#  PARALLELISM RATIONALE — keep TP/EP intra-node (NVLink), cross nodes with DP.
#    Do NOT let TP span nodes. Extra nodes add pure DP (gradient all-reduce).
#
#  PREREQUISITES
#    * Shared FS for code + checkpoints (e.g. /personal) reachable by all nodes;
#      model present on every node (e.g. /cluster-storage/models/<MODEL>).
#    * Editable installs on every node (sglang, Megatron-Bridge `bridge` branch,
#      miles). Inter-node TCP; inter-node NIC (NCCL_IFNAME).
#    * For WANDB on/offline-online: WANDB_API_KEY exported (FAIL FAST if unset).
#
#  USAGE  (run from a miles checkout; this script cd's to the miles root)
#    # ===== FULL GLM-5.2, 8 nodes x 8 GPU (DEFAULT) =====
#    # 1) HEAD node — start Ray head, wait for the 7 workers:
#    HEAD_IP=<ip> GPUS_PER_NODE=8 NODES=8 \
#      bash scripts/run_glm5_lora_multinode.sh head
#    # 2) EACH of the other 7 nodes — join:
#    HEAD_IP=<ip> GPUS_PER_NODE=8 \
#      bash scripts/run_glm5_lora_multinode.sh worker
#    # 3) HEAD node, once `ray status` shows 8 nodes — launch:
#    export WANDB_API_KEY=...
#    HEAD_IP=<ip> GPUS_PER_NODE=8 NODES=8 \
#      bash scripts/run_glm5_lora_multinode.sh launch
#
#    # ===== TOY RECIPE (GLM-5.2_5layer, 2 nodes x 4 GPU) =====
#    HEAD_IP=<ip> GPUS_PER_NODE=4 NUM_NODES=2 MODEL_NAME=GLM-5.2_5layer \
#      TASK=gsm8k SEQ= RECOMPUTE=off WANDB=off \
#      bash scripts/run_glm5_lora_multinode.sh {head|worker|launch}
#    #   (SEQ= empty drops the seq-length override; the toy fits with PP1/CP1/EP=GPUS_PER_NODE.)
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

# ⚠ Raise the open-file limit BEFORE `ray start` / job submit. A 64-GPU multi-node
# colocate job spawns many workers (driver + GCS + per-GPU actors + sglang engines);
# the container's DEFAULT soft `ulimit -n` is 1024, which makes the HEAD raylet die
# fatally with "worker_pool.cc: Too many workers, failed to create a file. Try setting
# `ulimit -n` then restart Ray." -> node marked dead -> ActorDiedError -> job crash.
# Raise the soft limit to the hard cap (524288 on these pods). Override via ULIMIT_NOFILE.
ulimit -n "${ULIMIT_NOFILE:-$(ulimit -Hn)}" 2>/dev/null || true

# ============================================================================
#  TOP-OF-FILE KNOBS. Back-compat: this unified script accepts BOTH the old base
#  names (MODEL_NAME / NUM_NODES / DSA_BACKEND) and the full-model names
#  (MODEL / NODES / BACKEND). New names win if both are set.
# ============================================================================
MODEL="${MODEL:-${MODEL_NAME:-GLM-5.2}}"        # FULL GLM-5.2 (78L) default; toy = GLM-5.2_5layer
NODES="${NODES:-${NUM_NODES:-8}}"               # node count (==> DP=this, cross-node)
BACKEND="${BACKEND:-${DSA_BACKEND:-megatron-bridge}}"  # unfused DSA; "slime" = fused TileLang
NUM_NODES="$NODES"                              # internal canonical alias
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"             # REAL GPUs per node (==> TP=EP=this intra-node)
LORA_RANK="${LORA_RANK:-16}"
TASK="${TASK:-dapo-math}"                       # dataset: dapo-math | gsm8k
# Seq window (--seq-length + --rollout-max-context-len). The rollout caps generation at
# min(RESP_LEN, SEQ - prompt), so a window LARGER than RESP_LEN hard-bounds prompt+response (and the
# colocate memory window). dapo: 8192 (RESP 4096 + prompt headroom -> total can never exceed 8192).
# gsm8k: leave UNSET -- 512-tok responses sit well within megatron's window. NB: an unset SEQ is NOT
# the model's native max -- miles defaults --seq-length to a flat 4096 (megatron_utils/arguments.py).
if [[ "$TASK" == "dapo-math" ]]; then SEQ="${SEQ:-8192}"; else SEQ="${SEQ:-}"; fi
# ----------------------------------------------------------------------------

RAY_PORT="${RAY_PORT:-6379}"
DASH_PORT="${DASH_PORT:-8265}"
NUM_ROLLOUT="${NUM_ROLLOUT:-50}"               # == number of train steps
SAVE_INTERVAL="${SAVE_INTERVAL:-10}"           # keep ~NUM_ROLLOUT/SAVE_INTERVAL adapters
if [[ "$TASK" == "dapo-math" ]]; then RESP_LEN="${RESP_LEN:-4096}"; elif [[ "$TASK" == "gsm8k" ]]; then RESP_LEN="${RESP_LEN:-512}"; else RESP_LEN="${RESP_LEN:-7168}"; fi  # --rollout-max-response-len: dapo long-CoT 4096 (>2048 -> DSA indexer SPARSE), gsm8k short-answer 512
DAPO_DYNAMIC_SAMPLING="${DAPO_DYNAMIC_SAMPLING:-on}"  # on for a REAL model; off for gsm8k/toy smoke
EVAL="${EVAL:-on}"                             # periodic held-out eval (gsm8k->test split; dapo->AIME-2024 if present)
EVAL_INTERVAL="${EVAL_INTERVAL:-5}"            # eval every N rollout steps (only used when EVAL=on)

# ----- rollout (sglang) engine size — CRITICAL for the FULL 744B model -----
#   744B bf16 ~1488GB. An sglang rollout engine must span enough GPUs to HOLD the
#   whole model at --sglang-mem-fraction-static. On ONE node (8 GPU x 0.5 x 140 =
#   560GB, or even x0.9 = 1008GB) it OOMs ("CUDA out of memory" in SGLangEngine.init).
#   miles' addr_allocator DOES support multi-node engines (addr_allocator.py:41,96-97),
#   so the full model uses a MULTI-NODE engine. Auto: full -> 32 GPUs/engine (4 nodes;
#   1488/32 = ~46GB weights/GPU at mem-frac 0.5, 2 engines on 64 GPU); toy -> 2.
ROLLOUT_GPUS_PER_ENGINE="${ROLLOUT_GPUS_PER_ENGINE:-}"
SGLANG_MEM_FRACTION="${SGLANG_MEM_FRACTION:-}"   # empty => run_glm5_lora.py default (0.5)
# fp8 rollout: serve sglang from the pre-converted GLM-5.2_fp8 ckpt (engine=8, fp8 KV cache);
# megatron TRAIN stays bf16 (dequantizes the fp8 HF base via the bridge). Needs the _fp8 dir
# (tools/convert_hf_to_fp8.py). off => bf16 rollout (engine=32). NOTE: the gate_up LoRA crash is
# orthogonal to fp8 — the MLP-target drop in run_glm5_lora.py still applies.
FP8_ROLLOUT="${FP8_ROLLOUT:-off}"
# sglang LoRA kernel backend (triton|csgmv|ascend|torch_native). sglang default is csgmv,
# which has shown issues for the fused MLA (fused_qkv_a_proj_with_mqa) multi-slice path on
# GLM-5.2 DSA -> rollout gibberish. triton is the more robust kernel. Passed via --extra-args
# (run_glm5_lora.py does not expose it; train.py accepts --sglang-lora-backend). Empty => sglang default.
SGLANG_LORA_BACKEND="${SGLANG_LORA_BACKEND:-triton}"
# --lora-base-cpu-backup: keep a HOST-RAM mirror of the base weights on the sglang side
# (enable_weights_cpu_backup) so they survive torch_memory_saver.pause() across rollout<->train
# swaps without per-step re-ship. DEFAULT off: at full 744B on the slime backend the mirror
# (~372 GB/node) + megatron's slime init blew the colocate pod past its ~1.78 TB cgroup
# memory.max -> RolloutManager host-OOM (SIGTERM) -> sglang dp-attn all_gather "connection reset"
# -> cluster-wide scheduler crash. Turn on only when host RAM is known to fit. Trade-off when off:
# skip_base_sync=False so the trainer re-ships the base each swap (slower, but no host mirror).
LORA_BASE_CPU_BACKUP="${LORA_BASE_CPU_BACKUP:-off}"
# R3 = rollout ROUTING replay (MoE top-8) for rollout<->train on-policy parity. DEFAULT on.
# run_glm5_lora.py adds ONLY --use-rollout-routing-replay (cheap routed-experts capturer ~0.5GB/rank).
# It intentionally does NOT add --use-rollout-indexer-replay: the DSA indexer top-k replay is a
# DEBUG aid (the slime kernel recomputes the top-k, so training doesn't need it), and enabling it
# makes sglang allocate the IndexerTopkCapturer HOST pinned buffer (~78-128GB/rank x8) that
# host-OOM'd the colocate pod. So R3 on is safe (no indexer host buffer). R3=off (-> --no-use-r3)
# drops routing replay too -> fully off-policy; only for a quick mechanics bring-up.
R3="${R3:-on}"
# ----- MoE-expert LoRA (gate/up/down on the 256 experts) — DEFAULT ON -----
# KEEP_MOE_LORA=1 (default) keeps gate_proj/up_proj/down_proj in the LoRA target-modules, so the MoE
# expert layers (>= first_k_dense_replace, i.e. layer 3+) get LoRA — not just attention. It also
# turns on BOTH MoE-expert-LoRA flags in run_glm5_lora.py (two independent knobs, each must be on):
# --experts-shared-outer-loras (train-side layout) + --sglang-lora-use-virtual-experts (serve-side path).
# KEEP_MOE_LORA=0 -> attention-only LoRA (q/k/v/o + MLA q_a/kv_a/q_b/kv_b), the previous bring-up default.
KEEP_MOE_LORA="${KEEP_MOE_LORA:-1}"
# MOE_LORA_LAYERS: restrict the MoE-EXPERT LoRA to a subset of layers (attention LoRA stays on ALL
# layers) to cut actor_train backward-activation / optimizer memory of the many-layer expert
# grouped-GEMM. Empty = every MoE layer. Accepts ranges/commas, e.g. "58-77". Read by run_glm5_lora.py.
# Full GLM-5.2 (78 layers) defaults to the last 20 layers (58-77); toy / others default to all layers.
if [[ "$MODEL" == "GLM-5.2" ]]; then
  MOE_LORA_LAYERS="${MOE_LORA_LAYERS:-58-77}"
else
  MOE_LORA_LAYERS="${MOE_LORA_LAYERS:-}"
fi
if [[ -z "$ROLLOUT_GPUS_PER_ENGINE" ]]; then
  if [[ "$MODEL" == *5layer* ]]; then ROLLOUT_GPUS_PER_ENGINE=2
  elif [[ "$FP8_ROLLOUT" == "on" ]]; then ROLLOUT_GPUS_PER_ENGINE=8   # fp8: 744B fits 1 node/engine
  else ROLLOUT_GPUS_PER_ENGINE=32; fi
fi
# full bf16 GLM-5 rollout wants mem-fraction 0.70 (matches run_glm5_744b_a40b.py); toy stays at
# run_glm5_lora.py's 0.5 default. Only auto-set when the user left it empty.
[[ -z "$SGLANG_MEM_FRACTION" && "$MODEL" != *5layer* ]] && SGLANG_MEM_FRACTION=0.70
# NOTE: compute the _fp8 suffix on its own line. An inline $(...) that exits non-zero
# (the `[[ ]] && echo` when FP8 is off) makes the surrounding assignment fail, and under
# `set -e` that silently kills the launch before the banner (bf16 default + HF_CHECKPOINT unset).
_FP8_SUFFIX=""
[[ "$FP8_ROLLOUT" == "on" ]] && _FP8_SUFFIX="_fp8"
HF_CHECKPOINT="${HF_CHECKPOINT:-/cluster-storage/models/${MODEL}${_FP8_SUFFIX}}"  # LOCAL dir; _fp8 for fp8 rollout
DATA_DIR="${DATA_DIR:-/personal/datasets}"     # persistent (NOT /root)
HF_HOME="${HF_HOME:-/cluster-storage/models}"
JOB_ID="${JOB_ID:-glm5_lora_mn_$(date +%y%m%d-%H%M%S)}"

# ----- recompute (effectively required for the full model @ seq 8192) -----
RECOMPUTE="${RECOMPUTE:-on}"
RECOMPUTE_ARGS="--recompute-granularity full --recompute-method uniform --recompute-num-layers 1"

# ----- parallelism override hook (full model only; toy uses the single-node default) -----
#   _get_parallel_config emits a SINGLE-NODE layout (TP=EP=GPUS_PER_NODE, PP1, CP1).
#   For the FULL model PP1 forces all frozen MoE experts onto every GPU => OOM, so we
#   shard experts via EP (PP-free, to respect the cross-layer freq=4 shared-index group).
#   EP/CP choice = "whatever launches clean": raise EP first, then CP>1, never PP>1.
#   If PARALLEL_EXTRA is set in the env it is used verbatim. Toy (*5layer*) => empty (fits).
PARALLEL_EXTRA="${PARALLEL_EXTRA:-}"
if [[ -z "$PARALLEL_EXTRA" ]]; then
  if [[ "$MODEL" == *5layer* ]]; then
    : # toy: single-node TP=EP=GPUS_PER_NODE / PP1 / CP1 default fits — no override
  elif [[ "$GPUS_PER_NODE" -eq 8 && ( "$NODES" -eq 8 || "$NODES" -eq 4 ) ]]; then
    # full model, 64 or 32 GPU: TP8 x DP{8,4}, PP1, CP1, EP32 -> ~52 GiB/GPU base.
    PARALLEL_EXTRA="--pipeline-model-parallel-size 1 --context-parallel-size 1 --expert-model-parallel-size 32"
    echo "[parallel] auto PARALLEL_EXTRA (full-model PP-free, EP32; validate at first launch): $PARALLEL_EXTRA"
  else
    echo "[warn] no auto PARALLEL_EXTRA for full MODEL=$MODEL NODES=$NODES GPUS_PER_NODE=$GPUS_PER_NODE:"
    echo "[warn]   single-node default EP=GPUS_PER_NODE OOMs the full model. Set PARALLEL_EXTRA: keep"
    echo "[warn]   PP=1 CP=1; set EP large (>=16, <=DP*TP) so the frozen MoE experts fit per GPU."
  fi
fi

# ----- train-only / dumped-rollout fallback (sglang may not serve GLM-5.2) -----
TRAIN_ONLY="${TRAIN_ONLY:-off}"
DUMP_ROLLOUT="${DUMP_ROLLOUT:-}"

# ----- wandb (default ON; FAIL FAST if key missing when online) -----
WANDB="${WANDB:-on}"
WANDB_API_KEY="${WANDB_API_KEY:-}"             # read from env / `wandb login`; do NOT commit it
WANDB_TEAM="${WANDB_TEAM:-}"
WANDB_PROJECT="${WANDB_PROJECT:-}"
WANDB_GROUP="${WANDB_GROUP:-}"

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
    echo "         HEAD_IP=${HEAD_IP} GPUS_PER_NODE=${GPUS_PER_NODE} bash scripts/run_glm5_lora_multinode.sh worker"
    echo "[head] waiting for ${NUM_NODES} nodes to join (Ctrl-C to stop waiting)..."
    export RAY_ADDRESS="${HEAD_IP}:${RAY_PORT}"
    while :; do
      n="$(ray status 2>/dev/null | grep -cE '^ 1 node_' || true)"
      echo "[head]   nodes joined: ${n}/${NUM_NODES}"
      [[ "${n}" -ge "${NUM_NODES}" ]] && break
      sleep 5
    done
    echo "[head] cluster ready (${NUM_NODES} nodes). Next, on the head node run:"
    echo "         HEAD_IP=${HEAD_IP} GPUS_PER_NODE=${GPUS_PER_NODE} NODES=${NUM_NODES} MODEL=${MODEL} bash scripts/run_glm5_lora_multinode.sh launch"
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
    echo "[launch] GLM-5 multi-node LoRA RL"
    echo "[launch]   MODEL=${MODEL}  BACKEND=${BACKEND}  LORA_RANK=${LORA_RANK}"
    echo "[launch]   NODES=${NUM_NODES} x ${GPUS_PER_NODE} GPU  (world_size=$((GPUS_PER_NODE * NUM_NODES)))"
    echo "[launch]   TASK=${TASK}  SEQ=${SEQ:-<none>}  RESP_LEN=${RESP_LEN}  steps=${NUM_ROLLOUT}"
    echo "[launch]   HF_CHECKPOINT=${HF_CHECKPOINT}"
    echo "[launch]   JOB_ID=${JOB_ID}  NCCL/GLOO iface=${NCCL_IFNAME}"
    echo "============================================================"

    # ----- wandb: FAIL FAST when on (online) but key missing -----
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
    TASK_FLAGS="--task ${TASK} --rollout-max-response-len ${RESP_LEN}"
    if [[ "$TASK" == "dapo-math" && "$DAPO_DYNAMIC_SAMPLING" == "on" ]]; then
      TASK_FLAGS+=" --dapo-dynamic-sampling"
    fi
    echo "[launch] task flags: ${TASK_FLAGS}"

    # ----- rollout (sglang) engine sizing -----
    ROLLOUT_FLAGS="--rollout-num-gpus-per-engine ${ROLLOUT_GPUS_PER_ENGINE}"
    [[ -n "$SGLANG_MEM_FRACTION" ]] && ROLLOUT_FLAGS+=" --sglang-mem-fraction-static ${SGLANG_MEM_FRACTION}"
    echo "[launch] rollout flags: ${ROLLOUT_FLAGS}  (engine spans $(( (ROLLOUT_GPUS_PER_ENGINE + GPUS_PER_NODE - 1) / GPUS_PER_NODE )) node(s); $(( GPUS_PER_NODE * NUM_NODES / ROLLOUT_GPUS_PER_ENGINE )) engine(s))"

    # ----- sequence length (NO native --seq-length flag; SEQ empty => omit) -----
    SEQ_EXTRA=""
    [[ -n "$SEQ" ]] && SEQ_EXTRA="--seq-length ${SEQ} --rollout-max-context-len ${SEQ}"

    # ----- recompute -----
    RC_EXTRA=""
    [[ "$RECOMPUTE" == "on" ]] && RC_EXTRA=" ${RECOMPUTE_ARGS}"

    # ----- train-only / dumped-rollout fallback -----
    TRAINONLY_EXTRA=""
    if [[ "$TRAIN_ONLY" == "on" ]]; then
      [[ -z "$DUMP_ROLLOUT" ]] && { echo "[launch] FATAL: TRAIN_ONLY=on requires DUMP_ROLLOUT=<rollout_data/0.pt>." >&2; exit 4; }
      TRAINONLY_EXTRA=" --debug-train-only --load-debug-rollout-data ${DUMP_ROLLOUT}"
      echo "[launch] TRAIN-ONLY replay from ${DUMP_ROLLOUT} (no live sglang rollout)"
    fi

    # ----- assemble --extra-args (miles/train.py passthrough; argparse last-wins) -----
    EXTRA_ARGS="--actor-num-nodes ${NUM_NODES} --num-gpus-per-node ${GPUS_PER_NODE}"
    EXTRA_ARGS+=" --save-interval ${SAVE_INTERVAL}"
    [[ -n "$SEQ_EXTRA" ]] && EXTRA_ARGS+=" ${SEQ_EXTRA}"
    EXTRA_ARGS+="${RC_EXTRA}"
    [[ -n "$PARALLEL_EXTRA" ]] && EXTRA_ARGS+=" ${PARALLEL_EXTRA}"
    # alltoall MoE dispatcher for ALL GLM-5 LoRA models incl. the 5-layer toy (NOT deepep/flex —
    # deepep is fp8-only and conflicts with the triton LoRA MoE runner). Redundant for the actor
    # (bridge_lora_helpers.py hardcodes provider.moe_token_dispatcher_type="alltoall") but kept
    # explicit so the 5-layer toy's train.py command matches the full recipe exactly.
    EXTRA_ARGS+=" --moe-token-dispatcher-type alltoall"
    [[ -n "$SGLANG_LORA_BACKEND" ]] && EXTRA_ARGS+=" --sglang-lora-backend ${SGLANG_LORA_BACKEND}"
    # opt-in host-RAM base mirror (default off; see knob note above). argparse last-wins.
    [[ "$LORA_BASE_CPU_BACKUP" == "on" ]] && EXTRA_ARGS+=" --lora-base-cpu-backup"
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
    # MoE-expert LoRA master switch, read by run_glm5_lora.py via os.environ (driver-side, at command
    # build); export so the `python scripts/run_glm5_lora.py train` child inherits the default (ON).
    export KEEP_MOE_LORA MOE_LORA_LAYERS

    # NOTE: --num-gpus-per-node appears BOTH as a script flag (drives TP=EP via
    # _get_parallel_config + --actor-num-gpus-per-node) AND inside --extra-args.
    python scripts/run_glm5_lora.py train \
      --model-name "${MODEL}" \
      --hf-checkpoint "${HF_CHECKPOINT}" \
      --dsa-attention-backend "${BACKEND}" \
      --lora-rank "${LORA_RANK}" \
      $( [[ "$FP8_ROLLOUT" == "on" ]] && echo --fp8-rollout ) \
      $( [[ "$R3" == "off" ]] && echo --no-use-r3 ) \
      $( [[ "$EVAL" == "off" ]] && echo --no-enable-eval ) \
      --eval-interval "${EVAL_INTERVAL}" \
      --num-gpus-per-node "${GPUS_PER_NODE}" \
      --num-rollout "${NUM_ROLLOUT}" \
      --data-dir "${DATA_DIR}" \
      ${TASK_FLAGS} \
      ${ROLLOUT_FLAGS} \
      ${WANDB_SCRIPT_FLAG} \
      --extra-args "${EXTRA_ARGS}"

    echo
    echo "[launch] submitted job '${JOB_ID}'. Monitor with:"
    echo "    RAY_ADDRESS=http://${HEAD_IP}:${DASH_PORT} ray job status ${JOB_ID}"
    echo "    RAY_ADDRESS=http://${HEAD_IP}:${DASH_PORT} ray job logs   ${JOB_ID} --follow"
    echo "[launch] verify in logs: world_size=$((GPUS_PER_NODE * NUM_NODES)), actor-num-nodes=${NUM_NODES}, TP=${GPUS_PER_NODE}, qkv-format=bshd."
    ;;
esac
