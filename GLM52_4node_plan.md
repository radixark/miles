# GLM-5.2 Full LoRA + DAPO — 4-Node Plan (32× H200)

Pivot from 8→4 nodes: easier to land on CSI-healthy nodes, lets the experiment start sooner.
Script unchanged — `NODES=4` is a launch-time knob.

## Fixed conditions
| Item | Value |
|---|---|
| Compute | 4 nodes × 8 H200 = **32 GPU** (world_size=32) |
| Model | **GLM-5.2 full** (`glm5.2-744B-A40B`, 78 layers, 256 experts top-8, cross-layer DSA) |
| Train | **LoRA rank=16**, unfused (`--dsa-attention-backend megatron-bridge`) |
| Data | **dapo-math**, SEQ=8192 (RESP_LEN=7168), DAPO dynamic sampling on |
| Logging | wandb on (key from `$WANDB_API_KEY`, never hardcoded) |
| Script | `scripts/run_glm5_lora_multinode_full_model.sh`, `NODES=4` |

## ⚠️ Real root cause of the acquire failures (NOT image pull)
`rx devbox why` revealed: pods get **Scheduled** then stick on
`FailedMount … csi: driver name s3.radixark.io not found in the list of registered CSI drivers`
— the **`/global-s3` mount's CSI node-plugin is missing/unhealthy on some H200 nodes**. The pod
never initializes → 17-min provisioning timeout. prep-cpu ran fine on `gpu1-10-220-51-39`, so
**some nodes are healthy, some are broken** — it's node-level. The earlier "image pull" /
"RDMA" / "warm-cache via re-acquire" theories were wrong.

## Phase 0 — devbox (4 nodes, dodge CSI-broken nodes)
- [ ] **0.1** Auto-skip-bad-nodes acquire loop (`scratchpad/acquire_4node_skipbad.sh`): acquire
      32 GPU, watch `rx devbox why`; when a node reports the `s3.radixark.io` FailedMount, add it
      to `--skip-nodes` and re-acquire — converges to 4 CSI-healthy nodes. Seeded skip: `…51-51`.
- [ ] **0.2** (If the loop can't find 4 healthy nodes) escalate: ops/KZ restore the s3 CSI
      DaemonSet, OR run read-only kubectl to map healthy nodes (commands below) → feed to
      `--prefer-nodes`. *I have no operator/kubectl; `rx cluster kubectl` returns 403 for me.*

```bash
# kubectl (run with your own kubeconfig, or hand to KZ) — find CSI-healthy nodes:
NS=devbox-glm5-2-lora-<id>
kubectl -n $NS describe pod devbox-glm5-2-lora-0 | grep -A4 -iE "Events|FailedMount|csi|s3"
kubectl get csidrivers | grep -i s3                      # is s3.radixark.io registered at all?
kubectl get csinodes -o custom-columns='NODE:.metadata.name,DRIVERS:.spec.drivers[*].name' | grep -i s3
kubectl get pods -A -o wide | grep -i csi | grep -i radixark   # s3 CSI node-plugin DaemonSet health
```

## Phase 1 — env prep (mostly already done on shared storage)
- [x] **1.1 Model** — GLM-5.2 full **already downloaded** (1.4T, 282/282 shards, complete) and
      symlinked `/cluster-storage/models/GLM-5.2 → models--zai-org--GLM-5.2/snapshots/e32aaf03`.
      config.json = `glm_moe_dsa`, 78 layers. **No `hf download` needed.**
- [x] **1.2 Dataset** — `/personal/datasets/dapo-math-17k` present. **No download needed.**
- [ ] **1.3 Wrapper re-sync** — the updated `run_glm5_lora_multinode_full_model.sh` (with the
      PP-free PARALLEL_EXTRA fix) must be re-synced to `/personal/miles/scripts/` once a devbox
      is up (prev sync was the pre-fix version; prep-cpu that did it has TTL-expired).
- [ ] **1.4 Pre-install** — `bash /personal/miles/2_pre-install.sh` on the nodes (Megatron-Bridge
      **bridge** branch / editable install) if not already effective on the env.
- [ ] **1.5 wandb** — `export WANDB_API_KEY=<key>` in the head shell (fail-fast if unset).

## Phase 2 — launch risks (resolved by Phase-2 workflow, refined for 4-node + cross-layer)
- [ ] **R1 — parallel/memory (the key risk).** Default `_get_parallel_config` emits PP=1/EP=8 →
      ~161 GiB/GPU → **OOM**. Fix (auto-set by wrapper for NODES=4): **PP-free**
      `--pipeline-model-parallel-size 1 --context-parallel-size 1 --expert-model-parallel-size 32`
      (TP=8, DP=4, EP=32=DP×TP full expert shard) → **~52 GiB/GPU base**. PP avoided on purpose:
      PP>1 risks splitting a cross-layer freq=4 group across stages (skip layer loses its shared
      index). **Tight under colocate sglang** (mem-frac 0.5 ≈ 70 GiB; 52+70 with time-share ≈ peak
      ~70–122 GiB < 141): if it OOMs, lower `--sglang-mem-fraction-static`, then raise EP.
- [x] **R2 — sglang rollout.** Full 78-layer GLM-5.2 cross-layer DSA serving **IS supported**
      (config-driven, no layer-count gating; the `run_glm5_lora.py:60` "can't serve" comment is
      stale). → **TRAIN_ONLY=off, run live RL rollout.** Verify with one short sglang smoke-serve
      that *this cluster's* sglang binary picks `attention_backend=dsa` (use `--disable-cuda-graph`;
      do NOT enable `--enable-two-batch-overlap` — incompatible with cross-layer). Fallback only if
      the deployed binary is old: `TRAIN_ONLY=on DUMP_ROLLOUT=<dump.pt>`.
- [x] **R3 — SEQ.** 8192 ≪ max_position_embeddings (1048576). Fine.

## Phase 3 — launch (head → 3 workers → launch)
```bash
HEAD_IP=<head-node-ip-on-inter-node-NIC>; export HEAD_IP
# 1) head node
export WANDB_API_KEY=<key>
HEAD_IP=$HEAD_IP GPUS_PER_NODE=8 NODES=4 bash /personal/miles/scripts/run_glm5_lora_multinode_full_model.sh head
# 2) each of the other 3 nodes
HEAD_IP=$HEAD_IP GPUS_PER_NODE=8 bash /personal/miles/scripts/run_glm5_lora_multinode_full_model.sh worker
# 3) on head, once `ray status` shows 4 nodes / 32 GPU
export WANDB_API_KEY=<key>
HEAD_IP=$HEAD_IP GPUS_PER_NODE=8 NODES=4 bash /personal/miles/scripts/run_glm5_lora_multinode_full_model.sh launch
#   (wrapper auto-sets PP-free PARALLEL_EXTRA for NODES=4; override via env if needed)
```

## Phase 4 — monitor + outputs
- [ ] `ray job logs <JOB_ID> --follow`; verify in logs: **world_size=32, actor-num-nodes=4, TP=8,
      PP=1, EP=32, qkv-format=bshd**, and that PARALLEL_EXTRA won over the hardcoded PP=1/EP=8.
- [ ] No RDMA fail / no 600s TCPStore timeout (GPUS_PER_NODE must = real per-node count) / no OOM /
      no s3-CSI FailedMount; wandb run appears; first train step completes.
- [ ] Watch the first allocator peak after weights load + step 1 (the ~52 GiB estimate is unverified;
      this is the first real-scale full-model run — only the 5-layer toy + 50-step toy ran before).
- [ ] **Output:** LoRA adapters under `/personal/checkpoints/<run_id>` + wandb curves.

## Changes vs 8-node
world_size 32 (was 64); 3 workers (was 7); **PP-free EP=32** layout (was the abandoned PP=4/EP=16);
need only 4 CSI-healthy nodes; memory **tighter** (~52 GiB/GPU) → first watch point is OOM at the
colocate handoff.
