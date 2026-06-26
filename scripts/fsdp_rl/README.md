# FSDP-backend RL reference scripts

One-command GRPO RL runs on the experimental **FSDP backend**, one per representative model
(one dense + one MoE per family). All share the same policy via [`common.sh`](common.sh):

- **train data:** DAPO-math-17k &nbsp;•&nbsp; **eval:** gsm8k &nbsp;•&nbsp; **seq:** 8k response len
- **GRPO**, AdamW, colocate rollout (sglang), `--use-dynamic-batch-size`
- **no checkpointing** (no `--save`/`--load`)

## Run

```bash
bash scripts/fsdp_rl/qwen3-30b-a3b.sh      # any model script
```

Each script just sets a few env vars and sources `common.sh`. Override paths with
`MODELS_DIR`, `DATA_DIR`, `MILES_DIR`; set `MODEL` to an HF hub id to pull from the Hub.
`wandb` turns on automatically when `WANDB_API_KEY` is set. Multi-node scripts print the
`ray start --address=...` line the worker nodes need.

**CPU offload is on by default** (`CPU_OFFLOAD=1`) — optimizer/params to CPU so the colocated training
step fits alongside sglang (validated: a 4B trains end-to-end this way with the 8k seq). If a model has
GPU room, `CPU_OFFLOAD=0 bash scripts/fsdp_rl/<m>.sh` runs faster. `ROLLOUT_GPUS_PER_ENGINE` sets
sglang's TP per engine — `1` for ≤24B, `2` for 26B+ (they don't fit one GPU at the default sglang
fraction). On OOM: lower `SGLANG_MEM` or `MAX_TOKENS_PER_GPU` (defaults 0.4 / 10240).

## Models & GPU sizing

GPU counts are sized for FSDP RL (bf16 weights+grads on GPU, AdamW state on CPU when
offloaded, plus the colocated sglang weights + KV cache) on H200 (140 GB).

| script | model | type | GPUs | offload |
|---|---|---|---|---|
| `qwen3-4b` | Qwen3-4B | dense | 1×4 | – |
| `qwen3.5-4b` | Qwen3.5-4B | dense | 1×4 | – |
| `nemotron3-nano-4b` | Nemotron-3-Nano-4B | dense (Mamba2 hybrid) | 1×4 | – |
| `gemma-4-31b` | Gemma-4-31B | dense | 1×8 | ✓ |
| `gpt-oss-20b` | gpt-oss-20B | MoE | 1×8 | – |
| `qwen3-30b-a3b` | Qwen3-30B-A3B | MoE (`qwen3_moe`) | 1×8 | ✓ |
| `qwen3.5-35b-a3b` | Qwen3.5-35B-A3B | MoE (GatedDeltaNet) | 1×8 | ✓ |
| `gemma-4-26b-a4b` | Gemma-4-26B-A4B | MoE | 1×8 | ✓ |
| `nemotron3-nano-30b-a3b` | Nemotron-3-Nano-30B-A3B | MoE (`nemotron_h` hybrid) | 1×8 | ✓ |
| `glm4.7-flash` | GLM-4.7-Flash | MoE (`glm4_moe_lite`, fp32-master) | 1×8 | ✓ |
| `qwen3-next-80b-a3b` | Qwen3-Next-80B-A3B | MoE (GatedDeltaNet) | 1×8 | ✓ |
| `deepseek-v3` | DeepSeek-V3 (671B) | MoE | 8×8 (64) | ✓ |
| `kimi-k2.5` | Kimi-K2.5 (~1T) | MoE | 16×8 (128) | ✓ |

**Note on the very large ones:** `deepseek-v3` (671B) and `kimi-k2.5` (~1T) are sized for *pure*
FSDP data-parallel sharding — this backend has no expert/pipeline parallelism, so at this scale it's
aggressive (sized for weights+grads+sglang with the optimizer on CPU; in practice they'd want EP/PP).
Adjust `NNODES`/`GPUS_PER_NODE`/`MAX_TOKENS_PER_GPU` to your cluster.
