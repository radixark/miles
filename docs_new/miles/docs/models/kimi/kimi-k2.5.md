---
title: Kimi K2.5 / K2.6
description: Launch recipe for Kimi-K2.5 — full-parameter GRPO on 32 × 8 H200, INT4 actor with a BF16 reference.
---
The reference launcher is [`scripts/run-kimi-k25.sh`](https://github.com/radixark/miles/blob/main/scripts/run-kimi-k25.sh), which sources the shared model definition in `scripts/models/kimi-k2-thinking.sh`.

<Note title="K2.6 status">
At the time of writing, `radixark/miles@main` ships a K2.5 launcher only — there is no separate K2.6 script or model config yet. K2.6 reuses the K2.5 recipe, so the flags on this page are validated against `scripts/run-kimi-k25.sh` and serve as the K2.6 starting point until a dedicated launcher lands.
</Note>

## 1. Model Introduction

[Kimi-K2](https://moonshotai.github.io/Kimi-K2/) is a Mixture-of-Experts (MoE) language model from Moonshot AI. Kimi-K2.5 reuses the same MoE + Multi-head Latent Attention (MLA) architecture as Kimi-K2-Thinking — the K2.5 launcher sources the Thinking model definition directly — so it carries the 1 T-total / 32 B-active shape of the K2 family.

**Key highlights:**

- **K2-Thinking architecture, reused as-is**: 61 layers (1 dense + 60 MoE, set by `FIRST_K_DENSE_REPLACE=1`), hidden-size 7168, `ffn_hidden_size=18432`, 64 attention heads, and a 163840-token vocabulary with untied input/output embeddings.
- **Multi-head Latent Attention**: `q_lora_rank=1536`, `kv_lora_rank=512`, `qk_head_dim=128`, `qk_pos_emb_head_dim=64`, `v_head_dim=128`, with QK layernorm. RoPE uses `--rotary-base 50000` and `--rotary-scaling-factor 64.0` (`mscale=1.0`), the same setup as K2-Thinking.
- **MoE topology**: 384 routed experts + 1 shared expert (`moe_ffn_hidden_size=2048`, `moe_shared_expert_intermediate_size=2048`), top-8 routing with a sigmoid score function, pre-softmax routing, expert bias, and `seq_aux_loss` load balancing at `moe_aux_loss_coeff=0`. The router runs in FP32 with `--moe-router-topk-scaling-factor 2.827`.
- **INT4 actor with a BF16 reference**: the actor loads an INT4 checkpoint and trains under simulated INT4 QAT (`OPEN_TRAINING_INT4_FAKE_QAT_FLAG=1`, group size 32), while the reference model loads the BF16 checkpoint. Kimi-K2-Thinking is the canonical INT4 QAT target in miles, and K2.5 follows the same path — see [INT4 QAT](/miles/docs/advanced/int4-qat).
- **Megatron ↔ HF bridge**: weights move through `--megatron-to-hf-mode bridge`, and the recipe selects the K2.5 wiring with `--model-name kimi_k25`.

## 2. Supported Variants

The K2.5 launcher expects two checkpoints under `$BASE_DIR`: an INT4 actor checkpoint and a BF16 reference checkpoint.

| Role | Checkpoint | Loaded with |
|---|---|---|
| Actor (trained) | `$BASE_DIR/Kimi-K2.5-int4` | `--hf-checkpoint` |
| Reference | `$BASE_DIR/Kimi-K2.5-bf16` | `--ref-load` |

Both share the K2 family's 1 T-total / 32 B-active MoE + MLA shape inherited from Kimi-K2-Thinking.

## 3. Quick start

### 3.1 Prerequisites

The launcher references two environment variables but never sets them, so you should export them yourself before launch:

```bash
export BASE_DIR=<shared FS path, reachable from every node>
export MASTER_ADDR=<head node IP>
```

The `$BASE_DIR` directory must already hold the two K2.5 checkpoints from §2 alongside the DAPO-Math-17k training set (`dapo-math-17k/dapo-math-17k.jsonl`) and the AIME-2024 eval set (`aime-2024.jsonl`).

### 3.2 One-line launch

The script submits to an **already-running Ray cluster** (`ray job submit --address http://127.0.0.1:8265`); it does not run `ray start --head` itself. It also runs a `pkill` / `ray stop` cleanup pass at the top so a failed run can be re-launched cleanly.

```bash
cd /root/miles
export BASE_DIR=...; export MASTER_ADDR=...
bash scripts/run-kimi-k25.sh
```

### 3.3 Multi-node fan-out

Bring up Ray on every node before launching, the same way as the other Kimi recipes:

```bash
# head
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats
# each worker
ray start --address=${MASTER_ADDR}:6379 --num-gpus 8 --node-ip-address ${WORKER_IP}
```

## 4. Script breakdown

The launcher groups its flags into the arrays that are passed to `train.py`. The model shape comes from `MODEL_ARGS`, which is sourced from `scripts/models/kimi-k2-thinking.sh`; the K2.5 recipe then layers the following on top:

- **`CKPT_ARGS`** wires up the dual checkpoint — INT4 actor via `--hf-checkpoint`, BF16 reference via `--ref-load` — with `--megatron-to-hf-mode bridge` and `--model-name kimi_k25`.
- **`ROLLOUT_ARGS`** and **`EVAL_ARGS`** configure GRPO sampling and periodic AIME evaluation (covered in §5.2).
- **`PERF_ARGS`** sets the parallelism layout and recomputation (§5.1).
- **`GRPO_ARGS`** and **`OPTIMIZER_ARGS`** set the algorithm and CPU-offloaded Adam (§5.2, §5.4).
- **`SGLANG_ARGS`** configures the colocated rollout engine (§5.3).

The job runs colocated (`--colocate`) across 32 nodes (`--actor-num-nodes 32 --actor-num-gpus-per-node 8`) and uses the miles router (`--use-miles-router`) with `--update-weight-buffer-size $((4*512*1024*1024))`.

## 5. Example Recipe Configuration

### 5.1 Megatron Parallelism

This is the validated layout shipped with the launcher. All parallelisms are supported, so you can supply any other TP / EP / PP / CP combination that fits your compute.

| Hardware | Nodes × GPUs | TP | PP | CP | EP | expert-TP | `--decoder-last-pipeline-num-layers` | `--max-tokens-per-gpu` |
|---|---|---|---|---|---|---|---|---|
| H200 | 32 × 8 = 256 | 8 | 8 | 4 | 32 | 1 | 5 | 4096 |

Sequence parallelism (`--sequence-parallel`) is on, and the trainer uses dynamic batching (`--use-dynamic-batch-size`) capped at `--max-tokens-per-gpu 4096`. Recomputation is full and uniform over a single layer:

```bash
--recompute-granularity full
--recompute-method uniform
--recompute-num-layers 1
```

### 5.2 Algorithm

The recipe uses GRPO with KL and entropy losses disabled:

```bash
--advantage-estimator grpo
--eps-clip 0.2 --eps-clip-high 0.28
--kl-loss-coef 0.00 --kl-loss-type low_var_kl
--entropy-coef 0.00
```

Rollouts draw from DAPO-Math-17k and score with the `deepscaler` reward:

```bash
--prompt-data $BASE_DIR/dapo-math-17k/dapo-math-17k.jsonl
--input-key prompt --label-key label
--apply-chat-template
--rollout-shuffle --balance-data
--rm-type deepscaler

--num-rollout 20
--rollout-batch-size 32
--n-samples-per-prompt 8
--rollout-max-response-len 16384
--rollout-temperature 1

--global-batch-size 256
--filter-zero-reward-samples
--use-dynamic-global-batch-size
```

Evaluation runs every 20 steps against AIME-2024, sampling 16 responses per prompt:

```bash
--eval-interval 20
--eval-prompt-data aime $BASE_DIR/aime-2024.jsonl
--n-samples-per-eval-prompt 16
--eval-max-response-len 16384
--eval-top-p 1
```

### 5.3 Rollout & SGLang

The rollout engine is colocated with training, spanning 8 GPUs per engine with 8-way expert parallelism:

```bash
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 8
   --sglang-mem-fraction-static 0.7
   --sglang-ep-size 8
   --sglang-server-concurrency 1024
   --sglang-cuda-graph-bs 1 2 4 8 16 24 32 40 48 56 64 72 80 88 96 104 112 120 128
   --use-rollout-routing-replay
)
```

The `--use-rollout-routing-replay` flag replays the rollout-time MoE routing decisions during training so the two stages stay consistent. On the Megatron side, attention uses the Flash backend (`--attention-backend flash`).

The launcher sets the required env vars for you, including the INT4 QAT pair (`OPEN_TRAINING_INT4_FAKE_QAT_FLAG=1`, `OPEN_TRAINING_INT4_GROUP_SIZE=32`), a long NCCL timeout (`NCCL_TIMEOUT=3600`), `CUDA_DEVICE_MAX_CONNECTIONS=1`, and NVLink-gated NVLS (`NCCL_NVLS_ENABLE` follows the script's NVLink autodetection).

### 5.4 Optimizer

CPU-offloaded Adam is combined with the distributed optimizer:

```bash
--optimizer adam
--lr 1e-6 --lr-decay-style constant
--weight-decay 0.1
--adam-beta1 0.9 --adam-beta2 0.98

--optimizer-cpu-offload
--overlap-cpu-optimizer-d2h-h2d
--use-precision-aware-optimizer
--use-distributed-optimizer
```

Adam states live on host RAM and are D2H/H2D-overlapped with the backward pass, freeing GPU memory for the 1 T-parameter weight footprint. Gradients accumulate and all-reduce in FP32 (`--accumulate-allreduce-grads-in-fp32`), and the attention softmax also runs in FP32 (`--attention-softmax-in-fp32`).

## 6. Pairs Well With

- [INT4 QAT](/miles/docs/advanced/int4-qat)
- [PD Disaggregation](/miles/docs/advanced/pd-disaggregation)
- [P2P Weight Transfer](/miles/docs/advanced/p2p-weight-transfer)
- [Fault Tolerance](/miles/docs/advanced/fault-tolerance)
- [Kimi K2](/miles/docs/models/kimi/kimi-k2) — sibling recipe; K2.5 reuses the K2-Thinking architecture.
