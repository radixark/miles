---
title: Kimi K2.5 / K2.6
description: Launch recipe for Kimi-K2.5, running full-parameter GRPO on 32 × 8 H200 with an INT4 actor and a BF16 reference.
---
The reference launcher is [`scripts/run_kimi_k25.py`](https://github.com/radixark/miles/blob/main/scripts/run_kimi_k25.py), which loads the shared model definition from `scripts/models/kimi-k2-thinking.py`.

## 1. Model Introduction

[Kimi-K2.5](https://huggingface.co/moonshotai/Kimi-K2.5) is an open-source, natively multimodal agentic model from Moonshot AI. It is built by continual pretraining on roughly 15 T mixed vision and text tokens on top of Kimi-K2-Base, and it pairs a Mixture-of-Experts (MoE) language backbone with a MoonViT vision encoder so a single model handles both image and text inputs. K2.5 keeps the 1 T-total / 32 B-active shape of the K2 family and extends the context window to 256K tokens.

**Architecture at a glance** (from the model card):

| Specification | Value |
|---|---|
| Architecture | Mixture-of-Experts (MoE) |
| Total / activated parameters | 1 T / 32 B |
| Layers (1 dense + 60 MoE) | 61 |
| Attention hidden dimension | 7168 |
| MoE hidden dimension (per expert) | 2048 |
| Attention heads | 64 |
| Routed experts (selected per token) | 384 (top-8) |
| Shared experts | 1 |
| Attention mechanism | Multi-head Latent Attention (MLA) |
| Activation | SwiGLU |
| Vocabulary size | 160K |
| Context length | 256K |
| Vision encoder | MoonViT (400 M) |

**Key features:**

- **Native multimodality.** K2.5 is pretrained on both vision and language tokens, so it covers visual knowledge, cross-modal reasoning, and tool use grounded in images alongside text.
- **Coding with vision.** It generates code from visual specifications such as UI designs and video workflows, and drives tools for visual data processing.
- **Agent swarm.** It decomposes a complex task into parallel sub-tasks run by dynamically instantiated, domain-specific agents, rather than scaling a single agent.

## 2. Supported Variants

The K2.5 launcher expects two checkpoints under `--model-dir` (default `/root/models`): the published INT4 actor checkpoint and a BF16 reference dequantized from it.

| Role | Checkpoint | Loaded with |
|---|---|---|
| Actor (trained) | `<model-dir>/Kimi-K2.5` | `--hf-checkpoint` |
| Reference | `<model-dir>/Kimi-K2.5-bf16` | `--ref-load` |

The `prepare` subcommand downloads the first and produces the second with `tools/convert_kimi_int4_to_bf16.py`.

Both share the K2 family's 1 T-total / 32 B-active MoE + MLA shape inherited from Kimi-K2-Thinking.

## 3. Quick start

### 3.1 Prerequisites

Export the ray head address before launch:

```bash
export MASTER_ADDR=<head node IP>
```

Everything else is a Typer flag: `--model-dir` (default `/root/models`) for the two K2.5 checkpoints from §2, `--data-dir` (default `/root/datasets`) for the DAPO-Math-17k training set (`dapo-math-17k/dapo-math-17k.jsonl`) and the AIME-2024 eval set (`aime-2024/aime-2024.jsonl`), and `--output-dir` (default `/root/shared_data`) for what the run writes. On a multi-node run all three must be on a shared FS. `prepare` populates the first two for you.

### 3.2 One-line launch

The launcher runs a `pkill` / `ray stop` cleanup pass first so a failed run can be re-launched cleanly, then starts the ray head and submits with `ray job submit --address http://127.0.0.1:8265`. Export `MILES_SCRIPT_EXTERNAL_RAY=1` to skip the `ray start` and submit to an **already-running Ray cluster** instead.

```bash
cd /root/miles
export MASTER_ADDR=...
python scripts/run_kimi_k25.py train
```

`train` submits only. `prepare` does the download and the INT4 → BF16 dequantization; `full-train` runs both plus the training submit in one go, which is how the single-node 2-layer smoke test runs (`--model-name Kimi-K2.5-2layer --num-nodes 1`).

### 3.3 Multi-node fan-out

Bring up Ray on every node before launching, the same way as the other Kimi recipes:

```bash
# head
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats
# each worker
ray start --address=${MASTER_ADDR}:6379 --num-gpus 8 --node-ip-address ${WORKER_IP}
```

## 4. Script breakdown

The launcher builds the flags it passes to `train.py` as one f-string group per concern. The model shape comes from `scripts/models/kimi-k2-thinking.py`, which sets the MLA latent ranks (`q_lora_rank=1536`, `kv_lora_rank=512`, `qk_head_dim=128`, `qk_pos_emb_head_dim=64`, `v_head_dim=128`), the MoE routing (384 experts, top-8, sigmoid pre-softmax scoring, FP32 router, `--moe-router-topk-scaling-factor 2.827`), and RoPE (`--rotary-base 50000`, `--rotary-scaling-factor 64.0`). The K2.5 recipe then layers the following on top:

- **Checkpoints** — the dual checkpoint (INT4 actor via `--hf-checkpoint`, BF16 reference via `--ref-load`) together with `--megatron-to-hf-mode bridge` and `--model-name kimi_k25`.
- **Rollout and eval** — GRPO sampling and periodic AIME evaluation (covered in §5.2).
- **Parallelism** — the layout and recomputation (§5.1).
- **Algorithm and optimizer** — GRPO plus CPU-offloaded Adam (§5.2, §5.4).
- **SGLang** — the colocated rollout engine (§5.3).

The job runs colocated (`--colocate`) across `--num-nodes` nodes, 32 by default (`--actor-num-nodes 32 --actor-num-gpus-per-node 8`), with `--update-weight-buffer-size 2147483648`.

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
--prompt-data <data-dir>/dapo-math-17k/dapo-math-17k.jsonl
--input-key prompt --label-key label
--apply-chat-template
--rollout-shuffle --balance-data
--rm-type deepscaler

--num-rollout 3000
--rollout-batch-size 32
--n-samples-per-prompt 8
--rollout-max-response-len 16384
--rollout-temperature 1

--global-batch-size 256
--use-dynamic-global-batch-size
```

Evaluation is off by default; `--enable-eval` runs it every 20 steps against AIME-2024, sampling 16 responses per prompt:

```bash
--eval-interval 20
--eval-prompt-data aime <data-dir>/aime-2024/aime-2024.jsonl
--n-samples-per-eval-prompt 16
--eval-max-response-len 16384
--eval-top-p 1
```

### 5.3 Rollout & SGLang

The rollout engine is colocated with training, spanning 8 GPUs per engine with 8-way expert parallelism:

```bash
--rollout-num-gpus-per-engine 8
--sglang-mem-fraction-static 0.7
--sglang-ep-size 8
--sglang-server-concurrency 1024
--sglang-cuda-graph-bs 1 2 4 8 16 24 32 40 48 56 64 72 80 88 96 104 112 120 128
--use-rollout-routing-replay
```

The engine size and expert-parallel size both follow `--num-gpus-per-node`.

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

- [INT4 QAT](/advanced/int4-qat)
- [PD Disaggregation](/advanced/pd-disaggregation)
- [P2P Weight Transfer](/advanced/p2p-weight-transfer)
- [Fault Tolerance](/advanced/fault-tolerance)
- [Kimi K2](/models/kimi/kimi-k2): sibling recipe; K2.5 reuses the K2-Thinking architecture.
