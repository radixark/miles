---
title: Kimi K2
description: Launch recipes for Kimi-K2-Instruct and Kimi-K2-Thinking — 32 nodes × 8 GPU.
---

# Kimi K2

## 1. Model Introduction

[Kimi-K2](https://moonshotai.github.io/Kimi-K2/) is a state-of-the-art MoE language model from Moonshot AI with 32 B activated parameters and 1 T total parameters.

**Key highlights:**

- **Trillion-parameter MoE**: 1 T total / 32 B active per token, 61 layers (1 dense + the rest MoE), MLA attention shaped like DeepSeek-V3.
- **Instruct and Thinking variants**: Instruct is the general-purpose chat / agentic post-train; Thinking adds step-by-step reasoning with a 256 K context and ships in native INT4.
- **DeepSeek-V3-shaped architecture**: miles loads it through the DeepSeek-V3 path (one `sed` away), reusing the conversion + parallelism plumbing.
- **INT4 QAT target**: Kimi-K2-Thinking is the canonical reference recipe for INT4 QAT in miles.

## 2. Supported Variants

| Model | Active / Total | HF ID |
|---|---|---|
| Kimi-K2-Instruct | 32 B / 1 T | [moonshotai/Kimi-K2-Instruct](https://huggingface.co/moonshotai/Kimi-K2-Instruct) |
| Kimi-K2-Thinking | 32 B / 1 T | [moonshotai/Kimi-K2-Thinking](https://huggingface.co/moonshotai/Kimi-K2-Thinking) |

## 3. Environment Setup

### 3.1 Required env vars

```bash
export BASE_DIR=<shared FS path, reachable from every node>
export MASTER_ADDR=<head node IP>
```

Both are referenced but never set inside the scripts — export them yourself before launch.

### 3.2 Download model + datasets

```bash
hf download moonshotai/Kimi-K2-Instruct --local-dir $BASE_DIR/Kimi-K2-Instruct
# or, for Thinking (FP8 by default):
hf download moonshotai/Kimi-K2-Thinking --local-dir $BASE_DIR/Kimi-K2-Thinking-fp8

hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir $BASE_DIR/dapo-math-17k
hf download --repo-type dataset zhuzilin/aime-2024     --local-dir $BASE_DIR/rl_data/aime-2024
```

### 3.3 HF → Megatron `torch_dist` conversion

Convert across 4 nodes (mirror the DeepSeek-V3 procedure):

```bash
cd /root/miles
source scripts/models/kimi-k2.sh   # or kimi-k2-thinking.sh
PYTHONPATH=/root/Megatron-LM/ torchrun \
   --nproc-per-node 8 \
   --master-addr ${MASTER_ADDR} --master-port 12345 \
   --nnodes=4 --node-rank ${NODE_RANK} \
   tools/convert_hf_to_torch_dist.py \
   ${MODEL_ARGS[@]} \
   --hf-checkpoint $BASE_DIR/Kimi-K2-Instruct/ \
   --save          $BASE_DIR/Kimi-K2_torch_dist/
```

## 4. Launch

### 4.1 Quick start

```bash
cd /root/miles
export BASE_DIR=...; export MASTER_ADDR=...

bash scripts/run-kimi-k2-Thinking.sh   # or run-kimi-k2-Instruct.sh
```

Both launchers submit to an **already-running Ray cluster** (`ray job submit ...`); neither runs `ray start --head` itself.

### 4.2 Multi-node fan-out

Bring up Ray on every node before launching:

```bash
# head
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats
# each worker
ray start --address=${MASTER_ADDR}:6379 --num-gpus 8 --node-ip-address ${WORKER_IP}
```

## 5. Recipe Configuration

### 5.1 Parallelism

Identical for both Instruct and Thinking:

| TP | PP | CP | EP | expert-TP | `decoder-last-pipeline-num-layers` | `max_tokens_per_gpu` | GPUs |
|---|---|---|---|---|---|---|---|
| 8 | 8 | 4 | 32 | 1 | 5 | 16384 | 256 (32 × 8) |

Both scripts pass `--actor-num-nodes 32 --actor-num-gpus-per-node 8 --colocate --update-weight-buffer-size $((4*512*1024*1024))` to `train.py`.

### 5.2 Algorithm

| Script | Advantage | TIS |
|---|---|---|
| Instruct | GRPO (`--eps-clip 0.2 --eps-clip-high 0.28`) | – |
| Thinking | GRPO (`--eps-clip 0.2 --eps-clip-high 0.28`) | `--use-tis` |

Both use `--use-kl-loss --kl-loss-coef 0.00 --kl-loss-type low_var_kl --entropy-coef 0.00`.

Rollout shape (both):

```bash
--rm-type math
--num-rollout 100
--rollout-batch-size 128
--n-samples-per-prompt 8
--rollout-max-response-len 32768   # Instruct
--rollout-max-response-len 16384   # Thinking
--over-sampling-batch-size 256
--dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std
--num-steps-per-rollout 4
--balance-data
```

DAPO-style dynamic sampling is on by default in both scripts.

### 5.3 Rollout & SGLang

Identical for both:

```bash
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 16
   --sglang-mem-fraction-static 0.7

   # dp attention
   --sglang-enable-dp-attention
   --sglang-dp-size 8
   --sglang-moe-dense-tp-size 1
   --sglang-enable-dp-lm-head

   --sglang-ep-size 16

   # deepep — commented out in both scripts
   # --sglang-enable-deepep-moe
   # --sglang-deepep-mode auto

   --sglang-server-concurrency 1024
)
```

Megatron-side `--moe-enable-deepep` and `--moe-token-dispatcher-type flex` are **on** in the Instruct script but **commented out** in the Thinking script.

### 5.4 Optimizer

CPU Adam is enabled in both:

```bash
--optimizer-cpu-offload
--overlap-cpu-optimizer-d2h-h2d
--use-precision-aware-optimizer
```

### 5.5 Notable quirks

- Instruct loads the BF16 HF checkpoint by default (FP8 commented out) and reads eval data from `$BASE_DIR/rl_data/`; Thinking loads the FP8 HF checkpoint by default and reads eval data from `$BASE_DIR/`.
- `--global-batch-size 1024` is commented out in both scripts.

## 6. Pairs Well With

- [PD Disaggregation](../../advanced/pd-disaggregation.md)
- [P2P Weight Transfer](../../advanced/p2p-weight-transfer.md)
- [Fault Tolerance](../../advanced/fault-tolerance.md)
- [INT4 QAT](../../advanced/int4-qat.md)
