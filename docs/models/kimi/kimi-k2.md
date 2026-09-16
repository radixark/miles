---
title: Kimi K2
description: Launch recipes for Kimi-K2-Instruct and Kimi-K2-Thinking — 32 nodes × 8 GPU.
---
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
export MASTER_ADDR=<head node IP>
export MILES_SCRIPT_EXTERNAL_RAY=1
```

`MASTER_ADDR` reaches the ray workers as the torch-distributed rendezvous address; `MILES_SCRIPT_EXTERNAL_RAY=1` tells the launcher the cluster is already up (see §4.2). Paths are Typer flags: `--model-dir` (default `/root/models`), `--data-dir` (default `/root/datasets`), `--output-dir` (default `/root/shared_data`) — all three on a shared FS reachable from every node.

### 3.2 Download model + datasets

```bash
hf download moonshotai/Kimi-K2-Instruct --local-dir /root/models/Kimi-K2-Instruct
hf download moonshotai/Kimi-K2-Thinking --local-dir /root/models/Kimi-K2-Thinking-fp8

hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir /root/datasets/dapo-math-17k
hf download --repo-type dataset zhuzilin/aime-2024     --local-dir /root/datasets/aime-2024
```

### 3.3 HF → Megatron `torch_dist` conversion

Convert across 4 nodes (mirror the DeepSeek-V3 procedure):

```bash
cd /root/miles
MODEL_ARGS_LINE="$(python3 miles/utils/external_utils/model_args_utils.py kimi-k2)" || exit 1   # or kimi-k2-thinking
read -ra MODEL_ARGS <<< "${MODEL_ARGS_LINE}"
PYTHONPATH=/root/Megatron-LM/ torchrun \
   --nproc-per-node 8 \
   --master-addr ${MASTER_ADDR} --master-port 12345 \
   --nnodes=4 --node-rank ${NODE_RANK} \
   tools/convert_hf_to_torch_dist.py \
   ${MODEL_ARGS[@]} \
   --hf-checkpoint /root/models/Kimi-K2-Instruct/ \
   --save          /root/models/Kimi-K2_torch_dist/
```

## 4. Launch

### 4.1 Quick start

```bash
cd /root/miles
export MASTER_ADDR=...

MILES_SCRIPT_EXTERNAL_RAY=1 python scripts/run_kimi_k2.py --model-name Kimi-K2-Thinking   # or Kimi-K2-Instruct
```

Both variants are one launcher selected by `--model-name`. With `MILES_SCRIPT_EXTERNAL_RAY=1` it skips `ray start` and submits to an **already-running Ray cluster** (`ray job submit ...`).

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

Both variants pass `--actor-num-nodes 32 --actor-num-gpus-per-node 8 --colocate --update-weight-buffer-size 2147483648` to `train.py`; `--num-nodes` overrides the node count.

### 5.2 Algorithm

| `--model-name` | Advantage | TIS |
|---|---|---|
| `Kimi-K2-Instruct` | GRPO (`--eps-clip 0.2 --eps-clip-high 0.28`) | – |
| `Kimi-K2-Thinking` | GRPO (`--eps-clip 0.2 --eps-clip-high 0.28`) | `--use-tis` |

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

DAPO-style dynamic sampling is on by default for both variants.

### 5.3 Rollout & SGLang

Identical for both:

```bash
--rollout-num-gpus-per-engine 16
--sglang-mem-fraction-static 0.7

# dp attention
--sglang-enable-dp-attention
--sglang-dp-size 8
--sglang-moe-dense-tp-size 1
--sglang-enable-dp-lm-head

--sglang-ep-size 16

--sglang-server-concurrency 1024
```

SGLang DeepEP (`--sglang-moe-a2a-backend deepep`, `--sglang-deepep-mode`) is not enabled for either variant. Megatron-side `--moe-enable-deepep` and `--moe-token-dispatcher-type flex` are **on** for `Kimi-K2-Instruct` and **off** for `Kimi-K2-Thinking`.

### 5.4 Optimizer

CPU Adam is enabled in both:

```bash
--optimizer-cpu-offload
--overlap-cpu-optimizer-d2h-h2d
--use-precision-aware-optimizer
```

### 5.5 Notable quirks

- Instruct loads the BF16 HF release, Thinking the FP8 one (`Kimi-K2-Thinking-fp8`). Their `torch_dist` reference directories differ too — `Kimi-K2_torch_dist` vs `Kimi-K2-Thinking_torch_dist`.
- Neither variant passes `--global-batch-size`; the batch is driven by `--rollout-batch-size` and `--num-steps-per-rollout`.

## 6. Pairs Well With

- [PD Disaggregation](/advanced/pd-disaggregation)
- [P2P Weight Transfer](/advanced/p2p-weight-transfer)
- [Fault Tolerance](/advanced/fault-tolerance)
- [INT4 QAT](/advanced/int4-qat)
