---
title: DeepSeek V4 Flash
description: Launch recipe for DeepSeek-V4-Flash (284 B) — sparse-MLA + DSA indexer + KV compressors, FP8 rollout / BF16 train, 8-node H200.
---

# DeepSeek V4 Flash

## 1. Model Introduction

[DeepSeek-V4-Flash](https://huggingface.co/sgl-project/DeepSeek-V4-Flash-FP8) is the publicly-distributed FP8 repackage of `deepseek-ai/DeepSeek-V4-Flash` — a 284 B-parameter MoE with a substantially different attention stack from V3/R1. The miles + Megatron-Core integration is shipped together in the [`radixark/miles#1045`](https://github.com/radixark/miles/pull/1045) and [`radixark/Megatron-LM#28`](https://github.com/radixark/Megatron-LM/pull/28) pull requests, plus the published image `radixark/miles:deepseek-v4`.

**Key highlights:**

- **Sparse multi-head latent attention (sparse-MLA)**: low-rank Q (`q_lora_rank=1024`), single-head latent KV (`head_dim=512`), grouped output projection (8 groups, LoRA rank 1024). A learned **DSA indexer** picks topk=512 KV per query at runtime.
- **KV compressors**: per-layer compression ratios (2 / 4 / 128) reduce the indexer's effective KV size; compressor is FP32-stable with optional FP8 QAT.
- **Hyper-connection (HC) routing**: each layer expands hidden state into `hc_mult=4` parallel streams and recombines via sinkhorn-normalised mixing. Pipeline-parallel buffers are 4-D `[s, b, hc_mult, d]` instead of 3-D.
- **YaRN RoPE base 160000**, per-head learnable attention sinks, MoE with hash-routed first `dsv4_n_hash_layers` layers.
- **FP8 weights with simulated FP8 QAT** on indexer and compressor activations; default training is BF16 on the cast checkpoint, default rollout is FP8 in SGLang with `--sglang-attention-backend compressed`.

## 2. Supported Variants

| Model | Active / Total | HF ID |
|---|---|---|
| DeepSeek-V4-Flash-FP8 | ~37 B / 284 B | [sgl-project/DeepSeek-V4-Flash-FP8](https://huggingface.co/sgl-project/DeepSeek-V4-Flash-FP8) |
| DeepSeek-V4-Flash-FP8-4layer | smoke-only | [Pinaster/DeepSeek-V4-Flash-FP8-4layer](https://huggingface.co/Pinaster/DeepSeek-V4-Flash-FP8-4layer) |

DeepSeek-V4-Pro (1.6 T) is roadmap per [`radixark/miles#1046`](https://github.com/radixark/miles/issues/1046) and is **not yet validated** through this recipe.

## 3. Environment Setup

### 3.1 Required env vars

The published image `radixark/miles:deepseek-v4` (cu129 x86, H200/B200) preconfigures the launcher's path defaults via `MILES_SCRIPT_*` env vars:

```bash
MILES_SCRIPT_MODEL_DIR=/cluster_public/miles_data/models       # shared FS
MILES_SCRIPT_MODEL_LOCAL_DIR=/node_public/miles_data/models_local  # per-node NVMe
MILES_SCRIPT_DATA_DIR=/cluster_public/miles_data/datasets
MILES_SCRIPT_OUTPUT_DIR=/cluster_public/miles_data/outputs
```

Override with `--model-dir`, `--model-local-dir`, `--data-dir`, `--save-dir` on the Python launcher when your cluster mounts a different layout.

GB300 (cu130 / arm64) docker image is not yet published.

### 3.2 Download model + datasets

```bash
# inside the radixark/miles:deepseek-v4 container
hf download sgl-project/DeepSeek-V4-Flash-FP8 --local-dir /root/models/DeepSeek-V4-Flash-FP8
hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir /root/datasets/dapo-math-17k
hf download --repo-type dataset zhuzilin/aime-2024     --local-dir /root/datasets/aime-2024
```

The Python launcher's `prepare-download` subcommand does the dataset fetch automatically; pass `--hf-checkpoint <path>` to skip the model download when the FP8 weights are already on a shared filesystem.

### 3.3 HF → Megatron `torch_dist` conversion

Convert FP8 → BF16 first, then run distributed conversion:

```bash
cd /root/miles
python tools/fp8_cast_bf16.py \
   --input-fp8-hf-path  /root/models/DeepSeek-V4-Flash-FP8 \
   --output-bf16-hf-path /root/models/DeepSeek-V4-Flash-FP8-bf16/

source scripts/models/deepseek-v4-flash.sh
PYTHONPATH=/root/Megatron-LM torchrun \
   --nproc-per-node 4 --nnodes 8 \
   --master-addr ${MASTER_ADDR} --master-port 12345 \
   --node-rank ${NODE_RANK} \
   tools/convert_hf_to_torch_dist.py \
   ${MODEL_ARGS[@]} \
   --tensor-model-parallel-size 1 \
   --pipeline-model-parallel-size 8 \
   --expert-model-parallel-size 4 \
   --decoder-first-pipeline-num-layers 7 \
   --decoder-last-pipeline-num-layers 6 \
   --hf-checkpoint /root/models/DeepSeek-V4-Flash-FP8-bf16/ \
   --save          /root/models/DeepSeek-V4-Flash-FP8_torch_dist/
```

`--nproc-per-node 4` (not 8) is intentional — see [§5.5 Notable quirks](#55-notable-quirks). The Python launcher's `prepare-spmd` subcommand drives the same conversion; pair it with `--num-gpus-per-node 4` when invoked standalone to keep the conversion `world_size ≤ num_layers`.

## 4. Launch

### 4.1 Quick start

```bash
# Single-node 4-layer smoke (pipeline-only sanity check, won't generate readable output)
cd /root/miles
python scripts/run_deepseek_v4.py full-train \
   --model-name DeepSeek-V4-Flash-FP8-4layer \
   --num-nodes 1 --num-gpus-per-node 8

# Production 8-node Flash run
cd /root/miles
python scripts/run_deepseek_v4.py full-train \
   --model-name DeepSeek-V4-Flash-FP8 \
   --num-nodes 8 --num-gpus-per-node 8
```

`full-train` chains `prepare-download → prepare-single → prepare-spmd → prepare-cp → train`. Each stage has a sentinel-based skip so you can re-run safely after the first invocation.

### 4.2 Multi-node fan-out

The Python launcher manages Ray internally — start each pod with `radixark/miles:deepseek-v4` and a working `/cluster_public` (or equivalent shared FS), then on the head node:

```bash
ray start --head --num-gpus 8 --disable-usage-stats
# … then on each worker:
ray start --address=${HEAD_IP}:6379 --num-gpus 8 --disable-usage-stats
```

Or set `MILES_SCRIPT_EXTERNAL_RAY=1` and `RAY_ADDRESS=…` to point the launcher at an existing Ray cluster (e.g., one that an orchestration layer already brought up). When `RAY_ADDRESS` is unset the launcher boots a local Ray head.

## 5. Recipe Configuration

### 5.1 Parallelism

`scripts/run_deepseek_v4.py::_get_parallel_config` validates the layout per cluster shape:

| Hardware | Nodes × GPUs | TP | PP | EP | expert-TP | `max_tokens_per_gpu` | Pipeline layout |
|---|---|---|---|---|---|---|---|
| H200 | 7 × 8 = 56 | 8 | 7 | 8 | 1 | 2048 | `E,t*4\|t*7\|t*7\|t*7\|t*7\|t*7\|t*4,L` |
| H200 | 8 × 8 = 64 | 8 | 8 | 8 | 1 | 2048 | first 4 / last 3 layers |
| GB300 | 7 × 4 = 28 | 4 | 7 | 4 | 1 | 2048 | first 7 / last 6 layers |
| GB300 | 8 × 4 = 32 (CP=1) | 8 | 4 | 8 | 1 | 2048 | first 11 / last 10 layers |
| GB300 | 8 × 4 = 32 (CP=2) | 2 | 8 | 4 | 1 | 2048 | first 4 / last 3 layers |

7 H200 nodes is the documented minimum — `_get_parallel_config` raises `NotImplementedError` for any other GPU count.

### 5.2 Algorithm

GRPO with `--eps-clip 0.2 --eps-clip-high 0.28 --kl-loss-coef 0.00 --kl-loss-type low_var_kl --entropy-coef 0.00 --advantage-estimator grpo`. `--moe-router-freeze-gate` and `--freeze-e-score-correction-bias` are required and asserted on the mcore side — bias-update during RL is forbidden.

### 5.3 Rollout & SGLang

```bash
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 8
   --sglang-tp-size 8
   --sglang-dp-size 8
   --sglang-ep-size 8
   --sglang-enable-dp-attention
   --sglang-attention-backend compressed       # V4 sparse-MLA backend
   --sglang-page-size 256
   --sglang-max-running-requests 64
   --sglang-chunked-prefill-size 8192
   --sglang-mem-fraction-static 0.5            # leave headroom for Megatron during wake_up
   --use-rollout-routing-replay                # MoE routing replay (R3)
   --use-miles-router                          # miles router fronts /generate
)
```

Required env vars (the launcher sets these for you): `SGLANG_SKIP_CHECKPOINT_LOAD_CHECK=1`, `SGLANG_DSV4_FP4_EXPERTS=0`, `MILES_HACK_TRAIN_TORCH_DETERMINISTIC=1`, `NCCL_ALGO=Ring`.

Megatron side: `--qkv-format bshd` (V4 needs `bshd` with CP-aware data slicing). The DSA indexer additionally supports replay via `--use-rollout-indexer-replay` (off by default).

### 5.4 Optimizer

```bash
--optimizer adam
--lr 1e-6 --lr-decay-style constant
--weight-decay 0.1
--adam-beta1 0.9 --adam-beta2 0.98
--accumulate-allreduce-grads-in-fp32
--attention-softmax-in-fp32
--clip-grad 1.0                # Megatron default; not overridden by the launcher
```

`--low-memory-resume` (off by default) puts optimizer states on CPU during ckpt resume to avoid OOM on the very first iteration.

### 5.5 Notable quirks

- **Conversion world_size ≤ num_layers.** `tools/convert_hf_to_torch_dist.py` asserts `world_size ≤ args.num_layers`. Flash has 43 layers, so 8 × 8 = 64 ranks fails. Run `prepare-spmd` (or the standalone `torchrun`) with `--num-gpus-per-node 4` (32 ranks) and let the subsequent `prepare-cp` / `train` stages resume at 8 GPUs/node.
- **`--hf-checkpoint <path>` skips download but `prepare-cp` still rsyncs `{model_dir}/{model_name}/`.** When you point at pre-staged FP8 weights outside `{model_dir}`, the launcher's second rsync stat()s a missing source (rsync exit 23). Symlink the expected path: `ln -s /path/to/DeepSeek-V4-Flash-FP8 {model_dir}/DeepSeek-V4-Flash-FP8`.
- **`train.py` argparse rejects `--wandb-run-name`.** Only `--wandb-project` / `--wandb-group` / `--wandb-key` are supported. Pass them via `--extra-args`.
- **Bursty rollouts can cycle stale TCP connections in `httpx`'s pool.** Under default `httpx.AsyncClient` settings, miles' `_http_client` may reuse a server-side-closed socket and raise `httpx.ReadError` (or trip the miles router circuit breaker, surfacing as a 500 storm). Patch both `httpx.Limits(...)` constructions in `miles/utils/http_utils.py` to set `max_keepalive_connections=0` (covers `init_http_client` and `_HttpPosterActor.__init__`).
- **GPU memory is tight during sglang `wake_up`.** With image defaults (`--sglang-mem-fraction-static 0.7 --train-memory-margin-bytes 3221225472`) a wake_up step can hit `cuMemCreate CUDA_ERROR_OUT_OF_MEMORY` stochastically on 141 GB H200s. Lower the fraction to 0.5 and raise the train margin to 8 GiB (`8589934592`).
- **Save-storm pod death.** With the default `--save-interval 20`, every save triggers 64-rank concurrent writes to per-pod `/root/models/...` overlay; one pod's overlay can fill enough to trip kubelet/Ray health checks and the actor dies. For smoke runs use `--skip-saving`. For real runs route saves to a shared filesystem via `--save-dir`.
- **Truncation-drift trap on small response budgets.** `--rollout-max-response-len 256` (the launcher's gsm8k default) leads to ~64 % truncation, low gradient SNR per step, and a fast-growing `train/train_rollout_logprob_abs_diff` (12× over 19 steps observed). Bump to ≥ 4096 for any run intended to learn.
- **Custom `transformers` patch.** miles ships `with_transformers_patch()` (`miles/utils/transformers_patch.py`) so HF's `AutoConfig.from_pretrained` recognises `model_type=deepseek_v4` / `deepseek_ref` until support lands upstream. The 4-layer smoke variant is auto-renamed `deepseek_v4 → deepseek_ref` for SGLang compatibility.

## 6. Pairs Well With

- [FP8 & Low Precision](../../advanced/fp8-low-precision.md)
- [Miles Router](../../advanced/miles-router.md) — `--use-miles-router` is required for V4 rollouts.
- [Architecture Support](../../advanced/architecture-support.md) — the V4 plugin lives under `miles_plugins/models/deepseek_v4/`.
- [Fault Tolerance](../../advanced/fault-tolerance.md) — `--use-fault-tolerance` is on by default in the launcher.
