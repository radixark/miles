---
title: Kimi-K3.1
description: W4A16 LoRA RL recipe for Kimi-K3.1 (Kimi-K3 plus a DSA indexer) on 32 H200 or 16 B300, training on the checkpoint's MXFP4 experts as published.
---
## 1. Model Introduction

[Kimi-K3.1](https://huggingface.co/moonshotai/Kimi-K3.1) is [Kimi-K3](/models/kimi/kimi-k3) with a DSA lightning indexer on each of its 24 MLA layers. miles trains a rank-32 LoRA on its text decoder with RL, on 4 × 8 H200 or 2 × 8 B300, half the GPUs of the K3 recipe.

**Key highlights:**

- **K3's backbone**: 93 layers that mix KDA (69) and MLA (24) attention, and an 896-expert latent MoE at top-16 with a shared expert.
- **DSA indexer on every MLA layer**: each query attends its top 2048 keys, picked by 32 index heads of width 128 with LayerNorm keys and no RoPE. The indexer stays frozen.
- **MXFP4 routed experts**: the checkpoint is compressed-tensors MXFP4 (1.5 TB), and both the trainer and the rollout read it as published, with no BF16 copy and no `torch_dist` conversion.
- **W4A16 LoRA**: the trainer keeps the frozen experts as MXFP4 bytes and dequantizes them per layer; only a BF16 adapter trains, and a weight sync sends only the adapter.

## 2. Supported Variants

| `--model-name` | What it is | HF source |
|---|---|---|
| `Kimi-K3.1` | text decoder, 93 layers, 896 routed experts | [moonshotai/Kimi-K3.1](https://huggingface.co/moonshotai/Kimi-K3.1) |

The vision tower stays in the checkpoint but is not trained (text-only RL). One launcher per hardware trains it, both sharing the workload, loss and checks in `scripts/kimi_k31_w4a16_lora_common.py`:

| Launcher | GPUs |
|---|---|
| `scripts/run_kimi_k31_w4a16_lora_32xh200.py` | 32 (4 × 8 H200) |
| `scripts/run_kimi_k31_w4a16_lora_16xb300.py` | 16 (2 × 8 B300) |

## 3. Environment Setup

### 3.1 Image and storage

```bash
docker pull radixark/miles:kimi-k31
```

Run it on every node, on InfiniBand. The image (multi-arch; arm64 hosts pull arm64) carries miles at `/root/miles` from [radixark/miles#4667](https://github.com/radixark/miles/pull/4667), which adds these recipes on top of the Kimi-K3.1 model support in [radixark/miles#4666](https://github.com/radixark/miles/pull/4666); Megatron-LM (`radixark/Megatron-LM` `miles-main` at `eb547f33e`); and the SGLang branch of [section 3.3](#33-sglang) at `/sgl-workspace/sglang/python`.

`--model-dir` must hold the checkpoint at the same path on every node, on shared storage or as one copy per node. `--save-dir` must be shared by all nodes: every training rank writes its own adapter shard into the directory rank 0 creates.

### 3.2 Download

`prepare-download` downloads the release to `<model-dir>/Kimi-K3.1` and dapo-math-17k plus AIME-2024 to `<data-dir>`:

```bash
hf download moonshotai/Kimi-K3.1 --local-dir /root/models/Kimi-K3.1
```

Nothing is converted. The trainer loads the native checkpoint in raw bridge mode (`--hf-checkpoint` and `--ref-load` both point at it), and the engine serves it on SGLang's Marlin MXFP4 MoE runner.

### 3.3 SGLang

The image carries the [`sglang-miles-kimi-k31`](https://github.com/sgl-project/sglang/tree/sglang-miles-kimi-k31) branch (at `35708f7`), which adds Kimi-K3.1 support and the rollout changes this recipe relies on:

- a BF16 KV cache for K3.1's DSA attention;
- MoE LoRA on the Marlin MXFP4 runner under expert parallelism, with each rank keeping only its own experts' rows of a streamed adapter;
- LoRA under attention DP with attention TP above 1;
- the DSA top-k broadcast (`SGLANG_DSA_TOPK_BROADCAST`);
- the DSA prefill padding-row fix.

These are not in `sglang-miles` yet.

## 4. Launch

### 4.1 32 × H200

Start one Ray cluster across the 4 nodes, then launch from node 0:

```bash
# node 0
ray start --head --port 6379 --num-gpus 8
# other nodes
ray start --address <node-0-address>:6379 --num-gpus 8

cd /root/miles
export MILES_SCRIPT_EXTERNAL_RAY=1
python scripts/run_kimi_k31_w4a16_lora_32xh200.py prepare-download --model-dir /root/models --data-dir /root/datasets
python scripts/run_kimi_k31_w4a16_lora_32xh200.py train \
  --model-dir /root/models --data-dir /root/datasets \
  --num-rollout 100 --save-dir <shared-dir>
```

Over 44 steps of the validated run:

| Launcher | GPUs | Step | Rollout | Training | Weight sync | `train/train_rollout_logprob_abs_diff` | `train/train_rollout_kl` |
|---|---|---|---|---|---|---|---|
| `run_kimi_k31_w4a16_lora_32xh200.py` | 32 H200 | 1422–2010 s (mean 1715) | about 20 min | about 9 min | about 3 min | 0.014–0.018 | 4.4e-4–8.2e-4 |

`train/grad_norm` stays around 0.011, with isolated spikes (0.026, and 0.49 once). `rollout/raw_reward` rises from 0.61 (mean of steps 0–4) to 0.74 (steps 39–43) while the mean response shortens from 1890 to 1370 tokens and truncation falls from 36% to 18%; deepscaler scores a truncated response 0 in practice, so part of the gain is fewer truncations. GPU memory peaks at 130 GiB per card and host memory at 1483 GB per node.

### 4.2 16 × B300

Start one Ray cluster across the 2 nodes as above, then launch from node 0. With the checkpoint on node-local disks, run `prepare-download` on every node:

```bash
cd /root/miles
export MILES_SCRIPT_EXTERNAL_RAY=1
python scripts/run_kimi_k31_w4a16_lora_16xb300.py prepare-download --model-dir /scratch/models --data-dir /scratch/datasets
python scripts/run_kimi_k31_w4a16_lora_16xb300.py train \
  --model-dir /scratch/models --data-dir /scratch/datasets \
  --num-rollout 100 --save-dir <shared-dir>
```

The recipe keeps the H200 recipe's per-stage layout on half the GPUs. Each card holds twice the H200 share of the model, so the host has no room for SGLang's copy of the base next to the trainer's offloaded weights (about 1 TB a node on their own), and the base is re-sent at every sync. An earlier version of this recipe (32 prompts per step, refilled 16 groups at a time) ran 15 steps on 16 B300 with `train/train_rollout_logprob_abs_diff` 0.014–0.015 and `train/train_rollout_kl` 3.6e-4–3.8e-4, in the H200 range; the current 16-prompt version has not run.

### 4.3 Save and resume

Every 5 rollouts and after the last, the run saves the adapter shards and `adapter_config.json` under `<save-dir>/<run-id>` (48 GB each at rank 32), without the Adam state (about 294 GB); `--skip-saving` turns saving off. To continue from a save, pass a directory holding that save's `adapter_config.json` and `adapter_megatron_rank*.pt`:

```bash
--extra-args "--lora-adapter-path <adapter-dir> --no-load-optim"
```

The shards are per training rank, so the training layout must stay the same.

## 5. Recipe Configuration

### 5.1 Parallelism

| Launcher | TP | PP | CP | EP | expert-TP | first/last stage layers | `max_tokens_per_gpu` |
|---|---|---|---|---|---|---|---|
| `run_kimi_k31_w4a16_lora_32xh200.py` | 4 | 4 | 2 | 8 | 1 | 24/21 | 8192 |
| `run_kimi_k31_w4a16_lora_16xb300.py` | 4 | 2 | 2 | 8 | 1 | 47/46 | 8192 |

`--sequence-parallel`, THD packing (`--use-dynamic-batch-size`) and full recomputation are on. The 8 ranks of a pipeline stage (TP4 × CP2) form its EP8 group.

### 5.2 Algorithm

[Score centering](/examples/infra-features/score-centering) (`--loss-type score_centering`) on the sampler's top-128 candidates (`--rollout-top-logprobs-num 128`), with per-token loss, no std normalization and no KL or entropy term. Sampling is unfiltered (temperature 1, top-p 1), which miles maps to the "selected" candidate mode, and the rollout log-probs replace the trainer's forward-only pass (`--use-rollout-logprobs`).

`--rm-type deepscaler` rewards dapo-math-17k answers, with responses up to 3072 tokens (`--rollout-max-response-len`) and the tokenizer's default (max) thinking effort. Each rollout feeds one optimizer step:

| Launcher | Prompts × samples | Global batch | Dynamic-sampling filter |
|---|---|---|---|
| `run_kimi_k31_w4a16_lora_32xh200.py` | 32 × 16 | 512 | off |
| `run_kimi_k31_w4a16_lora_16xb300.py` | 16 × 16 | 256 | on, refilled 4 groups at a time |

The filter drops a group whose 16 rewards are all equal and draws new prompts until 16 groups remain; `rollout/raw_reward_unfiltered` keeps the mean reward before the filter.

[R3](/advanced/miles-router) (`--use-rollout-routing-replay`) is on; indexer replay is off, and the trainer selects the DSA top-k from its own indexer scores.

### 5.3 Rollout & SGLang

One engine spans all the GPUs, colocated with the trainer, with up to 256 concurrent requests:

```bash
--sglang-tp-size 32                     # 16 on B300
--sglang-ep-size 32                     # 16 on B300; whole experts per rank
--sglang-attn-dp-size 4                 # 2 on B300; one attention DP group of TP 8 per node
--sglang-moe-runner-backend marlin
--sglang-lora-backend triton
--sglang-kv-cache-dtype bfloat16
--sglang-max-total-tokens 262144        # per DP group: 64 requests x (3072 + 1024 tokens); 524288 on B300
--sglang-load-balance-method total_requests
```

The adapters are a BF16 bypass on both sides and are never merged. On H200, SGLang keeps a host copy of the base (`--lora-base-cpu-backup`), so a sync sends only the adapter: about 46 GiB at rank 32 in 2 GiB buckets, 99% of it the per-expert factors, of which each engine rank keeps only its own experts' rows. On B300 the base is re-sent at every sync as well.

### 5.4 Optimizer

Adam at `--lr 1e-5`, constant, with weight decay 0.1 and betas 0.9 / 0.98. 80% of the optimizer state lives on the CPU:

```bash
--optimizer-cpu-offload --optimizer-offload-fraction 0.8
--overlap-cpu-optimizer-d2h-h2d
--use-precision-aware-optimizer
```

### 5.5 LoRA

Rank 32 and alpha 64 on `all-linear`, which selects K3's HF defaults: the MLA query and KV down projections, the attention output projections, the dense and shared MLPs, and the routed experts. The experts share their outer factors (`--experts-shared-outer-loras`).

The frozen base would need about 180 GB per GPU in BF16 on 32 H200. The trainer instead keeps every routed expert as the checkpoint's MXFP4 bytes (`weight_packed` plus `weight_scale`, 0.53 B per parameter, about 47 GB per GPU) and dequantizes them exactly into BF16 inside the grouped GEMM, so a layer's BF16 experts live only through its forward and recompute.

### 5.6 Notable quirks

- **`--check-lora-weight-equal` is off**: it hashes every adapter tensor on the CPU, on the trainer and again on every engine rank, about 167 s of each 187 s sync. Pass it through `--extra-args` for a checked run.
- **`NCCL_RAS_ENABLE=0`**: with NCCL 2.30.7, the RAS thread segfaulted on every node right after the engine first released its memory.
- **DSA top-k broadcast**: with attention DP the launcher sets `SGLANG_DSA_TOPK_BROADCAST=1`, so every attention-TP rank of a DP group attends rank 0's DSA top-k; without it the ranks can pick different keys beyond the first 2048.
- **No dp-aware router**: the engine spreads requests over its DP groups (`--sglang-load-balance-method total_requests`). `sglang-router` 0.3.2 reads the engine's `dp_size`, which attention DP reports as 1, and would send every request to DP group 0.
- **Per-group KV budget**: under attention DP, `--sglang-max-total-tokens` is per DP group, while running requests, mamba slots and chunked prefill are engine-wide and split over the groups.
- **Indexer replay is off**: shipping 196 KB of top-k per token cost about 30 minutes a step after the rollout.
- **InfiniBand port down**: a port going down mid-run kills it with `IBV_WC_RETRY_EXC_ERR`; leave the port out with `--extra-env-vars "NCCL_IB_HCA=^<hca>"` before resuming.

## 6. Pairs Well With

- [Kimi-K3](/models/kimi/kimi-k3)
- [LoRA](/advanced/lora)
- [Rollout Routing Replay (R3)](/advanced/miles-router)
