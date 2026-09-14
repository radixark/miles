---
title: DeepSeek-V4.1 Flash
description: Launch recipe for DeepSeek-V4.1 with radixark/miles:deepseek-v41 — BF16 train / BF16 rollout, colocated, 4-node GB300 (16 GPUs) with optimizer state streamed to NVMe.
---
## 1. Model Introduction

[DeepSeek-V4.1 Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) is a sparse-attention Mixture-of-Experts model (`model_type: deepseek_v41` (the legacy `deepseek_v4.1` name is still accepted)): 40 decoder layers, 384 routed experts at top-6 plus one shared expert, fp8 dense weights at a 32-wide ue8m0 block scale with fp4 routed experts.

**Architecture.** Every layer keeps one 512-wide KV latent per position and uses it as both key and value for all 64 query heads, with no separate value projection. Each layer reads two stores that behave oppositely: compressed latents, produced only at source layers and shared forward, collapsing 2 positions into 1 in layers 2–19 and 1-to-1 from layer 20 on; and a 128-position sliding window, recomputed per layer from that layer's own activations, so it can never be shared but also never grows with context. The residual stream is four parallel copies mixed by a per-token doubly stochastic matrix; each sublayer's mixing coefficients are consumed by the next sublayer.

**Sparse retrieval.** KV source layers and index source layers are different lists — four of the former, eight of the latter. The four extra index layers produce no keys at all; they re-score layer 20's keys with their own query, so retrieval decisions are made twice as often as keys are stored. Each retrieving layer picks a top-512 candidate set, and from layer 20 on a coarser candidate-block pass bounds the positions later layers score.

**Engram.** An additive n-gram hash memory at two layers. Token ids are normalized before hashing, so " The", "the" and "THE" cannot fork into separate rows. Its two fp8 tables are the largest single block of weight in the checkpoint.

On the training side the Megatron plugin lives under `miles_plugins/models/deepseek_v41/`, the model definition is `scripts/models/deepseek-v4.1.py`, and the launcher is `scripts/run_deepseek_v41.py`. The plugin reproduces the model's fp4/fp8 quantization points in the training forward so the trainer scores what the served model computes, and carries the cross-layer state (mixing coefficients, source latents, top-k, candidate mask) inside the inter-layer hidden tensor so pipeline parallelism and full activation recompute work.

## 2. Supported Variants

| Model | Active / Total | Checkpoint |
|---|---|---|
| DeepSeek-V4.1 Flash | ~5 B dense + 6 of 384 experts / ~750 B (544 B experts + 197 B frozen engram) | [deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash), converted to a BF16 HF checkpoint on node-local NVMe (see [§3.2](#32-prepare-the-checkpoint)) |

`model_type` is `deepseek_v4.1`; the launcher selects the recipe with `--model-name DeepSeek-V4.1`.

## 3. Quick Start

The validated configuration is 4 nodes x 4 GB300 (16 GPUs), colocated rollout engines, BF16 training and BF16 rollout, optimizer state streamed to NVMe. Each step below runs inside the image on every node unless stated otherwise.

### 3.1 Pull the image

```bash
docker pull radixark/miles:deepseek-v41
```

The image is the `radixark/miles` dev image (arm64) with the DeepSeek-V4.1 SGLang fork installed at `/sgl-workspace/sglang/python` and this branch of miles at `/root/miles`. The pinned revisions are in `/sgl-workspace/sglang/.sglang_rev` and `/root/miles/.miles_rev` and in the image labels `dsv41.sglang-commit` and `dsv41.miles-commit`.

Start one container per node with the GPUs, the host network and a node-local NVMe volume mounted at the same path everywhere (`/scratch` below):

```bash
docker run -d --name miles --gpus all --network host --ipc host --shm-size 512g \
   -v /scratch:/scratch radixark/miles:deepseek-v41 sleep infinity
```

### 3.2 Prepare the checkpoint

The trainer loads a **BF16 HF checkpoint** directly (no `torch_dist` conversion). The served checkpoint is fp8 dense (32x32 ue8m0 scales) with fp4 experts, and has to be cast once:

- dense fp8 and expert fp4 weights dequantized to bf16;
- the two engram tables kept as fp8 bits plus e8m0 scales (about 197 GB; they are never upcast);
- MTP, DSpark and vision tensors dropped;
- a flattened `config.json` with `model_type: deepseek_v41` (the legacy `deepseek_v4.1` name is still accepted).

`tools/fp8_cast_bf16.py` covers the fp8 dense format only; the cast tool that also handles the fp4 experts and the engram layout (`dsv41_cut_cast.py --src <fp8 ckpt> --dst <bf16 ckpt> --config <flattened config.json> --layers 0,...,39`) is kept with the bring-up scripts. The result is about 1.2 TB and must be present on **every training node's local NVMe** at the same path, for example `/scratch/models/DeepSeek-V4.1-bf16`. Reserve a further ~1 TB per node for the streamed optimizer state.

### 3.3 Prepare the data

Once, on the head node:

```bash
cd /root/miles
python scripts/run_deepseek_v41.py prepare-data --task dapo_aime --data-dir /scratch/datasets
```

This downloads `dapo-math-17k` and `aime-2024`. Copy or re-run on the other nodes if `/scratch` is not shared.

### 3.4 Bring up the Ray cluster

```bash
# on node 0
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 4 --disable-usage-stats
# on every other node
ray start --address=${MASTER_ADDR}:6379 --node-ip-address ${WORKER_IP} --num-gpus 4 --disable-usage-stats
```

`ray status` on the head must list 16 GPUs before launching.

### 3.5 Launch

On the head node:

```bash
cd /root/miles
MILES_SCRIPT_EXTERNAL_RAY=1 python scripts/run_deepseek_v41.py train \
   --model-name DeepSeek-V4.1 \
   --hf-checkpoint /scratch/models/DeepSeek-V4.1-bf16 \
   --model-dir /scratch/models --data-dir /scratch/datasets \
   --num-nodes 4 --num-gpus-per-node 4 --hardware GB300 \
   --task dapo_aime --mode debug_minimal \
   --num-rollout 400 --rollout-batch-size 16 --n-samples-per-prompt 8 \
   --rollout-max-response-len 2048 --max-tokens-per-gpu 2048 \
   --load-from-hf --disk-offload --offload-disk-dir /scratch/offload \
   --colocate-memory-peak-device cpu \
   --rollout-gpus-per-engine 8 --sglang-mem-fraction-static 0.7 \
   --pp-size 2 --recompute full \
   --sglang-engram-host-table --sglang-cuda-graph --sglang-radix-cache \
   --no-check-weight-update \
   --extra-env-vars "NCCL_SOCKET_IFNAME=eth0 GLOO_SOCKET_IFNAME=eth0" \
   --extra-args "--stream-optimizer-state-moment-dtype bf16"
```

Replace `eth0` with the cluster interface. Add `--wandb-team`, `--wandb-project` and `--wandb-group` through `--extra-args` to log to Weights & Biases.

Three flags are not optional in this configuration:

| Flag | Why |
|---|---|
| `--sglang-engram-host-table` | Colocated engines release all GPU state when they sleep; the engram tables are constants that the weight update does not carry. Without the shared host table they come back empty and `train/train_rollout_logprob_abs_diff` jumps from ~0.02 to ~10. |
| `--disk-offload --offload-disk-dir` | Full-parameter Adam state for the full model does not fit in GPU or host memory on 16 GPUs; it is streamed to node-local NVMe between steps. |
| `--load-from-hf` | The BF16 HF checkpoint is mapped through the mbridge at startup, so no multi-node `torch_dist` conversion is needed. |

### 3.6 What to expect

| Signal | Expected |
|---|---|
| Step 0 `train/train_rollout_logprob_abs_diff` | ~0.02 nats (0.0246 / 0.0247 in the two validated runs) |
| `train/train_rollout_kl` | 0.0012-0.0017, flat over training |
| `rollout/raw_reward` | ~0.5 at step 0, rising above 0.75 within 80 steps |
| Step time | ~11 min per step (2 min weight sync + rollout, ~8 min train + log-prob) |
| Trainer peak GPU memory | ~218 GB of 277 GB per GPU |

`rollout/rewards` is the group-normalised advantage and is 0 by construction; watch `rollout/raw_reward`.

### 3.7 Launcher path defaults

| Flag | Default | Use |
|---|---|---|
| `--hf-checkpoint` | required | BF16 HF checkpoint; with `--load-from-hf` it is also the reference model (`--ref-load`) |
| `--data-dir` | `/root/datasets` | HF datasets (dapo-math-17k, aime-2024, gsm8k) |
| `--model-dir` | `/root/models` | parent of `DeepSeek-V4.1_torch_dist` when not loading from HF |
| `--save-dir` | `/root/models` | training checkpoints under `{save-dir}/{run-id}/checkpoints/` (`--no-skip-saving`) |
| `--offload-disk-dir` | unset | node-local NVMe directory for the streamed optimizer state and the trainer's weight backups |

Every option also binds to `MILES_SCRIPT_<FIELD_NAME_UPPER>` (CLI flag > env var > default). At TP 8 the launcher adds `--make-vocab-size-divisible-by 32` automatically. On GB300 it exports `NCCL_CUMEM_ENABLE=1`; with it off the expert-parallel all-to-all fails across nodes.

## 4. Recipe Configuration

### 4.1 Megatron parallelism

Tensor, pipeline, context, expert and sequence parallelism are all supported. Full-model layouts validated on 4 x 4 GB300 (16 GPUs, colocated, 8-GPU engines):

| TP | PP | EP | Recompute | Decode CUDA graphs | Step time | Trainer peak GPU (of 276.6 GB) | Notes |
|---|---|---|---|---|---|---|---|
| 4 | 1 | 16 | selective (moe, mlp) | off | 15.7 min | 249.5 GB | 120+ steps validated |
| 4 | 2 | 8 | full (uniform, 1 layer) | on | 11.2 min | 218.1 GB | recommended; identical step-0 parity (abs_diff 0.0246 vs 0.0247) |

The second layout is the one in [§3.5](#35-launch). Its speedup comes from the rollout side (decode CUDA graphs cut the rollout wait from 493 s to 200 s per step); train time is unchanged because PP 2 halves the layers per rank while full recompute adds a forward.

Tensor, pipeline, context, expert and sequence parallelism can be combined freely; TP/CP/PP/EP combinations were cross-validated against each other and agree to the bf16 floor.

An 8 x 8 H200 bring-up (TP 8, 16-GPU engines) also trains the full model; on SM90 the DeepGEMM paged indexer kernel is unavailable and the engines fall back to the torch indexer for decode, and the engines need data-parallel attention, which is on the H200 branch of the SGLang fork rather than the mainline fork.

`--ep-size` defaults to `actor_nodes * gpus_per_node / pp_size`; `--cp-size > 1` adds `--allgather-cp`; expert TP is always 1.

### 4.2 Algorithm

```bash
--advantage-estimator grpo
--eps-clip 0.2
--eps-clip-high 0.28
--kl-loss-coef 0.00
--kl-loss-type low_var_kl
--entropy-coef 0.00
--rollout-temperature 0.8
--use-rollout-routing-replay   # R3, on by default (--no-enable-r3 to drop)
--moe-router-freeze-gate --freeze-e-score-correction-bias   # required
```

`--mode debug_minimal` (the default) drops over-sampling and the dynamic-sampling filter; `--mode normal` adds `--over-sampling-batch-size 512` with `check_reward_nonzero_std`. `--task dapo_aime` uses dapo-math-17k prompts with the thinking chat template and 4096-token responses by default; the validated runs cap responses at 2048.

The indexer top-k is **not** replayed into the trainer by default (`--enable-indexer-replay` exists; it pins ~300 GB of host memory per engine rank for the replay capture and measured no parity gain, so the residual 0.02 nats gap is kernel numerics).

### 4.3 Rollout & SGLang

```bash
--rollout-num-gpus-per-engine 8   # 2-node engines on GB300; = GPUs per node otherwise
--sglang-tp-size 8 --sglang-dp-size 1 --sglang-ep-size 8
--sglang-attention-backend dsv4
--sglang-moe-runner-backend auto
--sglang-cuda-graph-max-bs-decode 128   # with --sglang-cuda-graph; else --sglang-disable-cuda-graph
--sglang-max-running-requests 128
--sglang-mem-fraction-static 0.7
--sglang-disable-radix-cache            # unless --sglang-radix-cache
```

Environment set by the launcher: `SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1` (with `--sglang-engram-host-table`), `SGLANG_SKIP_CHECKPOINT_LOAD_CHECK=1`, `NCCL_CUMEM_ENABLE=1`, `SGLANG_DSV4_FP4_EXPERTS=0`, `SGLANG_HEALTH_CHECK_TIMEOUT=900`, `SGLANG_DG_CACHE_DIR_PER_PROCESS=1`, `SGLANG_OPT_FP8_WO_A_GEMM=0`, `SGLANG_OPT_FUSE_WQA_WKV=0`, `SGLANG_DISABLE_MULTIMEM_AG=1`, `TORCHINDUCTOR_COMPILE_THREADS=1`, `CUDA_DEVICE_MAX_CONNECTIONS=1`. With `--train-deterministic` (default) the trainer also gets `--deterministic-mode`, `NCCL_ALGO=Ring`, `NVTE_ALLOW_NONDETERMINISTIC_ALGO=0`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`.

**Engram host table.** In a colocated run the engines release all GPU memory during `sleep`; parameters come back through the weight update, but the engram tables are constants that the update does not carry. `--sglang-engram-host-table` keeps one copy of each table in pinned host memory shared by the tensor-parallel ranks of an engine, so a wake-up restores them. On hosts where the shared layout is not auto-selected (H200), set `SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT=shared` as well. The weight-update equality check skips `engram_hasher.` and `engram.embed.` for the same reason (`--check-weight-update` re-enables the check; the validated runs pass `--no-check-weight-update`).

**SGLang version.** The rollout side needs the DeepSeek-V4.1 SGLang fork, which `radixark/miles:deepseek-v41` carries at `/sgl-workspace/sglang/python`; the fork validates the V4.1 feature set from `server_args` at startup.

### 4.4 Optimizer and memory

```bash
--optimizer adam
--lr 1e-6 --lr-decay-style constant
--weight-decay 0.1
--adam-beta1 0.9 --adam-beta2 0.98
--accumulate-allreduce-grads-in-fp32   # --grad-reduce-bf16 switches to --grad-reduce-in-bf16
--attention-softmax-in-fp32
--micro-batch-size 1 --max-tokens-per-gpu 2048
--train-memory-margin-bytes 3221225472
```

Full-parameter training of the full model on 16 GPUs only fits because the optimizer state leaves the GPU. `--disk-offload` adds `--stream-optimizer-state-to-disk --offload-train-target disk --offload-train-disk-dir <dir>`; `--stream-optimizer-state-moment-dtype bf16` halves the streamed moments. Point `--offload-disk-dir` at node-local NVMe with ~1 TB free per node; the trainer's weight backups go to `MILES_WEIGHT_BACKUP_DIR` if set. `--colocate-memory-peak-device cpu` places the colocation peak on the host; on GB300 devboxes the container memory limit (about 626 GB) is below the GPU memory of the node, so keep host residency under ~560 GB per node. `--optimizer-offload` (CPU Adam) is the alternative when host memory allows.

If the run OOMs: lower `--max-tokens-per-gpu`, then add `--grad-reduce-bf16`; `--recompute full` with `--pp-size 2` is already part of the recommended launch.

## 5. Results

DAPO on dapo-math-17k, 16 GB300 GPUs, 2K response cap, 16 prompts x 8 samples per step, first layout of [§4.1](#41-megatron-parallelism). The reward rises from 0.51 to 0.78 (5-step means) over 80 steps while the trainer-vs-rollout policy gap stays flat: per-token KL 0.0012-0.0017 and mean |delta log p| 0.017-0.025 nats. The run completed 120+ steps without a failure; the recommended layout reproduces steps 0-1 within noise (abs_diff 0.0246 / 0.0220, reward 0.52 / 0.59) and reached 0.77 by step 47.

![Raw reward and trainer-vs-rollout policy mismatch over 80 DAPO steps](/assets/images/dsv41-rl-training.png)

| Metric | 80-step run |
|---|---|
| `rollout/raw_reward` | 0.51 -> 0.78 (5-step means) |
| `train/train_rollout_logprob_abs_diff` | 0.017-0.025 nats, no drift |
| `train/train_rollout_kl` | 0.0012-0.0017 |
| `train/grad_norm` at step 0 | 0.13-0.18 |

## 6. Pairs Well With

- [DeepSeek-V4 Flash](/models/deepseek/deepseek-v4-flash) - the parent architecture and the `torch_dist` conversion flow.
- [Architecture Support](/advanced/architecture-support) - the plugin lives under `miles_plugins/models/deepseek_v41/` (`deepseek_v41.py`, `engram.py`, `ops/{compressor,indexer,kvnorm,quant,rope}.py`).
- [Low Precision RL](/advanced/low-precision) - the fake-quantization points the plugin reproduces are the same ones an fp8/fp4 rollout would expose.
