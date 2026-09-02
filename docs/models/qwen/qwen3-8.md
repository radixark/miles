---
title: Qwen3.8
description: Launch recipe for the dense Qwen3.8-27B, plus where the 2.4T-A95B MoE recipe lives.
---
## 1. Model Introduction

[Qwen3.8](https://github.com/QwenLM/Qwen3) continues Alibaba's Qwen3 line. The
dense **Qwen3.8-27B** ships the same `config.json` as
[Qwen3.5-27B](/models/qwen/qwen3-5) and [Qwen3.6-27B](/models/qwen/qwen3-6) —
same hybrid GDN backbone, same gated attention, same tokenizer and vocabulary.
It therefore reuses the Qwen3.5 implementation
(`miles_plugins.models.qwen3_5`, the family's default `--model-impl miles`), and
`scripts/models/qwen3.8-27B.py` is a one-line derivation of the Qwen3.5-27B
model args; the three expand to byte-identical Megatron flags.

The sparse **Qwen3.8-2.4T-A95B** is a different recipe entirely — see
[section 6](#6-qwen38-24t-a95b).

**Key highlights (27 B):**

- **Dense GDN backbone**: 27 B parameters, hybrid linear / full attention (`full_attention_interval 4`).
- **Attention-output gate**: shared with Qwen3.5, trained alongside attention weights.
- **Extended rotary base**: `--rotary-base 10000000`, `--rotary-percent 0.25`.
- **Larger vocabulary**: 248320 tokens.
- **Shape**: `hidden-size 5120`, `ffn-hidden-size 17408`, 64 layers.
- **Multimodal**: `Qwen3_5ForConditionalGeneration`; the RL recipe below trains the text path only.

## 2. Supported Variants

| Model | Class | HF ID | Recipe |
|---|---|---|---|
| Qwen3.8-27B | Dense (GDN) | [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) | this page |
| Qwen3.8-2.4T-A95B | MoE | — | [section 6](#6-qwen38-24t-a95b) |

Sections 3–6 cover the dense 27 B.

## 3. Environment Setup

### 3.1 Download model + datasets

```bash
hf download Qwen/Qwen3.8-27B --local-dir /root/models/Qwen3.8-27B
hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir /root/datasets/dapo-math-17k
hf download --repo-type dataset zhuzilin/aime-2024     --local-dir /root/datasets/aime-2024
```

### 3.2 HF → Megatron `torch_dist` conversion

Run it on all eight GPUs; the tool shards the 64 layers over the ranks
(`--pipeline-model-parallel-size` is derived from `WORLD_SIZE`) and the output
re-shards at load, so the conversion layout does not have to match the training one:

```bash
cd /root/miles
MODEL_ARGS_LINE="$(python3 miles/utils/external_utils/model_args_utils.py qwen3.8-27B)" || exit 1
read -ra MODEL_ARGS <<< "${MODEL_ARGS_LINE}"
PYTHONPATH=/root/Megatron-LM torchrun --nproc-per-node 8 \
   tools/convert_hf_to_torch_dist.py \
   "${MODEL_ARGS[@]}" \
   --hf-checkpoint /root/models/Qwen3.8-27B \
   --save          /root/models/Qwen3.8-27B_torch_dist
```

## 4. Launch

### 4.1 Quick start

```bash
cd /root/miles
python scripts/run_qwen3_dense.py --model-name Qwen3.8-27B
```

`scripts/run_qwen3_dense.py` is the shared dense launcher; `--model-name Qwen3.8-27B`
selects this recipe, which targets 1 node × 8 GPU. Checkpoints come from `--model-dir`
(default `/root/models`) and datasets from `--data-dir` (default `/root/datasets`);
checkpoints are written under `--output-dir` (default `/root/shared_data`).

### 4.2 What one step costs

Measured on 1 × 8 H200 with the recipe below, `rollout-batch-size 32`,
`n-samples-per-prompt 8`, `rollout-max-response-len 8192`:

| Phase | Time |
|---|---|
| Rollout (256 sequences) | 307 s |
| Train (ref log-probs 100 s + actor log-probs 33 s + backward/optimizer) | 415 s |
| Weight sync back to SGLang | 2.5 s |
| **Per step** | **≈ 12 min** |

`perf/tokens_per_gpu_per_sec` 478. Mean response length 4585 tokens, 33 % truncated at the
8192 cap. Step 0 reported `rollout/raw_reward` 0.64 and
`train/train_rollout_logprob_abs_diff` 0.010, i.e. the SGLang and Megatron forward passes
agree closely — see [True On-Policy](/examples/infra-features/true-on-policy) for what that
metric does and does not tell you.

## 5. Recipe Configuration

### 5.1 Parallelism

| TP | PP | CP | EP | `max_tokens_per_gpu` | SGLang `mem-fraction-static` | CPU Adam | GPUs |
|---|---|---|---|---|---|---|---|
| 4 | 1 | 1 | 1 | 8192 | 0.8 | ✓ | 8 (1 × 8) |

`--sequence-parallel` is enabled. Activation checkpointing is on
(`--recompute-granularity full --recompute-method uniform --recompute-num-layers 1`).

### 5.2 Algorithm

GRPO with low-variance KL:

```bash
--advantage-estimator grpo
--use-kl-loss
--kl-loss-coef 0.00
--kl-loss-type low_var_kl
--entropy-coef 0.00
--eps-clip 0.2
--eps-clip-high 0.28
```

### 5.3 Rollout & SGLang

```bash
--rollout-num-gpus-per-engine 1
--sglang-mem-fraction-static 0.8
```

One engine per GPU, inherited from the Qwen3.5 line.

**Why 0.8 and not the 0.5 the rest of the line uses.** The BF16 weights are ~53 GB per
engine. At 0.5 SGLang targets 70 GB of the H200's 140 GB, so after the weights only ~17 GB
is left for the pools — and it reports 69 GB still free. That matters more here than on a
plain transformer: 48 of the 64 layers are linear attention, and their recurrent state is
resident per sequence (48 v-heads × 128 × 128 × FP32 × 48 layers ≈ **151 MB per in-flight
sequence**), an order of magnitude above the KV cost. Measured on one node:

| | `mem-fraction-static 0.5` | `mem-fraction-static 0.8` |
|---|---|---|
| KV pool | 158,018 tok | 517,999 tok |
| GDN state slots (`mamba_cache_size`) | 60 | 198 |
| `#running-req` per engine | 11–12 | 39 |
| decode throughput per engine | 553–690 tok/s | 1855 tok/s |

Decode re-reads all 53 GB of weights every step regardless of batch size, so the small
batch was pure bandwidth waste. Under colocate the training model is offloaded to host RAM
during rollout, and the training-phase peak at 0.8 measured 88–90 GB of 140 GB, so the
higher fraction costs nothing. Two further levers if rollout is still the bottleneck:
`--rollout-num-gpus-per-engine 2` (halves per-GPU weights *and* shards the GDN state), and
`--sglang-mamba-ssm-dtype bfloat16` (halves the state, at a precision cost worth checking
against `train/train_rollout_logprob_abs_diff`).

### 5.4 Optimizer

CPU Adam is enabled (`--optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d --use-precision-aware-optimizer`).

### 5.5 Notable quirks

From `scripts/models/qwen3.8-27B.py`, which defers to `scripts/models/qwen3.5-27B.py`:

- `--model-impl` defaults to `miles` — Qwen3.8 reuses the Qwen3.5 implementation (gated attention, FP32 `A_log`).
- `--rotary-base 10000000`, `--rotary-percent 0.25`.
- `--vocab-size 248320`.
- `--apply-layernorm-1p`, `--qk-layernorm`, `--group-query-attention`.
- `--attention-output-gate`.

See [Disk Offload](/advanced/disk-offload) for how miles keeps FP32-marked parameters like
the GDN `A_log` out of the low-precision optimizer path.

## 6. Qwen3.8-2.4T-A95B

The sparse 2.4 T / 95 B-active flagship shares nothing operational with the dense 27 B: the
rollout engines serve the native ModelOpt **NVFP4** experts-only checkpoint
([RadixArk/Qwen3.8-2.4T-A95B-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-2.4T-A95B-NVFP4))
while the Megatron trainer runs **BF16** off a `torch_dist` built from the BF16 parent, and
weight sync ships **LoRA adapters only** — the quantized base never leaves the engines. The
implementation is [radixark/miles#2488](https://github.com/radixark/miles/pull/2488)
("Qwen 3.8 day-0 lora RL support"): `scripts/run_qwen3_8.py` plus the
`scripts/models/qwen3.8-2.4T-A95B{,_4layer,_full}.py` definitions. It is **LoRA-only in
practice** — the native raw-mode `--lora` path is the validated route; treat full-weight
training as unvalidated.

### 6.1 Variants

`--model-variant` selects between them and sets the matching checkpoint paths and
`megatron_model_type` (`scripts/models/qwen3.8-2.4T-A95B{_4layer,_full}.py`):

| Variant | Layers | Purpose | GPUs |
|---|---|---|---|
| `full` | 92 | the real model | 64 (16 × 4 GB300), validated |
| `4layer` | 4 | smoke test, default | single node |

### 6.2 Environment Setup

Use the `docker.io/radixark/miles:dev` image — it is multi-arch, so the same tag serves
GB300 (aarch64) and x86 nodes. Inside the container, multi-node rollout engines need
**NCCL ≥ 2.30.7** (2.28.x deadlocks CUDA-graph-captured cross-node collectives):

```bash
pip install nvidia-nccl-cu13==2.30.7
export SGLANG_NCCL_SO_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib/libnccl.so.2
export LD_PRELOAD=$SGLANG_NCCL_SO_PATH
```

The launcher looks checkpoints up under `--model-dir` (default `/root/models`) as
`Qwen3.8-2.4T-A95B-NVFP4_<variant>` (rollout) and
`Qwen3.8-2.4T-A95B-bf16_<variant>` (trainer source):

```bash
hf download RadixArk/Qwen3.8-2.4T-A95B-NVFP4 \
    --local-dir /root/models/Qwen3.8-2.4T-A95B-NVFP4_full
# BF16 parent -> /root/models/Qwen3.8-2.4T-A95B-bf16_full
```

Unless `--skip-prepare` is passed, the launcher's prepare step downloads
`zhuzilin/dapo-math-17k` and runs the BF16 → `torch_dist` conversion itself
(`tools/convert_hf_to_torch_dist.py` via the `qwen3.8-2.4T-A95B_<variant>` model args); the
output re-shards at load, so the conversion layout does not have to match the training one.
Training then takes the **NVFP4** directory as `--hf-checkpoint` and the converted
`torch_dist` as `--ref-load` — both wired up by the launcher.

### 6.3 Launch

Single-node smoke test (default `--model-variant 4layer`, 8 GPUs):

```bash
python scripts/run_qwen3_8.py
```

Full model, validated on **16 nodes × 4 GB300** (bring up a ray cluster across the nodes,
`export MILES_SCRIPT_EXTERNAL_RAY=1`, then run on the head):

```bash
python scripts/run_qwen3_8.py \
  --mode normal --model-variant full --hardware GB300 \
  --num-nodes 16 --num-gpus-per-node 4 \
  --tp 1 --pp 4 --ep 16 --etp 1 --cp 1 \
  --lora --lora-rank 32 --lora-alpha 64 \
  --num-rollout 200 --rollout-batch-size 32 --n-samples-per-prompt 8 \
  --rollout-max-response-len 4096 --global-batch-size 256 \
  --rollout-max-concurrency 8 --recompute --skip-prepare \
  --extra-args "--rollout-num-gpus-per-engine 16 --lr 1e-5"
```

| Side | Parallelism | Notes |
|---|---|---|
| Trainer (BF16) | TP 1 / PP 4 / EP 16 / ETP 1 | `num_query_groups 4` caps native-LoRA TP at 4 |
| Rollout (NVFP4) | 4 engines × TP 16 / EP 1 | `flashinfer_trtllm` MoE runner, triton attention |
| LoRA | rank 32 / alpha 64, `q,k,v,o_proj` | attention projections only; the 69 GDN mixer layers and the routed experts carry no adapter, so the rollout MoE path is untouched |

### 6.4 Memory: the alternating-backup budget

The recipe fits 956 GB hosts only if, at any moment, **exactly one side keeps a host
backup**: during rollout the sleeping trainer's tms backup (~390 GB/node); during train the
sleeping engines' NVFP4 mirror (~372 GB/node, `--lora-base-cpu-backup` →
SGLang `enable_weights_cpu_backup`, which is also what lets every weight update skip the
base re-ship). Anything that double-books breaks it. The flags that hold the line:

- `--drop-checkpoint-page-cache-after-load` — a multi-TB DCP read leaves its page cache on
  every node; lazy reclaim loses the race against pinned-allocation spikes at the
  rollout/train handoff.
- `--colocate-memory-peak-device gpu` and no `--use-kl-loss` at `kl-loss-coef 0` — a zero
  coefficient still loads a full ref checkpoint and pins a second per-rank base copy.
- Weight sync must not clone the sleeping side's backup (`get_cpu_backup(zero_copy=True)`
  in #2488); the clone variant retains ~75 GB/rank in malloc arenas per update.

Healthy peaks sit at **600–650 GB per node**; sustained readings above ~750 GB mean a
second backup or the page cache came back.

## 7. Pairs Well With

- [Qwen3.5](/models/qwen/qwen3-5) — same architecture at 4 B / 9 B / 27 B
- [True On-Policy](/examples/infra-features/true-on-policy)
- [Low Precision RL](/advanced/low-precision)
