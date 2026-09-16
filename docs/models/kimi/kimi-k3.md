---
title: Kimi-K3
description: LoRA RL recipe for Kimi-K3, a KDA + MLA hybrid with an 896-expert latent MoE, trained colocated with SGLang.
---

The complete Kimi-K3 LoRA RL implementation is open at the Miles pull request:
[`radixark/miles#1825`](https://github.com/radixark/miles/pull/1825). The scripts and image
below come from that branch. Results and background are in the
[LMSYS day-0 write-up](https://www.lmsys.org/blog/2026-07-27-kimi-k3-day0-support).

## 1. Model Introduction

Kimi-K3 pairs two attention mechanisms in one stack, **KDA and MLA chosen per layer**,
with an **896-expert latent MoE** at top-16. The checkpoint ships in **MXFP4**.

miles trains it with **native LoRA adapters** rather than full fine-tuning, which is what
makes the recipe fit at all: the base weights stay frozen and only the adapters carry
gradients. The adapters are implemented under TP, EP, PP and CP, with **shared-A and
per-expert-B factors** across the 896 experts, and they are exported to the rollout engines
as HF-named chunks over CUDA IPC.

**Key highlights:**

- **Two attention types per layer**: KDA and MLA, with an attention-residual snapshot bank.
- **896-expert latent MoE**, top-16, `moe_latent_size=3584`, plus a shared expert.
- **LoRA RL, not full fine-tuning.** Rank 16 by default, 32 in the validated run.
- **Colocated rollout**: trainer and SGLang share the GPUs, with adapters synced over CUDA IPC.
- **MXFP4 checkpoint** upcast to BF16 once, offline.

## 2. Supported Variants

| Variant | Layers | Purpose | GPUs |
|---|---|---|---|
| `full` | full stack | the real model | 64 (16 × 4), validated |
| `4layer` | 4 | smoke test, default | single node |

`--model-variant` selects between them and sets the matching checkpoint paths and
`megatron_model_type`.

Architecture, from `scripts/models/kimi-k3.sh`: hidden 7168, FFN 33792, 96 attention heads,
`kv_channels=256`, MLA with `q_lora_rank=1536` / `kv_lora_rank=512` /
`qk_head_dim=128` / `qk_pos_emb_head_dim=64` / `v_head_dim=128`, 896 experts at
`moe_ffn_hidden_size=3072`, shared expert 6144, vocab 163840, no position embedding.

## 3. Environment Setup

Use the `docker.io/radixark/miles:kimi-k3` image, which pins miles, SGLang (the
[`sglang-miles-k3`](https://github.com/sgl-project/sglang/tree/sglang-miles-k3) branch) and
flashinfer `0.6.15.post1` at the validated versions. On Hopper set
`SGLANG_K3_ATTN_RES_MODE=jit`.

The only external asset is the Kimi-K3 MXFP4 HF checkpoint. Everything else derives in-repo.

### 3.1 Data

```bash
python scripts/run_kimi_k3_lora.py prepare-data --task dapo-math --data-dir <datasets>
```

### 3.2 MXFP4 to BF16

```bash
python tools/convert_mxfp4_to_bf16.py --model-dir <native-mxfp4> --save-dir <bf16-hf>
```

### 3.3 BF16 to `torch_dist`

Unlike the bridge-mode recipes, K3 needs an offline conversion. Run it on 32 ranks; the
output re-shards at load, so the conversion layout does not have to match the training one:

```bash
source scripts/models/kimi-k3.sh   # defines MODEL_ARGS
torchrun --nnodes=8 --nproc-per-node=4 ... \
    tools/convert_hf_to_torch_dist.py "${MODEL_ARGS[@]}" \
    --hf-checkpoint <bf16-hf> --save <torch-dist-dcp> \
    --bf16 --tensor-model-parallel-size 32 --sequence-parallel \
    --pipeline-model-parallel-size 1 --context-parallel-size 1 \
    --expert-model-parallel-size 32 --expert-tensor-parallel-size 1 \
    --megatron-to-hf-mode raw
```

Training then takes the **MXFP4** directory as `--hf-checkpoint` and the converted
`torch_dist` as `--ref-load`.

## 4. Launch

Validated on **16 nodes × 4 GPUs**. One container per node; bring up a ray cluster across
them, `export MILES_SCRIPT_EXTERNAL_RAY=1`, then:

```bash
python scripts/run_kimi_k3_lora.py train \
  --mode normal --model-variant full --task dapo-math --reward-model deepscaler \
  --num-nodes 16 --num-gpus-per-node 4 \
  --pipeline-parallel-size 8 --context-parallel-size 2 \
  --rollout-tp-size 16 --rollout-max-concurrency 8 \
  --lora-rank 32 --lora-alpha 64 \
  --num-rollout 1000 --rollout-batch-size 8 --n-samples-per-prompt 8 \
  --rollout-max-response-len 4096 --sglang-max-total-tokens 65536 \
  --global-batch-size 64 --lr 1e-05 --eval-interval 10 \
  --distributed-timeout-minutes 60 \
  --hf-checkpoint <native-mxfp4> --ref-load <torch-dist-dcp> \
  --sglang-path /sgl-workspace/sglang/python \
  --data-dir <datasets> --enable-wandb
```

`--rollout-max-concurrency 8` is passed explicitly: the field default is 64, and the
validated runs pin 8.

For a single-node smoke test, drop to the default `--model-variant 4layer`.

## 5. Recipe Configuration

### 5.1 Parallelism

The resolved config at startup should show `expert_model_parallel_size 8`,
`max_tokens_per_gpu 8192`, `colocate_memory_peak_device gpu` and
`lora_base_cpu_backup True`. Checking those four lines is the fastest way to confirm the
run came up in the intended shape.

| Knob | Validated value |
|---|---|
| Pipeline parallel | 8 |
| Context parallel | 2 |
| Expert parallel | 8 (resolved) |
| Rollout TP | 16 |
| `max_tokens_per_gpu` | 8192 |

### 5.2 LoRA

Rank 32 / alpha 64 in the validated run; the script defaults to 16 / 32. Adapters attach to
attention output and both MLA down-projections, the dense MLP, and both expert projections:

```
self_attention.o_proj
self_attention.q_a_proj
self_attention.kv_a_proj_with_mqa
mlp.linear_fc1, mlp.linear_fc2
mlp.experts.linear_fc1, mlp.experts.linear_fc2
```

The 896 experts share one A factor and carry per-expert B factors, which is what keeps the
adapter count tractable at this expert width.

### 5.3 Rollout

Rollout is **colocated**: the trainer and SGLang share GPUs, and adapters sync over CUDA
IPC as HF-named chunks. `lora_base_cpu_backup` keeps a host copy of the frozen base so the
GPU copy can be reclaimed during rollout.

### 5.4 What a healthy run looks like

From the GB300 validation runs:

- 11 to 13 minutes per rollout cycle
- trainer allocated memory returns to about 91 GB after every weight sync
- `rollout/raw_reward` between 0.5 and 0.75 from rollout 0
- `eval/aime` 0.37 to 0.43 at eval@0 (that spread is temperature-0 nondeterminism), rising
  by at least 0.06 by eval@9; the measured run went 0.367 to 0.467

The memory figure is the one to watch. If allocated does not return to its baseline after a
weight sync, the adapter export is leaking and the run will die later rather than sooner.

### 5.5 Notable quirks

- The image preloads a small `shm_unlink` shim through `/etc/ld.so.preload`. It tolerates a
  benign PyTorch CUDA-IPC unlink race that otherwise aborts colocated weight sync at scale.
- On Hopper, set `SGLANG_K3_ATTN_RES_MODE=jit`.
- Weight conversion for K3 lives in
  `miles/backends/megatron_utils/megatron_to_hf/kimi_k3.py`, and the model itself in
  `miles_plugins/models/kimi_k3/`.

## 6. Pairs Well With

- [LoRA](/advanced/lora)
- [Backends Beyond Megatron](/advanced/architecture-support)
- [Kimi-K2.5](/models/kimi/kimi-k2.5)
