---
title: Nemotron-3-Ultra
description: Launch recipe for NVIDIA Nemotron-3-Ultra-550B-A55B (hybrid Mamba2 + Attention + latent-MoE) via Megatron AutoBridge.
---
## 1. Model Introduction

[NVIDIA Nemotron-3-Ultra-550B-A55B](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16)
is the Ultra tier of the `nemotron_h` family: the same hybrid block pattern as
Nano and Super, scaled to **550 B total / 55 B active** across 108 layers, with
a **latent MoE** (512 experts, top-22, `moe_latent_size=2048`) and one shared
expert.

miles loads it through the `megatron.bridge` AutoBridge with the shared
NemotronH MoE shim (`miles_plugins/megatron_bridge/nemotron_h.py`), the same
path Super-120B uses. There is no offline `torch_dist` conversion.

**Key highlights:**

- **Hybrid + latent MoE**: Mamba2 and attention blocks with a latent-projection
  MoE FFN, 512 experts at top-22.
- **Bridge-mode load**: `--megatron-to-hf-mode bridge`, straight from the HF
  checkpoint.
- **Sigmoid routing** with aux-free expert-bias load balancing, plus an MTP head
  in the checkpoint.
- **A single-node 4-layer slice** is published alongside the full model, so the
  recipe can be smoke-tested without 16 nodes.

## 2. Supported Variants

| Model | Active / Total | Layers | GPUs | HF ID |
|---|---|---|---|---|
| Nemotron-3-Ultra-550B-A55B-BF16 | 55 B / 550 B | 108 | 128 (16 × 8) | [nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16) |
| Nemotron-3-Ultra-550B-A55B-BF16-4layer | slice | 4 | 8 (1 × 8) | pruned slice of the above, for smoke tests |

Tested on H200. Use the `radixark/miles:dev` image.

## 3. Environment Setup

### 3.1 Download model + datasets

```bash
hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir /root/datasets/dapo-math-17k
hf download nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16 \
   --local-dir /root/models/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16
```

`--model-dir` and `--data-dir` default to `/root/models` and `/root/datasets`.
`--model-name` names the checkpoint directory inside `--model-dir`; passing a
name matching `<N>layer` selects the pruned slice and switches the recipe to its
single-node parallelism automatically.

### 3.2 No `torch_dist` conversion

AutoBridge plus the NemotronH shim read the HF checkpoint directly, so
`--hf-checkpoint` and `--ref-load` both point at the download:

```bash
--hf-checkpoint <model-dir>/<model-name>
--ref-load      <model-dir>/<model-name>
--megatron-to-hf-mode bridge
```

## 4. Launch

### 4.1 Single-node smoke test

```bash
cd /root/miles
python scripts/run_nemotron_3_ultra_550b_a55b.py full-train \
    --model-name NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16-4layer --num-nodes 1
```

### 4.2 Full model

The full 108-layer model needs **16 nodes × 8 GPU**. Bring up the ray cluster
yourself, tell the launcher it is external, and submit from the head — the
launcher has no worker-side subcommand:

```bash
# on the head pod
ray start --head --num-gpus 8 --disable-usage-stats
# on every worker pod
ray start --address=${HEAD_IP}:6379 --num-gpus 8 --disable-usage-stats

# then, on the head pod
export MILES_SCRIPT_EXTERNAL_RAY=1
export RAY_ADDRESS=http://${HEAD_IP}:8265
python scripts/run_nemotron_3_ultra_550b_a55b.py train --num-nodes 16
```

Without `MILES_SCRIPT_EXTERNAL_RAY=1` the launcher runs `ray stop --force` and
starts a fresh single-node head, tearing down the cluster the workers joined.

The recipe defaults to a 30-rollout run (`--num-rollout`), rollout batch 32 at 8
samples per prompt, global batch 128.

## 5. Recipe Configuration

### 5.1 Parallelism

| Variant | TP | PP | CP | EP | ETP | GPUs |
|---|---|---|---|---|---|---|
| Full 108-layer | 8 | 4 | 1 | 32 | 1 | 128 (16 × 8) |
| 4-layer slice | 1 | 1 | 1 | 8 | 1 | 8 (1 × 8) |

**Mamba `n_groups=8` caps attention and Mamba tensor parallelism at 8**, because
Megatron requires `n_groups % tp == 0`. That constraint drives the whole layout:
TP cannot grow past 8, so scale comes from PP and EP instead.

The 4-layer slice fits on one node, so it gives every rank to expert parallelism
(512 experts over EP=8 is 64 per rank) and keeps attention and Mamba at TP=1.

`--sequence-parallel` is on when TP > 1. Activation checkpointing is enabled, and
`--log-probs-chunk-size 128` keeps the log-prob pass inside the memory budget.

### 5.2 Algorithm

GRPO with low-variance KL:

```bash
--advantage-estimator grpo
--kl-loss-coef 0.00
--kl-loss-type low_var_kl
--entropy-coef 0.00
--eps-clip 0.2
--eps-clip-high 0.28
--rm-type deepscaler
```

### 5.3 Rollout & SGLang

| Variant | GPUs per engine | `--sglang-ep-size` | `--sglang-dp-size` | mem fraction |
|---|---|---|---|---|
| Full 108-layer | 32 | 32 | 4 | 0.7 |
| 4-layer slice | 8 | 8 | 2 | 0.6 |

The 550 B model is roughly 1.1 TB in BF16 and **does not fit one 8-GPU engine**,
so rollout runs 32-GPU engines with EP=32 and DP-attention. The DP size is chosen
so that `attn_tp = gpus_per_engine / dp_size` lands on 8, satisfying the same
Mamba `n_groups` constraint the training side has. The launcher asserts both
divisibility rules rather than letting a bad combination fail deep in SGLang.

### 5.4 Optimizer

CPU Adam, with the host transfer overlapped:

```bash
--optimizer-cpu-offload
--overlap-cpu-optimizer-d2h-h2d
--use-precision-aware-optimizer
--lr 1e-6
```

### 5.5 Notable quirks

- **Routing replay is not enabled yet.** `--use-rollout-routing-replay` is off
  for the 108-layer model: the routing capturer needs a fix for per-layer top-22
  under DP-attention. Train and rollout log-probs differ by about 0.01 without
  it. The Super-120B recipe does enable it.
- **No `--spec`**: AutoBridge and the NemotronH shim synthesize the Megatron MoE
  spec from the HF config.
- The shim is what wires `routed_scaling_factor`, `n_group` and `topk_group`
  onto the Megatron provider. Without it the routed output is silently scaled
  1.0×, the same drift class the Nano-MoE and Super recipes call out.
- The checkpoint carries an MTP head (`num_nextn_predict_layers=1`); the RL
  recipe does not train it.

## 6. Pairs Well With

- [Backends Beyond Megatron](/advanced/architecture-support)
- [Nemotron-3-Super](/models/nemotron/nemotron-3-super)
- [P2P Weight Transfer](/advanced/p2p-weight-transfer)
