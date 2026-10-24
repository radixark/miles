---
title: Step-5-Preview
sidebarTitle: Step-5-Preview
description: Launch recipe for StepFun's Step-5-Preview (600B MoE, sliding-window + DSA sparse attention) — FP8 rollout, BF16 training, on two 8-GPU B300 nodes.
---
## 1. Model Introduction

[Step-5-Preview](https://huggingface.co/stepfun-ai/Step-5-Preview) is StepFun's 600B-parameter MoE model (27B active per token). miles trains its text decoder.

**Key highlights:**

- **Hybrid attention**: 69 sliding-window layers (window 512) and 23 full-attention layers with DSA sparse attention: an indexer scores 8-token blocks, and each query attends up to 512 selected blocks plus its current one. The two types use different RoPE (base 1e4 on all dims vs 1e7 on a third).
- **GQA with a gate**: 64 query heads over 4 KV groups, head dim 192, and a head-wise sigmoid gate on the attention output.
- **MoE**: 352 routed experts, top-8 sigmoid routing scaled by 3.0, plus a shared expert; layers 0–2 and 91 are dense.
- **FP8 checkpoint**: routed experts in FP8 E4M3 with 128×128 block scales, except the last three MoE layers (88–90), which ship in BF16. The rollout serves the checkpoint as released; the trainer dequantizes the FP8 experts on load and trains in BF16.

## 2. Supported Variants

| `--model-name` | What it is | HF source |
|---|---|---|
| `Step5` | text decoder, 92 layers | [stepfun-ai/Step-5-Preview](https://huggingface.co/stepfun-ai/Step-5-Preview) |

The vision tower and MTP layers stay in the checkpoint but are not trained (text-only RL).

## 3. Environment Setup

### 3.1 Download

The launcher reads the checkpoint at `<model-dir>/<model-name>`; `prepare` fetches dapo-math-17k:

```bash
hf download stepfun-ai/Step-5-Preview --local-dir /root/models/Step5
python scripts/run_step5.py prepare --model-name Step5 --num-nodes 2 --num-gpus-per-node 8
```

The full model needs no Megatron conversion: the trainer reads the FP8 checkpoint through the Step5 bridge at startup and dequantizes the experts on read, since a `torch_dist` copy (1.2 TB) would not fit node disk.

The AIME 2026 evals read `<data-dir>/aime-2026/aime-2026.jsonl` (`prompt` and `label` rows), which `prepare` does not fetch; pass `--no-eval-aime` without it.

### 3.2 SGLang

The engine needs the Step-5-Preview support in [sgl-project/sglang#44356](https://github.com/sgl-project/sglang/pull/44356).

The `radixark/miles:step5` image (multi-arch, built on `lmsysorg/sglang:v0.5.20`) carries it and the recipe code:

```bash
docker pull radixark/miles:step5
```

## 4. Launch

### 4.1 Two nodes

Start a ray cluster over both nodes, then launch from the head. `--ray-hostfile` joins the other node over ssh; to start the cluster yourself, set `MILES_SCRIPT_EXTERNAL_RAY=1` instead:

```bash
MASTER_ADDR=<head ip> python scripts/run_step5.py train --model-name Step5 \
    --num-nodes 2 --num-gpus-per-node 8 --ray-hostfile /root/hostfile --num-rollout 20
```

On two 8×B300 nodes, with 32 prompts × 8 samples per step, responses capped at 2048 tokens (the recipe default is 32768), DAPO dynamic sampling on and evals off, a step took a median 274 s over 80 steps: 81 s of training and 95 s of weight update, the rest mostly rollout.

## 5. Recipe Configuration

### 5.1 Parallelism

| `--model-name` | TP | PP | CP | EP | expert-TP | GPUs | rollout engine |
|---|---|---|---|---|---|---|---|
| `Step5` | 4 | 1 | 1 | 16 | 1 | 16 (2 × 8) | TP8/EP8 per node, `--sglang-mem-fraction-static 0.75` |

TP is capped at 4 by the four KV groups. Context parallelism is not supported: each layer builds its own RoPE, which would re-slice THD positions per CP rank. `--sequence-parallel` is on whenever TP > 1. A micro-batch holds one prompt beside a full-length response (`--max-tokens-per-gpu 34816` on 2 × 8) with THD packing and `--use-dynamic-batch-size`.

### 5.2 Algorithm

GRPO with `--eps-clip 0.2 --eps-clip-high 0.28 --entropy-coef 0.00` and `--rm-type math` on dapo-math-17k, 8 samples per prompt at temperature 1.0 with top-p and top-k off. No KL loss. The rollout runs FP8 experts against a BF16 trainer, so `--use-rollout-routing-replay` replays the rollout's expert choices and `--use-tis` covers the rest.

### 5.3 Optimizer

Adam at `--lr 1e-6`. With EP 16 on 16 GPUs, expert data parallelism is 1, so the distributed optimizer cannot shard the expert Adam state, and it does not fit HBM beside the BF16 weights and gradients. It stays in pinned host memory (224 GB per GPU) at 6 bytes per parameter, BF16 moments plus the int16 low halves of the FP32 masters, and streams through the GPU in 1 GiB chunks:

```bash
--use-precision-aware-optimizer
--main-params-dtype fp32 --exp-avg-dtype bf16 --exp-avg-sq-dtype bf16
--chunked-optimizer-state-offload
--optimizer-state-offload-fraction 1.0
--optimizer-state-offload-chunk-size-mb 1024
--grad-reduce-in-bf16
--train-env-vars '{"PYTORCH_ALLOC_CONF": "pinned_max_cached_size_mb:4"}'
```

`pinned_max_cached_size_mb:4` stops PyTorch from rounding the 12 and 24 MiB expert tensors' pinned blocks up to a power of two, which costs a third more host memory.

### 5.4 Notable quirks

- **`--window-attn-skip-freq` reads backwards**: Megatron treats a 1 as window attention, not a skip. Step5 passes 1 for `layer % 4 != 3`, the 69 sliding-window layers, with `--window-size 511,0`: the query plus 511 previous keys, 512 in total.
- **Frozen DSA indexer**: every parameter but the indexer trains, and the full-attention layers recompute the rollout's region selection with it. The weight sync never sends it; the engine keeps its own copy.
- **`--sglang-max-total-tokens 2097152`** is a correctness bound: an unbounded KV pool overflows the DSA kernels' int32 strides.
- **`--sglang-moe-runner-backend triton`**: routing replay needs the engine's materialized top-k ids.

## 6. Pairs Well With

- [Rollout Routing Replay (R3)](/advanced/miles-router)
- [Low Precision RL](/advanced/low-precision)
