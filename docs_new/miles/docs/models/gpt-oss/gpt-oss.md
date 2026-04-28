---
title: GPT-OSS 20B
description: Two launchers — Megatron BF16 (8 GPU, mbridge) and FSDP (4 GPU, dequantises MXFP4 → BF16 first).
---

# GPT-OSS 20B

Two launchers ship in `scripts/`:

- `scripts/run-gpt-oss-20b-bf16.sh` — Megatron BF16 path on 8 GPU. Uses `--megatron-to-hf-mode bridge` (mbridge) so it loads the HF checkpoint directly — no `tools/convert_hf_to_torch_dist.py` step.
- `scripts/run-gptoss-20b-fsdp.sh` — FSDP path on 4 GPU. The script first runs an inline Python snippet that downloads `openai/gpt-oss-20b` and dequantises its MXFP4 weights to BF16, saving to `/root/models/gpt-oss-20b-bf16`.


## Variant

| HF ID | Local path (BF16 launcher) | Local path (FSDP launcher) |
|---|---|---|
| `openai/gpt-oss-20b` | `$BASE_DIR/gpt-oss-20b` (script hardcodes `BASE_DIR=/root/shared`) | `/root/models/gpt-oss-20b-bf16` (produced by the script's inline `convert_model.py`) |

The HF ID `openai/gpt-oss-20b` is from `run-gptoss-20b-fsdp.sh:18`.

## Quick start (Megatron BF16, 1 node × 8 GPU)

```bash
cd /root/miles
# Stage everything under /root/shared (the script's hardcoded BASE_DIR)
hf download openai/gpt-oss-20b --local-dir /root/shared/gpt-oss-20b
hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir /root/shared/dapo-math-17k

bash scripts/run-gpt-oss-20b-bf16.sh
```

No `convert_hf_to_torch_dist.py` step — the launcher's `CKPT_ARGS` includes `--megatron-to-hf-mode bridge` and points `--hf-checkpoint` directly at the HF directory.

## Quick start (FSDP, 1 node × 4 GPU)

```bash
cd /root/miles
hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir /root/dapo-math-17k

# The script downloads openai/gpt-oss-20b itself, dequantises MXFP4 → BF16,
# and writes /root/models/gpt-oss-20b-bf16 before launching.
bash scripts/run-gptoss-20b-fsdp.sh
```

The script restricts to GPUs 4–7 via `CUDA_VISIBLE_DEVICES=4,5,6,7` and starts Ray with `--num-gpus 4`.

## Paths the launchers expect

`run-gpt-oss-20b-bf16.sh:23-47` (BASE_DIR is hardcoded in the script):

```bash
BASE_DIR=/root/shared

CKPT_ARGS=(
   --hf-checkpoint $BASE_DIR/gpt-oss-20b
   # --hf-checkpoint $BASE_DIR/gpt-oss-20b-BF16
   --megatron-to-hf-mode bridge
   # --save $BASE_DIR/gpt-oss-20b-BF16
   # --save-interval 50
)

ROLLOUT_ARGS=( --prompt-data $BASE_DIR/dapo-math-17k/dapo-math-17k.jsonl
               --rm-type math ... )
```

`run-gptoss-20b-fsdp.sh:53-71`:

```bash
CKPT_ARGS=(
   --hf-checkpoint /root/models/gpt-oss-20b-bf16
)

ROLLOUT_ARGS=( --prompt-data /root/dapo-math-17k/dapo-math-17k.jsonl
               --rm-type deepscaler ... )
```

Note the `--rm-type` differs: `math` for the Megatron BF16 path, `deepscaler` for FSDP. Neither launcher writes `--save`/`--load`/`--save-interval`.

## Parallelism (Megatron BF16)

| TP | PP | CP | EP | expert-TP | `micro-batch-size` | GPUs |
|---|---|---|---|---|---|---|
| 8 | 1 | 1 | 8 | 1 | 1 | 8 (1 × 8) |

`--use-dynamic-batch-size` is **not** used here — the script's comment explains it: `--qkv-format bshd` (required for sink attention with TE) is incompatible with dynamic batch size. So only `--micro-batch-size 1` is set.

`--sequence-parallel` is on (required for TP + EP).

## FSDP shape

| `--rollout-num-gpus-per-engine` | `--sglang-tensor-parallel-size` | dtype | GPUs |
|---|---|---|---|
| 4 | 1 | bfloat16 | 4 |

Plus `--train-backend fsdp --bf16 --attn-implementation eager` passed directly to `train.py`.

## SGLang flags

`run-gpt-oss-20b-bf16.sh`:

```bash
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 4
   --sglang-dtype bfloat16
   --sglang-decode-log-interval 1000
   --sglang-mem-fraction-static 0.70
)
```

`run-gptoss-20b-fsdp.sh`:

```bash
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 4
   --sglang-tensor-parallel-size 1
   --sglang-dtype bfloat16
   --sglang-decode-log-interval 1000
)
```

## Algorithm

Both scripts use GRPO with `--eps-clip 0.2 --eps-clip-high 0.28 --entropy-coef 0.00`. **`--use-kl-loss` is commented out in both scripts** (the BF16 script's comment notes "need gpt oss ckpt conversion" before KL can be enabled).

The Megatron BF16 launcher enables CPU Adam (`--optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d --use-precision-aware-optimizer`); the FSDP launcher does not.

## Megatron BF16 attention quirks

`MISC_ARGS` in the BF16 script:

```bash
--attention-dropout 0.0
--hidden-dropout 0.0
--qkv-format bshd        # required for TE sink attention (SWA + learnable softmax offset)
--attention-backend fused
```

`--qkv-format bshd` is mandated by the sink-attention pattern; in turn it precludes `--use-dynamic-batch-size`. Don't toggle either flag without the other.
