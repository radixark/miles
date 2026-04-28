---
title: GLM4
description: Launch recipes for GLM-Z1-9B-0414. The 32 B model config ships without a launcher.
---

# GLM4

`scripts/` contains two GLM4 launchers, both targeting `GLM-Z1-9B-0414`:

- `scripts/run-glm4-9B.sh` — canonical 8-GPU recipe (4 actor + 4 rollout, non-colocate).
- `scripts/run-glm4-9B-4xgpu-radixtree.sh` — 4-GPU smoke-test recipe (`CUDA_VISIBLE_DEVICES=0,1,2,3`).

## Variants

| Model | HF ID | Model config | Has launcher? |
|---|---|---|---|
| GLM-Z1-9B-0414 | `zai-org/GLM-Z1-9B-0414` (per upstream) — script reads `/root/GLM-Z1-9B-0414` | `scripts/models/glm4-9B.sh` | ✓ `run-glm4-9B.sh` |


## Paths the launcher expects

From `run-glm4-9B.sh:29-62`:

```bash
CKPT_ARGS=(
   --hf-checkpoint /root/GLM-Z1-9B-0414/
   --ref-load /root/GLM-Z1-9B-0414_torch_dist
   --load /root/GLM-Z1-9B-0414_miles/
   --save /root/GLM-Z1-9B-0414_miles/
   --save-interval 20
)

ROLLOUT_ARGS=( --prompt-data /root/dapo-math-17k/dapo-math-17k.jsonl ... )
EVAL_ARGS=(   --eval-prompt-data aime /root/aime-2024/aime-2024.jsonl ... )
```

## Quick start (8 GPU)

```bash
cd /root/miles

hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir /root/dapo-math-17k
hf download --repo-type dataset zhuzilin/aime-2024     --local-dir /root/aime-2024
# Place the model checkpoint at /root/GLM-Z1-9B-0414

source scripts/models/glm4-9B.sh
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
   ${MODEL_ARGS[@]} \
   --hf-checkpoint /root/GLM-Z1-9B-0414 \
   --save          /root/GLM-Z1-9B-0414_torch_dist

bash scripts/run-glm4-9B.sh
```

## Parallelism

| Script | TP | PP | CP | EP | `max_tokens_per_gpu` | actor / rollout GPUs | Total |
|---|---|---|---|---|---|---|---|
| `run-glm4-9B.sh` | 2 | 1 | 2 | 1 | 4608 | 4 / 4 (non-colocate) | 8 |
| `run-glm4-9B-4xgpu-radixtree.sh` | 2 | 1 | 1 | 1 | 2304 | 4 / 2 | 4 |

Both scripts use GRPO with `--eps-clip 0.2 --eps-clip-high 0.28 --use-kl-loss --kl-loss-coef 0.00`. `run-glm4-9B-4xgpu-radixtree.sh` also has a commented-out `# --use-miles-router` line in `SGLANG_ARGS`.

## SGLang flags (from the scripts)

```bash
# run-glm4-9B.sh
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 2
)

# run-glm4-9B-4xgpu-radixtree.sh
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 2
   # --use-miles-router
)
```
