---
title: GLM-4.7-Flash on GSM8K
description: Step-by-step recipe for non-agentic math RL — GRPO fine-tuning of GLM-4.7-Flash on GSM8K, single 8×H200 node.
---
**What you'll learn:** how to run **reasoning-only** RL (no agent server, no tools) on
GLM-4.7-Flash using GSM8K. Rollouts are generated in-process by SGLang and scored by an
exact-match math reward. The whole loop fits on a **single 8×H200 node**.

This is the simplest useful RL setup in Miles: one dataset, one reward function, one
node. It's a good first run to validate a fresh cluster or image before moving on to
agentic training.

## How it works

Unlike agentic training (which needs an external environment/agent server), math RL is
fully self-contained:

1. **Rollout** — SGLang samples `n` completions per prompt, in-process.
2. **Reward** — `--rm-type math` extracts the boxed answer and compares it to the label
   with a sympy-based exact-match grader (`miles/rollout/rm_hub/math_utils.py`). Reward is
   `1.0` for a correct final answer, `0.0` otherwise.
3. **Advantage + update** — GRPO centers the reward within each prompt's sample group and
   updates the policy.

No agent server, no tools, no env server are involved.

## Prerequisites

- A single node with **8× H200** (80 GB+ per GPU).
- Miles + Megatron-LM installed (the standard `radixark/miles` image).
- `WANDB_API_KEY` exported if you want metric logging.

## Quick start

```bash
export WANDB_API_KEY=<your-key>

cd /root/miles
python scripts/run_glm47_flash_gsm8k.py --wandb-project glm47-flash-gsm8k
```

That single command downloads the model + dataset, converts the checkpoint (if needed),
and launches training. The steps below explain what it does, and how to point it at a
pre-staged checkpoint to skip the download.

### 1. Download model + dataset

```bash
hf download zai-org/GLM-4.7-Flash --local-dir /root/models/GLM-4.7-Flash
hf download --repo-type dataset zhuzilin/gsm8k --local-dir /root/datasets/gsm8k
```

The GSM8K parquet has columns `question`, `answer`, `label`, `messages`. The launcher
reads the chat-formatted `messages` column (`--input-key messages --apply-chat-template`)
and grades against `label` (`--label-key label`).

### 2. Convert HF → Megatron `torch_dist`

```bash
cd /root/miles
source scripts/models/glm4.7-flash.sh
PYTHONPATH=/root/Megatron-LM torchrun --nproc-per-node 8 \
   tools/convert_hf_to_torch_dist.py \
   ${MODEL_ARGS[@]} \
   --hf-checkpoint /root/models/GLM-4.7-Flash \
   --save          /root/models/GLM-4.7-Flash_torch_dist
```

The launcher does this automatically and skips it when the target already has a
`latest_checkpointed_iteration.txt` of `release`.

### 3. Launch

```bash
cd /root/miles
python scripts/run_glm47_flash_gsm8k.py --wandb-project glm47-flash-gsm8k
```

If the model and dataset are already staged (e.g. a shared cache), skip the download +
conversion entirely:

```bash
python scripts/run_glm47_flash_gsm8k.py \
   --skip-prepare \
   --model-dir /path/to/shared/models \
   --wandb-project glm47-flash-gsm8k
```

`--skip-prepare` requires that `<model-dir>/GLM-4.7-Flash`,
`<model-dir>/GLM-4.7-Flash_torch_dist`, and `/root/datasets/gsm8k/{train,test}.parquet`
already exist.

## Recipe configuration

### Parallelism

| TP | PP | CP | EP | expert-TP | `max_tokens_per_gpu` | GPUs |
|---|---|---|---|---|---|---|
| 4 | 1 | 1 | 8 | 1 | 32768 | 8 (1 × 8) |

`--rollout-num-gpus-per-engine 4` — TP must divide the 20 attention heads, so TP=4.

### Rollout & reward

```bash
--prompt-data /root/datasets/gsm8k/train.parquet
--input-key messages --label-key label --apply-chat-template
--rm-type math                 # exact-match math grader
--num-rollout 300              # >= 300 GRPO steps
--rollout-batch-size 32
--n-samples-per-prompt 8
--rollout-max-response-len 1024  # GSM8K answers are short
--rollout-temperature 1
--global-batch-size 256
```

`--rollout-max-response-len 1024` is deliberately short — GSM8K solutions are a few
hundred tokens. If you see `train_rollout_logprob_abs_diff` climbing while `raw_reward`
stays flat, responses are being truncated; raise this.

### Algorithm & optimizer

GRPO with `--eps-clip 0.2 --eps-clip-high 0.28 --use-kl-loss --kl-loss-coef 0.00`.
CPU-offloaded Adam (`--optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d
--use-precision-aware-optimizer`), `lr 1e-6`, constant schedule.

### SGLang

```bash
--rollout-num-gpus-per-engine 4
--sglang-mem-fraction-static 0.7
# EAGLE speculative decoding (MTP head)
--sglang-speculative-algorithm EAGLE
--sglang-speculative-num-steps 2
--sglang-speculative-eagle-topk 1
--sglang-speculative-num-draft-tokens 3
# Rollout Routing Replay
--use-rollout-routing-replay
```

Rollouts use the SGLang router. The Miles router (`--use-miles-router`) is intentionally
**not** enabled.

### Evaluation

Every 20 rollouts, the launcher evaluates on the GSM8K test split:

```bash
--eval-interval 20
--eval-prompt-data gsm8k /root/datasets/gsm8k/test.parquet
--n-samples-per-eval-prompt 1
--eval-max-response-len 1024
--eval-top-k 1
```

## What to watch

| Metric (W&B) | Meaning |
|---|---|
| `rollout/raw_reward` | Fraction of rollouts with a correct answer — the true solve rate. **Use this, not `rollout/rewards`.** |
| `rollout/rewards` | GRPO-centered advantage (≈ 0 by construction). Not a solve-rate signal. |
| `eval/gsm8k` | Held-out test accuracy, logged every 20 rollouts. |
| `train_rollout_logprob_abs_diff` | Rollout/train logprob drift. A fast climb usually means responses are truncating. |

A healthy run shows `rollout/raw_reward` rising over the first tens of steps and
`eval/gsm8k` tracking upward.

## Pairs well with

- [GLM4.7 Flash model card](/models/glm/glm4-7-flash) — the dapo-math recipe and full flag reference.
- [Reproducibility Recipe](/examples/reproducibility) — make this run bit-stable for A/B testing.
