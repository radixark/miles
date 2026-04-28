---
title: Kimi K2
description: Launch recipes for Kimi-K2-Instruct and Kimi-K2-Thinking — 32 nodes × 8 GPU.
---

# Kimi K2

Kimi K2 is Moonshot's 1 T-parameter MoE — 32 B active per token, 61 layers, Instruct and Thinking variants. Miles treats it as a DeepSeek-V3-shaped architecture (one `sed` away), converts on 4 nodes, and trains on 16. K2-Thinking is also the canonical target for INT4 QAT. `scripts/` ships two K2 launchers:

- `scripts/run-kimi-k2-Instruct.sh` — sources `scripts/models/kimi-k2.sh`, GRPO baseline.
- `scripts/run-kimi-k2-Thinking.sh` — sources `scripts/models/kimi-k2-thinking.sh`, GRPO + `--use-tis`, defaults to a pre-staged FP8 HF checkpoint.

Both launchers submit to an **already-running Ray cluster** (`ray job submit ...`); neither runs `ray start --head` itself.

## Architecture

| Property | Value |
|---|---|
| Layers | 61 |
| First-K dense layers | 1 (rest are MoE per `MOE_LAYER_FREQ`) |
| Hidden / FFN | 7168 / 18432 |
| Attention | MLA (DeepSeek-V3 shape) |

Full `MODEL_ARGS` are in `scripts/models/kimi-k2.sh` (Instruct) and `scripts/models/kimi-k2-thinking.sh` (Thinking).

## Required env

`BASE_DIR` and `MASTER_ADDR` are referenced but never set inside the scripts — export them yourself before launch.

## Paths the launchers expect

`run-kimi-k2-Instruct.sh:29-68`:

```bash
CKPT_ARGS=(
   --hf-checkpoint $BASE_DIR/Kimi-K2-Instruct/
   # --hf-checkpoint $BASE_DIR/Kimi-K2-bf16/
   --ref-load $BASE_DIR/Kimi-K2_torch_dist/
   --load $BASE_DIR/Kimi-K2_miles/
   --save $BASE_DIR/Kimi-K2_miles/
   --save-interval 20
)

ROLLOUT_ARGS=( --prompt-data $BASE_DIR/dapo-math-17k/dapo-math-17k.jsonl ...
               --rm-type math ... )
EVAL_ARGS=(   --eval-prompt-data aime $BASE_DIR/rl_data/aime-2024.jsonl ... )
```

`run-kimi-k2-Thinking.sh:29-68`:

```bash
CKPT_ARGS=(
   # --hf-checkpoint $BASE_DIR/Kimi-K2-Thinking-bf16/
   --hf-checkpoint $BASE_DIR/Kimi-K2-Thinking-fp8/
   --ref-load $BASE_DIR/Kimi-K2-Thinking_torch_dist/
   --load $BASE_DIR/Kimi-K2-Thinking_miles/
   --save $BASE_DIR/Kimi-K2-Thinking_miles/
   --save-interval 20
)

EVAL_ARGS=(   --eval-prompt-data aime $BASE_DIR/aime-2024.jsonl ... )
```

Note the differences: Instruct loads the BF16 HF checkpoint by default (FP8 commented out) and reads eval data from `$BASE_DIR/rl_data/`; Thinking loads the FP8 HF checkpoint by default and reads eval data from `$BASE_DIR/`.

## Quick start (32 nodes × 8 GPU)

```bash
export BASE_DIR=<shared FS path>
export MASTER_ADDR=<head node IP>

# Bring up Ray on every node first (the launcher only does `ray job submit`)
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats   # head
ray start --address=${MASTER_ADDR}:6379 --num-gpus 8 --node-ip-address ${WORKER_IP}    # each worker

# Stage data + (already-converted) torch_dist checkpoint under $BASE_DIR before launch.
bash scripts/run-kimi-k2-Thinking.sh   # or run-kimi-k2-Instruct.sh
```

## Parallelism (identical for both)

| TP | PP | CP | EP | expert-TP | `decoder-last-pipeline-num-layers` | `max_tokens_per_gpu` | Nodes × GPUs |
|---|---|---|---|---|---|---|---|
| 8 | 8 | 4 | 32 | 1 | 5 | 16384 | 32 × 8 = 256 |

Both scripts pass `--actor-num-nodes 32 --actor-num-gpus-per-node 8 --colocate --update-weight-buffer-size $((4*512*1024*1024))` to `train.py`.

## Algorithm differences

| Script | Advantage | TIS | Other |
|---|---|---|---|
| Instruct | GRPO (`--eps-clip 0.2 --eps-clip-high 0.28`) | – | – |
| Thinking | GRPO (`--eps-clip 0.2 --eps-clip-high 0.28`) | `--use-tis` | – |

Both use `--use-kl-loss --kl-loss-coef 0.00 --kl-loss-type low_var_kl --entropy-coef 0.00`.

## Rollout shape

Both:

```bash
--rm-type math
--num-rollout 100
--rollout-batch-size 128
--n-samples-per-prompt 8
--rollout-max-response-len 32768   # Instruct
--rollout-max-response-len 16384   # Thinking
--over-sampling-batch-size 256
--dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std
--num-steps-per-rollout 4
--balance-data
```

DAPO-style dynamic sampling is on by default in both K2 scripts. `--global-batch-size 1024` is commented out.

## SGLang flags (identical for both)

```bash
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 16
   --sglang-mem-fraction-static 0.7

   # dp attention
   --sglang-enable-dp-attention
   --sglang-dp-size 8
   --sglang-moe-dense-tp-size 1
   --sglang-enable-dp-lm-head

   --sglang-ep-size 16

   # deepep — commented out in both scripts
   # --sglang-enable-deepep-moe
   # --sglang-deepep-mode auto

   --sglang-server-concurrency 1024
)
```

CPU Adam is enabled in both (`--optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d --use-precision-aware-optimizer`).

Megatron-side `--moe-enable-deepep` and `--moe-token-dispatcher-type flex` are **on** in the Instruct script but **commented out** in the Thinking script.

## Pairs well with

- [PD Disaggregation](../../advanced/pd-disaggregation.md)
- [P2P Weight Transfer](../../advanced/p2p-weight-transfer.md)
- [Fault Tolerance](../../advanced/fault-tolerance.md)
- [INT4 QAT](../../advanced/int4-qat.md)
