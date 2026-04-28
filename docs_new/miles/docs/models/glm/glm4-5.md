---
title: GLM4.5
description: Launch recipes for GLM-4.5 (355B-A32B) — bash launcher and Python launcher.
---

# GLM4.5

GLM4.5 is Zhipu's MoE line: 106 B-A12B for two-node experimentation and 355 B-A32B for eight-node frontier runs. `scripts/` ships two launchers for the 355 B-A32B model and a model config for 106 B-A12B:

- `scripts/run-glm4.5-355B-A32B.sh` — canonical 8-node × 8-GPU bash launcher (FP8 HF checkpoint, GSPO, EAGLE speculative).
- `scripts/run_glm45_355b_a32b.py` — Typer-based Python launcher (GRPO, supports `--rollout-fp8`, `--enable-mtp`, `--dynamic-sampling`, `--enable-mis`).
- `scripts/models/glm4.5-106B-A12B.sh` — model config only, no `run-glm4.5-106B-A12B.sh` launcher exists.

## Variants

| Model | Active / Total | HF ID | Model config |
|---|---|---|---|
| GLM-4.5-355B-A32B | 32 B / 355 B | `zai-org/GLM-4.5` | `scripts/models/glm4.5-355B-A32B.sh` |
| GLM4.5-106B-A12B | 12 B / 106 B | — | `scripts/models/glm4.5-106B-A12B.sh` |


## Required env (bash launcher)

`run-glm4.5-355B-A32B.sh` reads:

- `BASE_DIR` — shared FS reachable from every node, holds the staged checkpoint and datasets
- `MASTER_ADDR` — the script overrides it to `${MLP_WORKER_0_HOST}` at L144, so this comes from the cluster orchestrator


## Paths the bash launcher expects

From `run-glm4.5-355B-A32B.sh:29-61`:

```bash
CKPT_ARGS=(
   --hf-checkpoint $BASE_DIR/GLM-4.5-355B-A32B
   --ref-load $BASE_DIR/GLM-4.5-355B-A32B_torch_dist/
)

ROLLOUT_ARGS=(
   --prompt-data $BASE_DIR/dapo-math-17k/dapo-math-17k.jsonl
   ...
   --rollout-stop-token-ids 151329 151336 151338
)

EVAL_ARGS=(
   --eval-prompt-data aime $BASE_DIR/rl_data/aime-2024.jsonl
   ...
)
```

Note: the bash launcher does **not** set `--load`/`--save` in `CKPT_ARGS` — `--load` defaults to the value of `--ref-load`.

## Quick start (bash launcher, 8 nodes × 8 GPU)

```bash
cd /root/miles
export BASE_DIR=...      # shared FS
# MASTER_ADDR is overridden to MLP_WORKER_0_HOST inside the script

bash scripts/run-glm4.5-355B-A32B.sh
```

The script handles the Ray fan-out itself via the `ssh` loop over `/root/mpi_rack_hostfile`. You need to have already produced `$BASE_DIR/GLM-4.5-355B-A32B_torch_dist/` (run `tools/convert_hf_to_torch_dist.py` over the appropriate parallelism — the bash launcher doesn't do this for you).

## Quick start (Python launcher)

`scripts/run_glm45_355b_a32b.py` automates the full flow (download → optional `tools/convert_hf_to_fp8.py` → `convert_checkpoint` → `rsync` to `model_local_dir` → submit). Note `_execute_train` asserts `args.hardware != "H100"` (L100), so this Python launcher is for Blackwell hardware.

```bash
cd /root/miles
python scripts/run_glm45_355b_a32b.py train --hardware GB300
```

## Parallelism


| Source | TP | PP | CP | EP | expert-TP | `max_tokens_per_gpu` | GPUs |
|---|---|---|---|---|---|---|---|
| `run-glm4.5-355B-A32B.sh` | 8 | 4 | 2 | 16 | 1 | 16384 | 64 (8 × 8) |
| `run_glm45_355b_a32b.py` (`num_nodes ≤ 4`, debug) | 4 | 1 | 1 | 4 | 1 | 16384 | up to 32 |
| `run_glm45_355b_a32b.py` (`num_nodes == 8`) | 4 | 8 | 2 | 8 | 1 | 16384 | 64 |

## Algorithms

| Source | Advantage | Notable flags |
|---|---|---|
| `run-glm4.5-355B-A32B.sh` | GSPO | `--eps-clip 1e-4 --eps-clip-high 2e-4 --use-tis`, `--use-kl-loss`|
| `run_glm45_355b_a32b.py` | GRPO | `--eps-clip 1e-4 --eps-clip-high 2e-4 --use-tis`, `--use-kl-loss`|

Neither launcher enables `--use-miles-router` / `--use-rollout-routing-replay` by default. The Python launcher exposes `--enable-mis` (TIS/RS config) as an opt-in.

## SGLang flags

```bash
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 32
   --sglang-mem-fraction-static 0.7
   --sglang-enable-dp-attention
   --sglang-dp-size 4
   --sglang-ep-size 32
   --sglang-enable-dp-lm-head
   --sglang-moe-dense-tp-size 1

   # mtp / EAGLE
   --sglang-speculative-algorithm EAGLE
   --sglang-speculative-num-steps 1
   --sglang-speculative-eagle-topk 1
   --sglang-speculative-num-draft-tokens 2
   --sglang-enable-draft-weights-cpu-backup
)
```

Megatron side: `--moe-token-dispatcher-type flex`, `--moe-enable-deepep`. CPU Adam (`--optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d --use-precision-aware-optimizer`) is on.

## Pairs well with

- [FP8 & Low Precision](../../advanced/fp8-low-precision.md)
- [INT4 QAT](../../advanced/int4-qat.md)
- [Miles Router (R3)](../../advanced/miles-router.md) — opt-in via `--enable-mis` on the Python launcher; not on by default in either launcher.
