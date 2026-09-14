---
title: "Privileged-context self-distillation (Qwen3-1.7B)"
description: "Privileged-context self-distillation of Qwen3-1.7B with forward KL and example-only clipping."
# Generated from examples/on_policy_distillation/qwen3_1_7b_opsd/README.md by scripts/tools/sync_example_docs.py. Edit that README, not this file.
---
The student rolls out from the problem alone. The teacher scores that same response on a
prompt that also contains the reference solution. Both start from the same base checkpoint.
The teacher stays frozen while the student's LoRA adapters train; no RLVR teacher is needed.

Based on [Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language
Models](https://arxiv.org/abs/2601.18734) (Zhao et al.).

## Running

```bash
hf download Qwen/Qwen3-1.7B --local-dir /root/Qwen3-1.7B
hf download --repo-type dataset open-r1/OpenThoughts-114k-math --local-dir /root/openthoughts-math
hf download --repo-type dataset HuggingFaceH4/aime_2024 --local-dir /root/aime24
pip install math_verify

bash examples/on_policy_distillation/qwen3_1_7b_opsd/run_opsd.sh
```

## How it works

`prepare_data.py` renders both prompts up front, so `--apply-chat-template` stays off.
That is what lets the student train with thinking mode off while the teacher and the
evaluation keep it on, which is the configuration the paper adopts. Each training row
carries the rendered teacher prompt in `metadata`.

`rm.py` scores the teacher once and stores its top-k support in `sample.train_metadata["opd"]`
using the core extraction helper. Training rewards are zero; held-out rows are graded for accuracy.
`loss.py` uses the core forward-KL contribution iterator and reads student probabilities
directly from training logits. The launcher selects it through `--custom-loss-function-path`
and disables advantage computation.

## Objective

Forward KL, `KL(teacher || student)`, minimised directly as a loss term over the
teacher's top-k support. The example clips each vocabulary contribution before summing,
using `opsd_kl_clip: 0.05` in `config.yaml` (the paper's `jsd_token_clip`). Core forward KL
is unclipped; clipping and its `opd_kl_clipfrac` metric belong to this example.

Teacher probabilities retain their full-vocabulary normalization. The truncated sum
approximates forward KL, and `opd_teacher_coverage` measures the retained teacher mass.
This example currently requires the Megatron backend, an SGLang teacher, and tensor-
and context-parallel size 1. Teacher and student must share a tokenizer.

## Hyperparameters

Table 6 of the paper: lr 5e-6, effective batch 32, LoRA r=64 alpha=128 over
q/k/v/o/gate/up/down, completions capped at 1024, one generation per prompt, sampling
temperature 1.1, gradient clipping 0.1, 100 steps. Evaluation follows their Table 8:
AIME24 Avg@12, temperature 1.0, top-p 0.95, top-k -1, 38912 new tokens, thinking enabled.

## Results

AIME24 Avg@12, best eval over 100 steps. Both arms ran on one node with identical data,
hyperparameters, teacher and evaluation; only the objective differs. These measurements
come from the original implementation and have not been rerun for this refactor.

| | AIME24 |
|---|---|
| baseline | 0.458 |
| reverse KL | 0.506 |
| forward KL | 0.569 |

## Notes

`math_verify` is graded with `parsing_timeout=None`. Its default timeout uses
`signal.alarm()`, which only works on the main thread, and reward functions run on a
worker thread. AIME24 labels are zero-padded, so a plain string comparison scores a
correct `\boxed{25}` against `025` as wrong, worth roughly 13pp.

`--eval-top-k -1` is explicit, because an unset eval top-k falls back to `--rollout-top-k`.
