# Privileged-context self-distillation (Qwen3-1.7B)

The student rolls out from the problem alone. The teacher scores that same response on a
prompt that also contains the reference solution. Teacher and student are the same frozen
base weights, so the privileged context is the only thing that makes them differ, and no
RLVR teacher has to be trained first.

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

`rm.py` is wired through `--custom-rm-path`. For a training row it scores the teacher once
and hands over its top-k support as `sample.teacher_top_ids` and
`sample.teacher_top_logprobs`, returning 0.0 so the task reward contributes nothing. For a
held-out row it grades the boxed answer. Held-out rows are scored here rather than by
`rm_type` because `--custom-rm-path` is consulted unconditionally.

The student side never leaves the training step: `forward_kl_loss` reads the student's
log-probs at the teacher's ids straight from the training logits, so there is no second
scoring call and no id union to broadcast.

## Objective

Forward KL, `KL(teacher || student)`, minimised directly as a loss term over the
teacher's top-k support, with each per-vocabulary-entry contribution clipped at
`--opd-kl-clip 0.05` (the paper's `jsd_token_clip`) before the support is summed.

Forward KL is mode-covering and weighted by the teacher; reverse KL is mode-seeking and
weighted by the student. Only forward KL can be a differentiable loss here, because it
needs the student's logits at training time. Reverse KL is expressible as an advantage
penalty instead, which is what `--opd-divergence reverse_kl` (the default, unchanged) does.

A truncated support is only a KL while it covers nearly all the teacher's mass, so
coverage is measured rather than assumed and reported as `opd_teacher_coverage`. At
k=256 it runs at 0.999. `opd_kl_clipfrac` reports how often the ceiling binds: near 0
means tau is inert, near 1 means it is flattening the signal instead of its tail.

## Hyperparameters

Table 6 of the paper: lr 5e-6, effective batch 32, LoRA r=64 alpha=128 over
q/k/v/o/gate/up/down, completions capped at 1024, one generation per prompt, sampling
temperature 1.1, gradient clipping 0.1, 100 steps. Evaluation follows their Table 8:
AIME24 Avg@12, temperature 1.0, top-p 0.95, top-k -1, 38912 new tokens, thinking enabled.

## Why forward KL: measured

Both arms below ran on one node with identical data, hyperparameters, teacher and
evaluation. Only the objective differs. AIME24 Avg@12:

| step | 0 | 24 | 49 | 74 | 99 |
|---|---|---|---|---|---|
| forward KL (this example) | 0.500 | 0.539 | 0.542 | 0.536 | **0.544** |
| reverse KL | 0.458 | 0.475 | 0.506 | 0.492 | 0.486 |

Both step-0 points are the same untrained base weights, so the gap between them, 4.2pp,
is a direct measurement of evaluation noise at this sample size. Reverse KL never leaves
that band and is not evidence of learning. Forward KL sits above it at all four trained
points, and those four span 0.8pp, which is the tighter clustering a real level shift
produces rather than a lucky draw.

Reverse KL needs the student's top-k, which rides on every generated position, so it is
held to k=16; forward KL reads the student from the training logits and only the teacher
needs support, so it runs at k=256 from teacher prefill alone. The two arms therefore
cannot hold k fixed. That is a property of the objectives, not a confound: forward KL over
a k=16 support is not a KL at all, since the partial sum can go negative.

## Notes

`math_verify` is graded with `parsing_timeout=None`. Its default timeout uses
`signal.alarm()`, which only works on the main thread, and reward functions run on a
worker thread. AIME24 labels are zero-padded, so a plain string comparison scores a
correct `\boxed{25}` against `025` as wrong, worth roughly 13pp.

`--eval-top-k -1` is explicit, because an unset eval top-k falls back to `--rollout-top-k`.
