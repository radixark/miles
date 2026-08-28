# Privileged-context self-distillation (Qwen3-1.7B)

The student rolls out from the problem alone. The teacher scores that same response on a
prompt that also contains the privileged context reference solution. This re uses the existing
`--opd-loss` derived from the privileged teacher's log probs.

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

`rm.py` is wired through `--custom-rm-path`. For a training row it scores the teacher and
sets `sample.teacher_log_probs`, returning 0.0 so the task reward contributes nothing. For
a held-out row it grades the boxed answer. Held-out rows are scored here rather than by
`rm_type` because `--custom-rm-path` is consulted unconditionally.

## Hyperparameters

Table 6 of the paper: lr 5e-6, effective batch 32, LoRA r=64 alpha=128 over
q/k/v/o/gate/up/down, completions capped at 1024, one generation per prompt, sampling
temperature 1.1, gradient clipping 0.1, 100 steps. Evaluation follows their Table 8:
AIME24 Avg@12, temperature 1.0, top-p 0.95, top-k -1, 38912 new tokens, thinking enabled.

## Objective

Forward KL over the teacher's top-k support, clipped per vocabulary entry at tau=0.05,
which is what the paper adopts (its beta=0 case with `jsd_token_clip`). It is computed in
`rm.py` and handed to miles as `sample.opd_reverse_kl`, the per-token divergence that
`--use-opd` subtracts from the advantage.

The student's log-probs at the teacher's ids need a second scoring call against the
rollout engine, which costs roughly 14 to 24 seconds per step depending on the support
width. The teacher is sharp because it has read the solution, so its top-16 union is only
about 1000 ids and the call stays cheap; scoring the student's own top-k union would be
several times larger.

`math_verify` is graded with `parsing_timeout=None`. Its default timeout uses
`signal.alarm()`, which only works on the main thread, and reward functions run on a
worker thread. AIME24 labels are zero-padded, so a plain string comparison scores a
correct `\boxed{25}` against `025` as wrong.
