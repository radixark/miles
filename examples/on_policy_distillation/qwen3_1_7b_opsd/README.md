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
sets `sample.opd_reverse_kl`, returning 0.0 so the task reward contributes nothing. For
a held-out row it grades the boxed answer. Held-out rows are scored here rather than by
`rm_type` because `--custom-rm-path` is consulted unconditionally.

## Hyperparameters

Table 6 of the paper: lr 5e-6, effective batch 32, LoRA r=64 alpha=128 over
q/k/v/o/gate/up/down, completions capped at 1024, one generation per prompt, sampling
temperature 1.1, gradient clipping 0.1, 100 steps. Evaluation follows their Table 8:
AIME24 Avg@12, temperature 1.0, top-p 0.95, top-k -1, 38912 new tokens, thinking enabled.

## Objective

Top-k reverse KL, clipped per vocabulary entry at tau=0.05, which is the paper's
`jsd_token_clip`. For each response position `rm.py` sums `p_S(v) * (log p_S(v) - log
p_T(v))` over the ids both sides ranked, clipping each entry before the sum so a few
stylistic tokens cannot carry the update, and hands the result to miles as
`sample.opd_reverse_kl`.

Neither side costs an extra scoring call. `--opd-log-prob-top-k` puts `top_logprobs_num`
on the rollout request, so the student's top-k arrives with generation and lands in
sample metadata; the teacher's comes from the same call that reads the privileged prompt.
That is what keeps the objective expressible as a reward function. It does require
`MILES_USE_LEGACY_ROLLOUT_V1=1`, since only the v1 rollout records the student's
per-position top-k.

The paper's headline objective is forward KL rather than reverse, and that one cannot
live here. The advantage path applies a per-token scalar through the policy gradient,
which recovers the reverse-KL gradient but not the forward-KL one, and forward KL needs
the student's logits at training time to be differentiable. It is a core feature, so this
example does not reproduce the paper's accuracy curve.

`math_verify` is graded with `parsing_timeout=None`. Its default timeout uses
`signal.alarm()`, which only works on the main thread, and reward functions run on a
worker thread. AIME24 labels are zero-padded, so a plain string comparison scores a
correct `\boxed{25}` against `025` as wrong.
