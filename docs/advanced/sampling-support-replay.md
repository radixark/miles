---
title: Sampling-Support Replay
description: Train against the same bounded sampling distribution used during rollout.
---

Top-p and top-k change the distribution that generates a rollout token. They
remove tokens from the vocabulary and renormalize the remaining probability
mass. Recomputing a token's log probability over the full vocabulary therefore
does not reproduce its behavior-policy probability, even when the rollout and
trainer weights are identical.

Sampling-support replay preserves that distribution. For every generated token,
SGLang returns the realized set of token IDs that remained after sampling
filters. Miles transports this ragged support with the sample and applies it to
the actor logits before the softmax:

$$
q_\theta(a_t \mid S_t) =
\frac{\exp(z_\theta(a_t) / T)}
{\sum_{j \in S_t} \exp(z_\theta(j) / T)}.
$$

The support `S_t` is fixed rollout data; `z_θ` comes from the actor being
trained. This gives PPO/GRPO a denominator and numerator in the same sampled
probability space while retaining gradients through the current actor logits.

## Enable replay

```bash
ROLLOUT_ARGS+=(
  --rollout-temperature 1.0
  --rollout-top-p 0.95
  --rollout-top-k 64
)
```

Replay is enabled whenever `--rollout-top-p` is below `1` or
`--rollout-top-k` is positive. A finite top-p run must also set a positive
top-k so that the captured support is bounded. Miles asks SGLang to capture its
native sampling support with `return_sampling_mask` and uses the returned
support-normalized log probability as the rollout log probability.

`--rollout-top-k` is the default for rollout requests, not a global upper bound
on request-specific top-k values. SGLang owns support capacity through
`--sglang-sampling-mask-max-tokens` (or the matching server-group override) and
rejects requests whose realized support cannot be represented.

## Correctness requirements

Miles rejects configurations that it cannot replay faithfully:

- Every resolved training request must have top-p, top-k, and temperature. Agentic
  sessions fill omitted values from the sampling defaults registered when the
  session is created.
- Request temperature must match the registered training temperature
  (`--rollout-temperature` for the standard rollout paths).
- Frequency, presence, and repetition penalties, `logit_bias`, and custom logit
  processors are not supported because the trainer does not replay those logit
  transformations.
- `--recompute-logprobs-via-prefill` is incompatible because that path does not
  preserve the per-token support.
- The selected router must preserve `return_sampling_mask`. The Miles router
  forwards raw request bodies. The native SGLang router requires typed chat and
  generate request schemas that declare the field.

The SGLang backend must return one complete support and its normalized log
probability for every sampled token. Miles validates that response contract but
does not add separate restrictions for speculative decoding.

Tool and environment tokens are recorded with singleton support. Evaluation
requests do not capture this training-only metadata.

## Current objective limitation

Replay currently exposes only the support-normalized actor score to the loss.
Reference KL and on-policy distillation also need a full-vocabulary actor score,
so Miles rejects `--use-kl-loss`, nonzero `--kl-coef`, and `--use-opd` with
replay until the loss interface carries both scores.

## Monitor replay

On the first update from identical weights,
`train/train_rollout_logprob_abs_diff` should be near numerical tolerance.
Afterward, interpret it with version-lag and clipping metrics because it also
reflects policy updates and staleness.
