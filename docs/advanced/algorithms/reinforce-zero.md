---
title: "REINFORCE Zero"
description: "Group-mean advantages with unclipped token-level importance sampling against the rollout log-probs."
---

REINFORCE Zero ([Hu et al., 2026](https://github.com/hijkzzz/REINFORCE-Zero)) is a minimal recipe for
asynchronous RL. It removes the numerical mismatch between the rollout engine and the trainer, then
corrects the remaining policy lag with ordinary importance sampling: no clipping and no sample or
token rejection.

Enable it with:

```bash
--advantage-estimator reinforce_zero --use-rollout-logprobs
```

## Loss

For each prompt, the advantage of response `i` is its reward minus the mean reward of the
responses to the same prompt, with no division by the group's standard deviation. Every action
token of the response carries that advantage. With `mu` the probability the rollout engine
recorded for the sampled token and `pi_theta` the trainer's current probability:

```text
w_it = exp(log pi_theta(a_it | h_it) - log mu(a_it | h_it))
loss = -mean over responses ( mean over the response's loss-mask tokens ( A_i * w_it ) )
```

The gradient flows through `log pi_theta` only, so it is `-A_i * w_it * grad log pi_theta`. The
weight is never clipped: `--eps-clip` and `--eps-clip-high` have no effect, and `pg_clipfrac` is
always 0. The log-ratio is only kept inside `[-20, 20]` so that `exp` stays finite.

Startup rejects the settings that would change this loss: `--use-tis`, `--use-opsm`,
`--get-mismatch-metrics`, `--eps-clip-c`, `--normalize-advantages`,
`--disable-rewards-normalization`, `--calculate-per-token-loss`,
`--custom-pg-loss-reducer-function-path`, and any `--loss-type` other than `policy_loss`.
KL and entropy terms (`--use-kl-loss`, `--entropy-coef`) are added as for the other estimators.

## Rollout and trainer alignment

Unclipped importance sampling is only stable when the trainer scores a token the way the rollout
engine sampled it. The paper establishes this trainer–inference invariance with batch-invariant
kernels and matched forward computation; in Miles that is
[true on-policy mode](/examples/infra-features/true-on-policy) (`--true-on-policy-mode`). Without
it, kernel mismatch enters every importance weight, and the paper reports weight spikes hundreds of
times the median with standard kernels. Watch `train/train_rollout_logprob_abs_diff` to check the
alignment of a run.

The importance weight corrects policy lag, so the recipe pairs with
[fully async RL](/user-guide/fully-async), where a training batch can hold rollouts from several
older policy versions.
