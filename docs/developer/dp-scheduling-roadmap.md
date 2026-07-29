# DP scheduling roadmap: three proposals beyond the variable-gbs series

Status: **proposal — for team review**. Builds on the rollout-side scheduling
series (#1927 → #1938): the DP/microbatch schedule is now a pure function on
the rollout side, `global_batch_size` counts rollouts, loss/metrics aggregate
per rollout, and per-step batch sizes may vary (`--allow-partial-train-step`).

The three proposals below are ordered by ascending invasiveness. Each stands
alone.

## 1. Pluggable loss weighting (`--loss-weighting`)

**Problem.** The loss normalizer semantics are hardcoded per mode: per-token
(`--calculate-per-token-loss`), per-sample (legacy), per-rollout
(`rollout_mask_sums`, since #1931). Research variants (GSPO-style sequence
weighting, DAPO-style token weighting, length-debiasing schemes) each require
touching `loss.py` + `cp_utils.py`.

**Proposal.** The `sample_denoms` mechanism already parameterizes "what each
sample's contribution is divided by". Promote it to a first-class knob:

```
--loss-weighting {per_token, per_sample, per_rollout, custom}
--custom-loss-weighting-path my_pkg.my_weighting  # (args, train_data) -> list[float] denoms
```

The rollout side computes denominators next to `rollout_mask_sums` (whole-batch
visibility, step-level precompute — same reason as #1931) and ships them per
shard. `loss_function` and `log_rollout_data` consume them uniformly; the
`(sum, count)` metric reduction already handles arbitrary weights.

**Scope.** ~1 PR: arguments + one dispatch function on the rollout side +
denominator selection in `loss_function`. No new distributed machinery.

**Risk.** Custom paths can produce degenerate denominators (0, negative);
validate and clamp at conversion time.

## 2. Online cost-model packing

**Problem.** Packing balances token counts; `--balance-by-flops` (#1938)
balances an analytic FLOPs estimate. Neither matches measured step time: MoE
expert imbalance, CP communication, recompute policy, and kernel effects all
shift the true cost, and the analytic model needs a hand-maintained formula
per architecture.

**Proposal.** Fit the cost model online:

1. Instrument `train_one_step` to record `(mbs composition, wall time)` —
   the timers already exist (`perf/actor_train_time`); add per-mb timing.
2. After each rollout batch, fit `cost(sample) ≈ a·L + b·L²` (least squares
   over observed mbs; 2 parameters, dozens of observations per batch).
3. The next rollout batch packs and distributes with fitted costs instead of
   tokens/analytic FLOPs. Fall back to tokens until enough observations.

**Why rollout-side scheduling makes this cheap:** the fitted `(a, b)` are two
floats shipped from the train actor to the rollout manager alongside
`train_parallel_config` — no new collective, no schedule-time GPU access.

**Scope.** ~2 PRs: (a) per-mb timing + fit + report; (b) feed fitted costs
into `_pack_step_into_mbs` / distribution weights behind
`--balance-by-cost-model`.

**Risk.** Noisy timings (stragglers, clock skew) → robust fit (median-of-runs,
exponential moving average); cost drift after recompute/parallelism changes →
reset on config change.

## 3. Streaming schedule: overlap rollout generation with training

**Problem.** Training waits for the entire rollout batch before the first
optimizer step. With `num_steps_per_rollout > 1`, the first
`global_batch_size` rollouts are complete long before the last ones — dead
time on the training side (particularly for disaggregated placements, where
the training GPUs idle during generation).

**Proposal.** The schedule is per-step independent by construction (pack
first, distribute second — each step only needs its own rollouts). Stream it:

1. Rollout manager forms step schedules incrementally: as soon as
   `global_batch_size` rollouts are complete, build that step's schedule and
   `ray.put` its shard slice.
2. Train actors consume a queue of per-step shard refs instead of one
   rollout_data blob; `train()` already iterates steps.
3. Weight-sync boundary stays at the rollout-batch level (same on-policy
   semantics as today); only intra-batch generation/training overlap changes.

Interactions to design carefully:
- advantage normalization needs whole-batch statistics (GRPO group norm) —
  either compute per-group as groups complete (groups are
  `n_samples_per_prompt`-local, not batch-global) or restrict streaming to
  estimators with per-group normalization;
- `log_rollout_data` aggregates per step instead of per batch;
- fault tolerance: a failed step retry must not re-consume the stream.

**Scope.** Design doc first (this section), then ~3 PRs (queue plumbing,
streaming schedule, overlap in the actor loop). Highest effort, highest
wall-clock upside for disaggregated deployments.

## Suggested order

1 (small, unblocks research) → 2 (perf, incremental) → 3 (architectural, needs
its own detailed design).
