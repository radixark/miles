---
title: Metrics and Debugging Deep Dive
description: Every metric Miles logs, with its exact formula and source line, plus a symptom-to-lever playbook and the offline debugging toolkit.
---
This page is the reference behind [Monitoring and Logging](/user-guide/monitoring) and
[Debugging](/developer/debug). It lists every key Miles sends to wandb, TensorBoard,
MLflow, Prometheus and stdout, states the formula each one is computed with, points at the
line that computes it, and then turns that into a vital-signs dashboard, a diagnostic
playbook and a recipe for reproducing one training step offline.

It assumes you know PPO/GRPO. What it adds is the part that is easy to get wrong when
reading a Miles dashboard: **which process computed a number, what it was averaged over
(tokens, samples, rollouts or ranks), whether it was computed before or after filtering
and masking, and which x-axis it is plotted against.**

Related pages, which this one links to rather than repeats:

- [Rollout Correction Methods](/examples/infra-features/train-infer-mismatch-helper) for the
  TIS/MIS algorithms and the full `mis_*` metric list.
- [Fully Async RL](/user-guide/fully-async#metrics) for the meaning of the
  `rollout/fully_async/*` staleness metrics.
- [`miles/dashboard/README.md`](https://github.com/radixark/miles/blob/main/miles/dashboard/README.md)
  for the dashboard's architecture and views.

<Note>
All line numbers refer to commit `edf2c7ae3e` (rebased onto upstream `e30d9a8ec6`). When one
drifts, search for the function name given next to it.
</Note>

## Where metrics come from

Four emitters write to the same run. They live in different processes, so they do not see
the same data.

| Emitter | Process | Function | Namespaces | Step key |
|---|---|---|---|---|
| Rollout logger | Rollout executor (Ray actor) | `log_rollout_data`, `miles/ray/rollout/metrics.py:75` | `rollout/*`, `perf/*`, `passrate/*`, plus whatever the rollout function returns | `rollout/step` |
| Eval logger | Rollout executor | `log_eval_rollout_data`, `miles/ray/rollout/metrics.py:25` | `eval/*` | `eval/step` |
| Train-side rollout logger | Training actor, DP-source rank | `log_rollout_data`, `miles/backends/training_utils/metrics/log_utils.py:169`; `log_multi_turn_data`, `:367` | `rollout/*`, `multi_turn/*` | `rollout/step` |
| Train-step logger | Training actor, first-replica main rank | `log_train_step`, `miles/backends/training_utils/metrics/log_utils.py:500` | `train/*` | `train/step` |
| Perf logger | Training actor, rank 0 (on Megatron, TP 0 / DP×CP 0 of the last PP stage) | `log_perf_data_raw`, `miles/backends/training_utils/metrics/perf.py:14` | `perf/*` | `rollout/step` |

All of them call `tracking.log(args, metrics, step_key)`
(`miles/utils/tracking_utils/tracking.py:46`), which fans out to every enabled backend:
`--use-wandb`, `--use-tensorboard`, `--use-mlflow`, `--use-prometheus` (skipped under
`--cluster-backend kubernetes`), `--use-miles-dashboard`, and the CI history recorder
(`miles/utils/tracking_utils/tracking.py:23`). The same dict is also printed to stdout.

Two consequences follow directly:

- The `rollout/*` namespace has **two writers**. The rollout-side keys (`rollout/truncated_ratio`,
  `rollout/response_len/mean`, `rollout/zero_std/*` and so on) are computed from `Sample`
  objects on the rollout executor. The train-side keys (`rollout/raw_reward`,
  `rollout/log_probs`, `rollout/advantages` and so on) are computed from the tensorized
  batch on the training ranks, after log-probs and advantages exist. They never collide,
  but they are not interchangeable: for example `rollout/response_len/mean` counts only
  loss-masked tokens while `rollout/response_lengths` counts the whole response.
- Anything the rollout function returns in `RolloutFnTrainOutput.metrics` is merged
  into the rollout-side dict **without a prefix**. The built-in functions prefix their own
  keys (`rollout/dynamic_filter/*`, `rollout/raw_reward_unfiltered`, `rollout/fully_async/*`);
  a custom function that returns `agent/turns_mean` (as
  `examples/swe-agent-harbor-docker/generate.py` does) gets a key outside every declared
  namespace. See the next section for why that matters. `--fully-async` refuses
  `--rollout-function-path` (`miles/utils/arguments.py:98`), so such custom-function keys
  exist only in runs without `--fully-async`; in fully-async runs extra rollout metrics
  have to come from the `get_metrics()` of a `--custom-async-data-buffer-path` class.

## Step semantics and the x-axis

`_init_wandb_common` (`miles/utils/tracking_utils/wandb_utils.py:162`) declares three step
keys and binds namespaces to them:

| Namespace | Plotted against |
|---|---|
| `train/*` | `train/step` |
| `rollout/*`, `multi_turn/*`, `passrate/*`, `perf/*` | `rollout/step` |
| `eval/*` | `eval/step` |

The values of the step keys are:

$$
\texttt{rollout/step} = \texttt{eval/step} = \begin{cases} k & \text{default} \\ \left\lfloor \dfrac{k \cdot \texttt{rollout\_batch\_size} \cdot \texttt{n\_samples\_per\_prompt}}{\texttt{global\_batch\_size}} \right\rfloor & \texttt{--wandb-always-use-train-step}\end{cases}
\qquad
\texttt{train/step} = k \cdot S + s
$$

where $k$ is the rollout id, $S$ is the number of optimizer steps in that rollout
(`len(num_microbatches)`) and $s$ is the step index within it
(`compute_rollout_step`, `miles/utils/metric_utils.py:162`; `log_train_step`,
`miles/backends/training_utils/metrics/log_utils.py:529`).

Things that trip people up:

- **One rollout id, several optimizer steps.** With `--num-steps-per-rollout S` (or
  `global_batch_size < rollout_batch_size * n_samples_per_prompt`), each `rollout/step`
  corresponds to $S$ points on the `train/*` panels. The `rollout/*` means are over the
  whole rollout, the `train/*` values are per optimizer step.
- **`train/step` is not monotone-safe under a variable step count.** When the rollout-side
  scheduler or `--use-dynamic-global-batch-size` changes $S$ between rollouts, $k \cdot S + s$
  can repeat or skip values.
- **Eval is off by one relative to training.** `eval/step = k` is logged after rollout
  $k$ has been trained, so it measures the weights after $k+1$ updates. The pre-training
  eval (`train.py:97`) is also logged at `eval/step = 0`, so with `--eval-interval 1` two
  points land on x = 0.
- **Keys outside the declared namespaces plot against wandb's internal `_step`**, which
  advances on every `wandb.log` call from every process sharing the run. `agent/*` from the
  SWE example, engine metrics forwarded by `--sglang-enable-metrics` (`sgl_engine.*`), and
  any custom key without a `rollout/` prefix fall in this bucket. Prefix custom metrics with
  `rollout/` to get the right axis.
- **TensorBoard and MLflow have no step bindings.** They receive the step key's value as
  the global step for every scalar, so `train/*` and `rollout/*` share one axis with
  different units.
- **Multi-policy runs (`train_multi_policy.py`) prefix every rollout, perf and train key
  with `<model_id>/`** and bind `<model_id>/*` to `<model_id>/rollout/step` and
  `<model_id>/train/*` to `<model_id>/train/step` (`namespace_metrics`,
  `miles/utils/metric_utils.py:23`; `define_policy_metric_groups`,
  `miles/utils/multi_policy/utils.py:166`). `eval/*` is not prefixed.

## How numbers are reduced

The single most important thing to know about Miles metrics is that **the default unit
of averaging is the rollout (the sample, in ordinary single-turn runs), not the token.**
A 30k-token trajectory and a 300-token one contribute equally to `train/pg_clipfrac`. This
differs from the token-weighted means in open-instruct and verl.

### Notation

For sample $i$ in a training step: $m_{i,t} \in \{0,1\}$ is the loss mask on response
token $t$ (0 on prompt tokens, on tool and observation tokens in multi-turn runs, and on
every token of a sample with `remove_sample=True`). Samples produced by the same rollout
share a `rollout_id` and are called siblings; $D_i$ is the loss-mask total over all
siblings of sample $i$ (`rollout_mask_sums`,
`miles/ray/rollout/train_data_conversion.py:249`). In ordinary runs every rollout is one
sample and $D_i = \sum_t m_{i,t}$.

The log-probs of the sampled token under the four policies Miles tracks are:

| Symbol | Meaning | Batch key |
|---|---|---|
| $\ell^\mu_{i,t}$ | SGLang, recorded at sampling time | `rollout_log_probs` |
| $\ell^{\text{old}}_{i,t}$ | trainer forward-only pass before the update (the PPO anchor) | `log_probs` |
| $\ell^\theta_{i,t}$ | trainer, inside the loss, with gradient | computed in `policy_loss_function` |
| $\ell^{\text{ref}}_{i,t}$ | reference model | `ref_log_probs` |

All four are computed on logits divided by `--rollout-temperature`. The trainer does this
in `_upcast_chunk_to_fp32` (`miles/backends/training_utils/loss/hub/math_utils.py:961`).
For $\ell^\mu$, SGLang divides by the request temperature, which Miles sets to
`--rollout-temperature`, before its softmax, and it returns the log-prob of the full-vocabulary
distribution before any top-p/top-k truncation
(`sglang/python/sglang/srt/layers/sampler.py:186`, `:233`, `:249-254`; verified against
`sglang-miles-v0.5.19-final` at `c9660a7e11`; the same path is at `sampler.py:248`, `:275-281`
in SGLang 0.5.21, while the 0.5.22 build used in production was not checked). With `--rollout-top-p < 1` or
`--rollout-top-k > 0`, sampling-support replay replaces $\ell^\mu$ with the log-prob
renormalized over the realized support, and the trainer masks its logits to that same
support (see [the RL step walkthrough, section 2.2](/deep-dive/rl-step-walkthrough)). There are two
exceptions where $\ell^\mu$ is **not** temperature-scaled. The first is
`SGLANG_RETURN_ORIGINAL_LOGPROB=1`, which Miles forces to `0` on every engine it launches
(`compute_inference_engine_env_vars`, `miles/ray/specs/inference.py:424`), so only an engine
Miles does not launch (`--rollout-external-engine-addrs`) can still have it set. The second is
`--recompute-logprobs-via-prefill`, whose input-logprob path is a raw `log_softmax`. At
`--rollout-temperature != 1` both make the mismatch metrics below partly a temperature
artifact. With `--use-rollout-logprobs`, the loss uses $\ell^\mu$ wherever this page writes
$\ell^{\text{old}}$.

The per-sample reducer that every loss metric goes through is

$$
\mathcal S(x) = \sum_{i \in \text{micro-batch}} \frac{\sum_t m_{i,t}\, x_{i,t}}{\max(D_i, 1)}
$$

(`get_sum_of_sample_mean`, `miles/backends/training_utils/data/context_parallel.py:247`). Under context
parallelism each CP rank sums its own zig-zag chunk with the full-sample denominator, so
the chunks add up to the exact sample mean.

### `train/*`

Each micro-batch returns `[count, S(x_1), S(x_2), ...]`
(`loss_function`, `miles/backends/training_utils/loss/objective.py:238`). `aggregate_train_losses`
(`miles/backends/training_utils/metrics/log_utils.py:446`) sums these over micro-batches, all-reduces
over the DP×CP group, and divides:

| Backend / mode | Divisor | What `train/X` is |
|---|---|---|
| Megatron, default | `num_rollouts` of this optimizer step, global across DP (`model.py:696`) | mean over rollouts of the per-rollout token-mean of $x$ |
| FSDP, torchtitan | all-reduced sample count / CP size (`training_utils/torch_native/actor.py:269`) | mean over samples of the per-sample token-mean |
| `--calculate-per-token-loss` | all-reduced token count (a fully masked sample counts as 1 token) / CP size | token-weighted mean over every trained token |

Pitfalls that follow from this:

- **Fully masked samples dilute every `train/*` mean.** A sample whose loss mask is all
  zeros contributes 0 to the numerator but still counts in the divisor, so a batch in which
  20% of samples were removed reports a `train/pg_clipfrac` 20% lower than the clip rate
  on the tokens that were actually trained.
- **Metrics that are not token means can exceed 1 with sibling samples.** `ess_ratio` and
  `opsm_clipfrac` are accumulated per sample but divided by the rollout count, so a
  compaction run with several samples per rollout can report values above 1.
- **Everything except the TIS/MIS metrics uses the post-rejection mask.** When a
  `--custom-tis-function-path` returns modified masks (rejection sampling), the reducer is
  rebuilt with them (`losses.py:269`) before `pg_clipfrac`, `ppo_kl`, `entropy_loss`,
  `kl_loss` and the mismatch metrics are reduced. Only `ois`, `tis*` and `mis_*` use the
  original mask (`losses.py:243`, `:390`).
- `train/grad_norm` and `train/lr-pg_*` are not reduced; they are the values on the
  logging rank.

### Train-side `rollout/*`

`log_rollout_data` (`miles/backends/training_utils/metrics/log_utils.py:169`) runs on TP rank 0 of
the last PP stage, builds a `(sum, count)` pair per key, gathers the pairs over the DP×CP
group and reports $\sum \text{sum} / \sum \text{count}$
(`reduce_gathered_log_dict`, `:33`). The pair depends on the key type:

| Key type | Per-rank `sum` | Per-rank `count` | Result |
|---|---|---|---|
| Token keys: `log_probs`, `ref_log_probs`, `rollout_log_probs`, `advantages`, `returns`, `values`, `teacher_log_probs`, `opd_reverse_kl`, `entropy` (`:226`) | CP size × $\mathcal S(x)$ over the rank's samples | rollouts / DP size (rollout-side schedule) or local sample count | mean over rollouts of the per-rollout token-mean |
| Other tensor lists (`ref_entropy`, `teacher_entropy`) | CP size × count × unmasked token mean | local sample count | see the known issue below |
| Numeric lists: `rewards`, `raw_reward`, `truncated`, `response_lengths`, `total_lengths`, `round_number` | plain sum | list length | mean over samples |

`raw_reward` is shipped to every DP rank unsplit
(`miles/ray/rollout/train_data_conversion.py:499-510`), so each rank reports the full-batch
mean and the reduction still returns it exactly.

### `multi_turn/*`

`log_multi_turn_data` (`miles/backends/training_utils/metrics/log_utils.py:367`) computes scalars
per rank and reduces `_max`/`_min` keys as extrema and everything else as the unweighted
mean of rank means (`_MULTI_TURN_REDUCTION_BY_KEY`, `:23`). With uneven shard sizes the
mean is not the batch mean.

### `perf/*`

Timers are a per-process singleton (`miles/utils/timer.py:15`). The train-side perf log
reports **one rank's timers only** (the perf logger's rank above) and resets every rank's
(`perf.py:21`). A straggler rank does not appear in `perf/*`; look at the dashboard's
per-rank phase lanes instead.

### Rollout-side `rollout/*`

These are computed on the list of `Sample` objects after the rollout function returned
**and after its filters ran** (`rollout_executor.py:219`). A group rejected by
`--dynamic-sampling-filter-path`, the aborted-group filter or the missing-reward filter
never reaches these metrics; it only shows up as a `rollout/dynamic_filter/drop_*` count
and, for the dynamic filter, in `rollout/raw_reward_unfiltered`.
The batch is also already trimmed to a multiple of `global_batch_size`
(`rollout_data_conversion.py:33`).

## The metric glossary

Each entry gives the key, the exact formula, where it is computed and how to read it.
"(flag)" marks keys that only exist under that flag. Formulas use the notation above;
$\mathcal R$ is the reduction for the key's section.

### Reward, advantage and batch composition

Rollout side:

| Key | Formula | Source | Reading |
|---|---|---|---|
| `rollout/episode_raw_reward` | mean over rollouts of (mean raw reward over the rollout's siblings); raw reward is `metadata["raw_reward"]` if present, else `get_reward_value` | `metrics.py:249` | The reward to watch in compaction and multi-sample runs: one vote per trajectory. Equals `rollout/raw_reward` in ordinary runs. |
| `rollout/raw_reward_unfiltered` | mean over samples of every finished group, taken before the dynamic filter whether the group is kept or dropped; `None` and un-keyed dict rewards skipped | `filter_hub/base_types.py:45`, `:67` | The pre-filter reward. Sync runs feed it before the aborted and missing-reward filters (oversampled extras included); fully-async runs feed it after them. |
| `rollout/num_training_samples` | number of samples after filtering and trimming | `metrics.py:268` | Should equal `rollout_batch_size × n_samples_per_prompt` in sync runs; more in compaction runs. |
| `rollout/zero_std/count_<r>` | number of groups (by `group_index`) whose rewards are all equal, keyed by `str(round(reward, 1))` | `metrics.py:306` | Computed on the **kept** batch. With a zero-std dynamic filter this is 0 by construction. |
| `rollout/zero_std/all_zero_percentage`, `all_one_percentage` | fraction of groups whose uniform reward `== 0` / `== 1` | `metrics.py:329` | "Too hard" and "too easy" fractions of the kept batch, so also 0 under a zero-std filter. The comparison is numeric, so integer and float rewards both count (integer rewards still produce the keys `count_0` and `count_1`); rewards other than 0 and 1 never count. Not emitted for `--advantage-estimator ppo`. |
| `rollout/dynamic_filter/drop_<reason>` | number of groups dropped this rollout, by filter reason | `filter_hub/base_types.py:61` | The pre-filter view the kept-batch metrics lack. Reasons: `group_has_aborted`, `group_has_missing_reward`, `zero_std_<r>` from `apply_reward_nonzero_std_filter`, or a custom filter's `reason`. A filter that returns a bare bool has no reason and is not counted. In fully-async runs aborted groups are counted in `rollout/fully_async/aborted_groups_filtered` instead. |
| `rollout/error_cat/<category>` (`--log-reward-category KEY`) | fraction of samples whose `reward[KEY]` equals the category | `metrics.py:368` | Failure taxonomy from your reward function. |
| `passrate/pass@k` (`--log-passrate`) | for every complete group of size $n$, the unbiased estimator $1 - \binom{n-c}{k}/\binom{n}{k}$ with $c$ = number of rewards **exactly equal to 1**, averaged over groups, for $k = 1, 2, 4, \ldots \le n$ | `metrics.py:378`, `metric_utils.py:40` | Kept batch only. `pass@1` is the mean solve rate of the kept groups. Meaningless for rewards that are never exactly 1. |

Train side (reduction: mean over samples for scalar lists, mean over rollouts for token keys):

| Key | Formula | Source | Reading |
|---|---|---|---|
| `rollout/raw_reward` | $\frac1N \sum_i R_i$ | `log_utils.py:259` | The headline reward. Samples, not rollouts: a rollout split into 3 samples gets 3 votes. |
| `rollout/rewards` | $\frac1N \sum_i \tilde R_i$, where for `grpo`/`gspo`/`reinforce_plus_plus_baseline` with reward normalization on, $\tilde R_i = R_i - \bar R_g$, divided by $\sigma_g + 10^{-6}$ for `grpo`/`gspo` (unless `--disable-grpo-std-normalization`) when $\sigma_g > 0$ and the group has more than one rollout. $\sigma_g$ is the **unbiased** (Bessel-corrected) std over the group's rollouts | `train_data_conversion.py:294`, `:344` | About 0 by construction for GRPO. Not a learning signal; a non-zero value means incomplete groups (trimming) or `--disable-rewards-normalization`. |
| `rollout/advantages` | $\mathcal R(A)$. GRPO/GSPO: $A_{i,t} = \tilde R_i$ broadcast (`advantages.py:53`). PPO: GAE over token rewards $-\beta_{\text{kl}} \cdot \text{kl}$ plus the terminal reward. REINFORCE++: discounted return. With `--normalize-advantages`, whitened over the DP group with the token mask (`advantages.py:108`) | `objective.py:100` | Mean near 0 is expected. Miles logs only the mean; there is no spread or histogram. |
| `rollout/returns` | $\mathcal R(G)$; equal to the un-whitened advantages for GRPO and REINFORCE++ | `objective.py:125` | Only interesting for PPO. |
| `rollout/values` (critic) | $\mathcal R(V)$ from the critic's forward pass, shipped to the actor | `megatron_utils/actor.py:750` | Compare with `rollout/returns`. |

### Lengths, truncation and multi-turn

| Key | Formula | Source | Reading |
|---|---|---|---|
| `rollout/response_len/{mean,median,max,min}` | stats over samples of `effective_response_length` $= \sum_t m_{i,t}$ | `metrics.py:110`, `types.py:343` | Model-written tokens only. |
| `rollout/episode_response_length/{mean,median,max,min}` | stats over rollouts of $\sum_{\text{siblings}} \sum_t m$, with removed samples counted as 0 | `metrics.py:168` | Trainable tokens per trajectory. |
| `rollout/episode_total_response_length/mean` | mean over rollouts of $\sum_{\text{siblings}}$ `response_length` | `metrics.py:193` | Includes tool and observation tokens. |
| `rollout/truncated_ratio` | fraction of samples with `status == TRUNCATED` | `metrics.py:116` | Budget pressure. The dashboard flags ≥ 25%. |
| `rollout/truncated` | same fraction on the train side | `log_utils.py:259` | Duplicate of the above. |
| `rollout/response_lengths`, `rollout/total_lengths` | mean over samples of `response_length` and of prompt + response length | `log_utils.py:259` | Full lengths, including tool output. |
| `rollout/repetition_frac` | fraction of samples whose response text has a 10,000-character window with zlib compression ratio > 10 | `metrics.py:115`, `metric_utils.py:155` | Degenerate looping. **Blind to responses under 10,000 characters** (roughly 2.5k tokens). |
| `rollout/round_number`, `multi_turn/multi_turn_metric/round_number_{mean,max,min}` (`--log-multi-turn`) | turn count from `metadata["round_number"]` | `log_utils.py:398` | Only for rollout functions that set it. |
| `multi_turn/raw_response_length/response_length_{mean,max,min}` (`--log-multi-turn`) | stats of full response length | `log_utils.py:383` | Rank-level stats, then reduced as described above. |
| `multi_turn/raw_response_length/response_length_clip_ratio` (`--log-multi-turn`) | fraction with response length ≥ `--rollout-max-response-len` | `log_utils.py:387` | Hitting the context budget, as opposed to a per-turn stop. |
| `multi_turn/wo_obs_response_length/response_length_{mean,max,min}` (`--log-multi-turn`) | stats of $\sum_t m_{i,t}$ | `log_utils.py:395` | Same as `response_len/*`, per rank. |

### KL to the reference model

The reference model is loaded only when `--kl-coef != 0` or `--use-kl-loss` is set
(`miles/ray/specs/train.py:59`). Without it none of these keys exist.

The four estimators, for $\Delta = \ell^a - \ell^b$ (`compute_approx_kl`,
`miles/backends/training_utils/loss/hub/math_utils.py:139`):

$$
\texttt{k1} = \Delta,\qquad \texttt{k2} = \tfrac12 \Delta^2,\qquad \texttt{k3} = e^{-\Delta} - 1 + \Delta,\qquad \texttt{low\_var\_kl} = \operatorname{clamp}\!\big(e^{-\Delta'} - 1 + \Delta',\,-10,\,10\big),\ \Delta' = \operatorname{clamp}(\Delta, -20, 20)
$$

| Key | Formula | Source | Reading |
|---|---|---|---|
| `train/kl_loss` (`--use-kl-loss`) | $\mathcal R\big(\texttt{kl}(\ell^\theta, \ell^{\text{ref}})\big)$ with the estimator from `--kl-loss-type` (default **k1**, signed); with `--use-unbiased-kl` each token is multiplied by $e^{\ell^\theta - \ell^{\text{old}}}$ | `losses.py:318` | Odometer of drift from the reference. Added to the loss as `kl_loss_coef × kl_loss`; with `--kl-loss-coef 0` it is monitor-only. The CI checker asserts it is 0 at step 0 (skipped under `--use-rollout-routing-replay`). k1 can go negative; prefer `low_var_kl` for monitoring. |
| `rollout/ref_log_probs` | $\mathcal R(\ell^{\text{ref}})$ | `log_utils.py:226` | `rollout/log_probs − rollout/ref_log_probs` is a signed k1 estimate of KL(old ‖ ref), available even with `--kl-loss-coef 0`. |

`--kl-coef` is a reward-shaping penalty that only the REINFORCE++ variants consume
(`advantages.py:79`, `:92`). The PPO branch (`advantages.py:59`) would use it too, but PPO
asserts `--kl-coef 0` (`arguments.py:3755`), and GRPO/GSPO ignore it. It has no dedicated metric; it shows up inside
`rollout/advantages` and `rollout/returns`.

### Policy ratio, clipping and trust region

With the ratio $r_{i,t} = \exp(\operatorname{clamp}(\ell^\theta - \ell^{\text{old}}, -20, 20))$
(`compute_policy_loss`, `math_utils.py:254`) the surrogate is

$$
L^{(1)} = -A\,r,\qquad L^{(2)} = -A\,\operatorname{clip}(r, 1-\epsilon, 1+\epsilon_h),\qquad
L = \max(L^{(1)}, L^{(2)}),\qquad \text{dual-clip: } L \leftarrow \min(L, -\epsilon_c A) \text{ where } A < 0
$$

with $\epsilon$ = `--eps-clip` (0.2), $\epsilon_h$ = `--eps-clip-high` (defaults to
`--eps-clip`), $\epsilon_c$ = `--eps-clip-c`.

| Key | Formula | Source | Reading |
|---|---|---|---|
| `train/ppo_kl` | $\mathcal R(\ell^{\text{old}} - \ell^\theta)$; GSPO replaces the per-token value with the sequence mean | `losses.py:189` | Signed k1 estimate of KL(old ‖ θ). **0 on the first optimizer step of a rollout** (up to kernel nondeterminism) unless the anchor is $\ell^\mu$ or `--keep-old-actor` makes it an older weight version; the CI checker asserts $< 10^{-9}$ there (MLA models exempt from the abs check, LoRA relaxed to $10^{-8}$; `checks.py:17-23`). With several steps per rollout, it measures how far later steps have moved. |
| `train/pg_clipfrac` | $\mathcal R(\mathbb 1[L^{(2)} > L^{(1)}])$: tokens with $A>0, r>1+\epsilon_h$ or $A<0, r<1-\epsilon$ | `math_utils.py:265` | Same caveat: 0 on the first step unless `--use-rollout-logprobs`. Dual-clip activations are not counted. A few percent on later steps is normal. |
| `train/ess_ratio` | per sample, with $w = e^{\ell^\theta - \ell^{\text{old}}}$ over its $n$ masked tokens, $\text{ESS} = (\sum w)^2 / (n \sum w^2) \in [1/n, 1]$; averaged over samples (token-weighted under per-token loss) | `math_utils.py:35` | 1 when every token has the same ratio, which includes but is not limited to on-policy. A falling ESS means a few tokens dominate the update. |
| `train/ois` (`--use-tis` or `--get-mismatch-metrics`) | $\mathcal R(e^{\ell^\theta - \ell^{\text{old}}})$ on the pre-rejection mask | `losses.py:252`, `:390` | Mean on-policy ratio; about 1. |
| `train/opsm_clipfrac` (`--use-opsm`) | $\sum_i$ fraction of sample $i$'s tokens masked, where a sample is masked when $A<0$ and its sequence-mean $\ell^{\text{old}} - \ell^\theta >$ `--opsm-delta` | `math_utils.py:183` | Fraction of negative-advantage sequences dropped as off-policy. |
| `train/pg_loss` | $\mathcal R(L \cdot w_{\text{IS}})$, or `--custom-pg-loss-reducer-function-path`'s reduction | `losses.py:303` | Level is not meaningful; it scales with the advantage. |
| `train/loss` | `pg_loss − entropy_coef × entropy_loss + kl_loss_coef × kl_loss`, before Megatron's micro-batch and DP rescaling | `losses.py:314`, `:338` | What the optimizer minimizes, per rollout. |

### Importance-sampling corrections

The built-in correction, used with `--use-tis` when no custom function is given
(`vanilla_tis_function`, `miles/backends/training_utils/loss/hub/corrections.py:7`), has
$w = e^{\ell^{\text{old}} - \ell^\mu}$ and multiplies the per-token loss by
$\operatorname{clamp}(w, \texttt{tis\_clip\_low}, \texttt{tis\_clip})$ (defaults 0 and 2).

| Key | Formula | Source | Reading |
|---|---|---|---|
| `train/tis` | $\mathcal R(w)$, before clamping | `corrections.py:27` | Mean old-vs-rollout ratio. About 1. |
| `train/tis_clipfrac` | $\mathcal R(\mathbb 1[\operatorname{clamp}(w) \ne w])$ | `corrections.py:25` | Fraction of tokens whose weight was capped. |
| `train/tis_abs` | $\mathcal R(\lvert w - 1\rvert)$ | `corrections.py:23` | Symmetric mismatch magnitude, in probability-ratio units. |

`icepop_function` (`corrections.py:35`) reports the same three keys but zeroes out-of-range
tokens instead of clamping them. `examples/infra_features/train_infer_mismatch_helper/mis.py`
reports `train/mis_*` on top; its keys and formulas are on the
[Rollout Correction Methods](/examples/infra-features/train-infer-mismatch-helper#mismatch-metrics)
page. Two exactness notes for those: every `mis_*` value is built per sample
(for example `mis_training_ppl` is the mean over samples of $\exp(-\text{sample-mean } \ell^{\text{old}})$,
not the perplexity of the batch), and all of them are reduced with the pre-rejection mask.

### Train–inference mismatch

These two keys exist whenever the batch carries `rollout_log_probs`, which is every RL run
that records SGLang log-probs. They are the direct gauge of the gap between what SGLang
sampled from and what the trainer thinks the policy is.

| Key | Formula | Source | Reading |
|---|---|---|---|
| `train/train_rollout_logprob_abs_diff` | $\mathcal R(\lvert \ell^{\text{old}} - \ell^\mu \rvert)$ | `losses.py:344` | Mean absolute per-token log-prob gap, in nats. |
| `train/train_rollout_kl` | $\mathcal R\big(\operatorname{clamp}(\rho - 1 - \log\rho, -10, 10)\big)$ with $\log \rho = \operatorname{clamp}(\ell^{\text{old}} - \ell^\mu, -20, 20)$ | `losses.py:362` | k3 estimate of KL(rollout ‖ trainer) at the sampled tokens. Non-negative. |
| `rollout/rollout_log_probs`, `rollout/log_probs` | $\mathcal R(\ell^\mu)$, $\mathcal R(\ell^{\text{old}})$ | `log_utils.py:226` | Their difference is a signed version of the gap. Under `--ci-test` its absolute value must stay below 0.03 on every rollout (`log_utils.py:290`). |

$\ell^{\text{old}}$ here is the trainer's forward-only pass. When that pass is skipped
(`--use-rollout-logprobs` without `--get-mismatch-metrics`, or `--skip-actor-forward-only`),
the current-policy $\ell^\theta$ is used instead, so the metric then also absorbs how far
the optimizer has moved within the rollout.

How to read them:

- In a **synchronous** run the trainer and SGLang hold the same weights, so the gap is
  pure numerics: kernels, precision, MoE routing, the attention backend. It should be flat
  across training. `--true-on-policy-mode` makes it exactly 0 and the CI checker asserts
  so. `--use-rollout-routing-replay` narrows it for MoE.
- In **`train_async.py` and fully-async runs** the gap also includes weight staleness. By
  default the PPO anchor $\ell^{\text{old}}$ is recomputed with the current weights, so
  `train/ppo_kl` and `train/pg_clipfrac` say nothing about staleness; these two metrics are
  the only train-side signal of it (the argument validator's docstring at
  `miles/utils/arguments.py:4258` makes the same point for PPO). Under
  `--pause-generation-mode in_place` they also absorb tokens sampled under new weights over
  KV (and radix-cache prefixes) computed with the old ones. Only
  `--sglang-enable-prefill-weight-versions` stamps that prompt KV (the
  `rollout/weight_version/prefill_*` keys below); without it the effect shows up here and
  nowhere else ([Async and Off-Policy, 3.3](/deep-dive/async-and-off-policy)).
- A step change means something structural changed: a weight sync that did not land, a
  differing backend flag, a quantization path. Check `--check-weight-update-equal`.

### Entropy

With $z$ the temperature-scaled logits over the full vocabulary and $p = \operatorname{softmax}(z)$,
$H_{i,t} = \log \sum_v e^{z_v} - \sum_v p_v z_v$ (`compute_entropy_from_logits`,
`math_utils.py:458`). It is the entropy of the full softmax, not of the top-p/top-k
truncated sampling distribution.

| Key | Formula | Source | Reading |
|---|---|---|---|
| `rollout/entropy` | $\mathcal R(H)$ from the forward-only pass under $\ell^{\text{old}}$ | `log_utils.py:226` | Megatron: only with `--use-rollout-entropy` (`model.py:374`). FSDP and torchtitan: always (`training_utils/torch_native/actor.py:223`). Under `--ci-test` it must lie in (0, 0.7) on every rollout (`log_utils.py:294`). |
| `train/entropy_loss` | $\mathcal R(H)$ under $\ell^\theta$ inside the loss | `losses.py:307` | **Always present but identically 0** unless `--entropy-coef != 0` or `--observe-training-entropy`. |
| `rollout/ref_entropy`, `rollout/teacher_entropy` | produced by the ref and teacher passes when `--use-rollout-entropy` is on | `log_utils.py:249` | Not comparable to `rollout/entropy`; see the known issues. |
| `rollout/log_probs` | $\mathcal R(\ell^{\text{old}})$ | `log_utils.py:226` | $-$`rollout/log_probs` is the mean sampled-token NLL, a free entropy proxy when entropy is off. |

### Gradient norm, learning rate, optimizer

| Key | Formula | Source | Reading |
|---|---|---|---|
| `train/grad_norm` | Megatron: the global L2 norm returned by `optimizer.step()`, before clipping to `--clip-grad`; 0.0 with `--debug-disable-optimizer`, and 0.0 when a step is skipped for inf/NaN under `--no-check-for-nan-in-loss-and-grad` (`model.py:638`). FSDP: `clip_grad_norm_` total norm before clipping (`fsdp_utils/actor.py:350`) | `log_utils.py:536` | Primary stability gauge. |
| `train/lr-pg_<i>` | LR of optimizer param group $i$, read **after** the scheduler step | `model.py:868`, `fsdp_utils/actor.py:355` | The LR the next step will use. Megatron advances the scheduler by `num_rollouts` per step (`model.py:663`), so warmup and decay are in samples. |
| `train/mtp_loss` (`--enable-mtp-training`) | Megatron's MTP tracker value, scaled by `1/num_microbatches` unless per-token loss | `model.py:838` | Draft-head loss. |

### Staleness and async

| Key | Formula | Source | Reading |
|---|---|---|---|
| `rollout/weight_version/{mean,median,max,min}` | stats over samples of the oldest numeric weight version stamped on any of its generation calls | `metrics.py:197`, `types.py:152` | **Absolute** version, not lag. Megatron publishes the number of optimizer steps its weights have taken (`ModelCompanionWeightVersionUtils`, `training_utils/model_companion.py:147`, bumped at `model.py:662`; 0 at a fresh run's first sync), so with one optimizer step per rollout rollout $k$ reads $k$, and with $S$ steps it reads $kS$. FSDP and torchtitan count syncs from 1 (`training_utils/torch_native/actor.py:295`), so rollout $k$ reads about $k+1$ there. |
| `rollout/weight_version/mixed_version_ratio` | fraction of samples whose spans carry more than one version | `metrics.py:213` | Weights changed mid-trajectory (partial rollout, fully async). The pinned SGLang reports per-token spans even within one generate call, so this is exact on that build ([Async and Off-Policy, 3.1](/deep-dive/async-and-off-policy)). |
| `rollout/weight_version/prefill_{mean,median,max,min}`, `prefill_mixed_version_ratio` (`--sglang-enable-prefill-weight-versions`) | the same two statistics over the versions stamped on each call's prompt KV (computed or reused from the radix cache) | `metrics.py:123` | Which weights produced the prompt KV that decoding attended to. |
| `rollout/weight_version/prefill_stale_token_ratio`, `prefill_lag_max` (same flag) | over calls with numeric stamps: prompt tokens whose KV version is below the call's newest decode version / all prompt tokens; max over calls of newest decode version − oldest prompt-KV version | `metrics.py:218` | Nonzero means decoding under new weights over old-weight KV, which `--pause-generation-mode in_place` allows. `--ci-assert-prefill-lag-max` bounds it under `--ci-test`. |
| `rollout/fully_async/*` (`--fully-async`) | queue size, filtered counts and staleness in weight versions, computed in `DefaultDataBuffer.get_metrics` | `fully_async_data_buffer.py:287` | Definitions on [Fully Async RL](/user-guide/fully-async#metrics). Windowed: counters reset each training step. Under Megatron a version is an optimizer step, per the first row. |

### Timing, throughput and MFU

Train side (the perf logger's rank; `miles/backends/training_utils/metrics/perf.py:14`). Every
timer `X` becomes `perf/X_time`, accumulated over the rollout:

| Timer | Covers | Source |
|---|---|---|
| `train_wait` | from the end of the previous `train()` to the start of this one: rollout generation in sync runs, plus save, offload, weight update and data fetch | `megatron_utils/actor.py:114`, `:695` |
| `train` | the whole `train()` body after data fetch | `megatron_utils/actor.py:695` |
| `data_preprocess` | fetching and tensorizing the rollout shard (Megatron) | `megatron_utils/actor.py:593` |
| `log_probs`, `ref_log_probs`, `teacher_log_probs` | forward-only passes | `megatron_utils/actor.py:560` |
| `actor_train`, `critic_train` | forward-backward and optimizer steps | `megatron_utils/actor.py:769`, `:603` |
| `update_weights`, `update_weights_implementation`, `finalize_and_resume_engines` | the whole weight sync; streaming the weight buckets to the engines; finalizing, ending the engines' update session, stamping the version and resuming generation | `megatron_utils/actor.py:918`, `weight_update/updater.py:145`, `:161` |
| `sleep`, `wake_up`, `offload_grad_buffer` | colocate offload and onload | `megatron_utils/actor.py:446`, `:494`, `:481` |
| `save_model`, `export_hf`, `ref_model_update` | checkpointing and reference refresh | `megatron_utils/actor.py:834`, `:874`, `:804` |

Derived keys:

$$
\texttt{perf/step\_time} = t_{\text{train\_wait}} + t_{\text{train}},\qquad
\texttt{perf/wait\_time\_ratio} = \frac{t_{\text{train\_wait}}}{\texttt{perf/step\_time}}
$$

$$
F = \frac{\texttt{calculate\_fwd\_flops}(\text{all total lengths of the batch})}{W \cdot 10^{12}},\qquad
\texttt{perf/actor\_train\_tflops} = \frac{3F}{t_{\text{actor\_train}}},\qquad
\texttt{perf/log\_probs\_tflops} = \frac{F}{t_{\text{log\_probs}}}
$$

and `perf/ref_log_probs_tflops` likewise.

$$
\texttt{perf/actor\_train\_mfu} = \frac{\texttt{perf/actor\_train\_tflops}}{\texttt{perf/mfu\_peak\_tflops}},\qquad
\texttt{perf/actor\_train\_tok\_per\_s} = \frac{\sum \text{total lengths}}{t_{\text{actor\_train}}}
$$

$W$ is the training world size, so `tflops` is **per GPU**, while `tok_per_s` is **for
the whole training world**. The MFU is a fraction (0 to 1), not a percentage; the peak
comes from `--mfu-peak-tflops` or the detected device (`device_flops.py:42`). The FLOPs
model (`miles/utils/flops_utils.py:69`) counts QKV, causal attention, output projection,
SwiGLU MLP (top-k experts plus shared experts for MoE) and the LM head, and ignores
activation recomputation, so a run with full recompute under-reports its real hardware
utilization by roughly a third.

Other train-side perf keys: `perf/cpu_memory_after_offload_train_gb` (host memory after
offload, `log_utils.py:428`), and `perf/update_weights_density` /
`perf/update_weights_wire_bytes` under `--update-weight-transfer-mode disk-delta`
(`weight_update/protocols/delta.py:433`).

<Warning>
`update_weights` (with its `update_weights_implementation` and
`finalize_and_resume_engines` parts), `sleep`, `save_model` and the disk-delta keys are
measured after the rollout's perf log was written, so they land on the **next**
`rollout/step`. The `train_wait` of rollout $k$ likewise contains the weight update that
followed rollout $k-1$.
</Warning>

Rollout side (`_compute_perf_metrics_from_samples`, `miles/ray/rollout/metrics.py:273`),
with $T$ = `perf/rollout_time`:

| Key | Formula | Reading |
|---|---|---|
| `perf/rollout_time` | wall-clock of `RolloutExecutor._generate_rollout_data` up to logging, including filtering, trimming, reward and the debug dump (`rollout_executor.py:197`) | In fully-async runs this is the time spent draining the buffer, not generation time; when the buffer already holds a batch it is close to 0, which inflates every rate below. |
| `perf/tokens_per_gpu_per_sec` | $\sum_i \lvert y_i\rvert \,/\, T \,/$ `--rollout-num-gpus` | Full response lengths, tool output included. Meaningless in fully-async runs. |
| `perf/effective_tokens_per_gpu_per_sec` | same with $\sum_t m_{i,t}$ | Model-written tokens only. |
| `perf/longest_sample_tokens_per_sec`, `perf/longest_effective_sample_tokens_per_sec` | $\max_i \lvert y_i\rvert / T$ | Decode speed of the straggler. |
| `perf/non_generation_time/{mean,median,max,min}` | stats of `Sample.non_generation_time` (the agent's `total_tool_time`) | Only when some sample reports it. |
| `perf/longest_sample_non_generation_time`, `perf/longest_sample_tokens_per_sec_without_non_generation` (and `effective_` variants) | tool time of the longest samples, and $\max \lvert y\rvert / (T - \text{that})$ | Separates slow tools from slow decode. Goes negative when that tool time exceeds $T$, as it routinely does in fully-async runs. |

### Rollout engine and data quality

| Key | Formula | Source | Reading |
|---|---|---|---|
| `rollout/prefix_cache_hit_rate` | $\sum$ cached prompt tokens $/ \sum$ prompt tokens over every generation call of every sample | `metrics.py:357` | Multi-turn runs should be high; a drop means the radix cache is being evicted. |
| `rollout/avg_cached_tokens_per_sample` | $\sum$ cached tokens $/ N$ | `metrics.py:364` | |
| `rollout/spec_accept_rate`, `rollout/spec_accept_length` (speculative decoding) | accepted drafts / proposed drafts; completion tokens / verify steps | `metrics.py:335` | Session-server runs count each session once. |
| `rollout/tito_session_mismatch_rate/{v1,v2}` and `/special_token_count`, `/special_token_type`, `/non_assistant_text`, `/assistant_text` (session server) | fraction of samples with any token-in/token-out mismatch, then by mismatch type | `metrics.py:132` | Chat-template or TITO bug detector. Under `--ci-test` the first three have hard thresholds. |

### Eval

`log_eval_rollout_data` (`miles/ray/rollout/metrics.py:25`), per eval dataset `<name>`:

| Key | Formula | Reading |
|---|---|---|
| `eval/<name>` | mean reward over eval samples, with `None` rewards counted as 0 | The held-out score. |
| `eval/<name>-none_reward_ratio` | fraction of samples whose reward is `None` | Errored or aborted trials inflate failures silently; watch this. |
| `eval/<name>-truncated_ratio` | fraction truncated | |
| `eval/<name>/<key>` | every rollout-side metric above, computed on the eval samples (`eval/<name>/response_len/mean`, `eval/<name>/repetition_frac`, and so on) | `eval/<name>/num_training_samples` is just the eval sample count. |
| `eval/<name>-pass@k` (`--log-passrate`) | the pass@k estimator on the **flat** reward list reshaped into groups of `--n-samples-per-eval-prompt` | Assumes each prompt's samples are contiguous. |
| `eval/lag_steps`, `eval/duration_seconds`, `eval/export_time_seconds` (snapshot eval) | how many rollouts training has advanced since the snapshot; eval and export wall-clock | `rollout_executor.py:282` |
| `eval/skipped_<reason>` | 1 at a skipped eval point | Explains gaps in eval curves. |

### Critic, distillation and SFT

| Key | Formula | Source |
|---|---|---|
| `train/critic-value_loss` | $\mathcal R\big(\max((V^{\text{clip}} - G)^2, (V - G)^2)\big)$, $V^{\text{clip}} = V_{\text{old}} + \operatorname{clamp}(V - V_{\text{old}}, \pm$ `--value-clip`$)$ | `losses.py:430` |
| `train/critic-value_clipfrac` | $\mathcal R(\mathbb 1[\lvert V - V_{\text{old}}\rvert >$ `value_clip`$])$ | `losses.py:444` |
| `train/critic-grad_norm`, `train/critic-lr-pg_<i>` | as for the actor | `log_utils.py:530` |
| `rollout/opd_reverse_kl`, `train/opd_reverse_kl` (`--use-opd`) | $\mathcal R(\ell^{\text{old}}_{\text{student}} - \ell^{\text{teacher}})$, the per-token term subtracted from advantages with weight `--opd-kl-coef` | `loss/hub/opd.py:93`, `losses.py:400` |
| `rollout/teacher_log_probs` (`--use-opd`) | $\mathcal R(\ell^{\text{teacher}})$ | `log_utils.py:226` |
| `train/loss` (SFT) | $-\mathcal R(\ell^\theta)$ | `losses.py:502` |
| `train/sc_*`, `train/kl_invalid_fraction` (`--loss-type score_centering`) | defined on [Score Centering](/examples/infra-features/score-centering); this loss also emits `train_rollout_kl` and `train_rollout_logprob_abs_diff` but no `pg_clipfrac`, `ppo_kl` or `ess_ratio` | `loss/hub/score_centering_loss.py:157` |

### Engine telemetry and the dashboard

- `--sglang-enable-metrics` makes the secondary wandb writers scrape the router's
  `/engine_metrics` Prometheus endpoint into wandb system metrics named `sgl_engine.*`
  (`wandb_utils.py:136`). They plot against wall time, not a step key.
- `--use-miles-dashboard` (requires `--dump-details`) records phase intervals from every
  rank's timers, NVML GPU utilization and a whitelisted set of SGLang series
  (`sglang_num_running_reqs`, `sglang_token_usage`, `sglang_cache_hit_rate`, request and
  abort counters, TTFT and more; `dashboard/sglang_scraper.py:42`) under
  `<dump-details>/dashboard/`.
- When a dump has no metrics stream, the dashboard derives `dump/*` series from the dumped
  samples: `reward_mean`, `reward_std`, `response_length_mean`, `truncated_frac`,
  `zero_std_group_frac`, `mean_abs_lp_diff`, `mean_entropy`, `mixed_version_frac`
  (`dashboard/dump_reader.py:43`). These are **per-sample means over the dumped batch**,
  so they do not match the rollout-weighted `train/*` values exactly.
- The dashboard's advisory panel (`dashboard/advisory.py`) raises alarms for an open phase
  more than 6× its median duration and at least 30 minutes old, more than 5% of engine
  requests aborted in the last 30 minutes, at least half the groups zero-std, at least
  25% of samples truncated, and MFU below 0.15, plus engine-tuning hints (KV-bound engines,
  low concurrency, low cache hit rate, DP imbalance; `_tuning_advisories`, `:338`).

## The vital-signs dashboard

Pin these panels. Each row names the Miles key to use and the one it is most often
confused with.

| Vital sign | Use | Not | Why |
|---|---|---|---|
| Reward | `rollout/raw_reward` (or `rollout/episode_raw_reward` with compaction), `eval/<name>` | `rollout/rewards` | `rollout/rewards` is the group-normalized reward and sits at 0. |
| Usable signal | `rollout/dynamic_filter/drop_zero_std_*` with the zero-std filter on, `rollout/zero_std/all_*_percentage` with it off; `rollout/raw_reward_unfiltered` either way | `rollout/advantages` | The mean advantage is 0 by construction; Miles does not log its spread. What kills the gradient is zero-std groups, and `zero_std/*` reads 0 by construction once the filter has removed them. |
| Drift from reference | `train/kl_loss` (needs `--use-kl-loss`; set `--kl-loss-coef 0` for monitor-only) | `train/ppo_kl` | `ppo_kl` is drift within a rollout, not from the reference. |
| Train–inference gap | `train/train_rollout_logprob_abs_diff`, `train/train_rollout_kl` | `train/pg_clipfrac` | On the first step of a rollout the clip fraction is 0 whatever the gap. |
| Trust region | `train/pg_clipfrac`, `train/ess_ratio`, `train/ppo_kl` | | Only informative with `--use-rollout-logprobs` or more than one optimizer step per rollout. |
| Exploration | `rollout/entropy` (Megatron: `--use-rollout-entropy`) or `train/entropy_loss` (`--observe-training-entropy`) | | Both are off by default on Megatron; turn one on. |
| Stability | `train/grad_norm` | | Pre-clip norm; compare with `--clip-grad`. |
| Budget | `rollout/truncated_ratio`, `rollout/response_len/max` vs `--rollout-max-response-len` | | |
| Degeneration | `rollout/repetition_frac` | | Only sees long responses. |
| Balance | `perf/wait_time_ratio`, `perf/rollout_time`, `perf/actor_train_time`, `perf/update_weights_time` | `perf/rollout_time` and `perf/*tokens_per_*` in fully-async runs | High wait ratio means rollout-bound. In fully-async runs the rollout timer only measures the buffer drain; use `rollout/fully_async/queue_size`. |
| Staleness (async) | `rollout/fully_async/avg_staleness`, `stale_groups_filtered`, `rollout/weight_version/mixed_version_ratio` | `rollout/weight_version/mean` | The latter is an absolute version. |

## Diagnostic playbook

"Check" is what to put on one panel; "lever" is the Miles flag to change, one at a time.

| Symptom | Likely causes | Check | Lever |
|---|---|---|---|
| Reward flat from rollout 0 | Reward read from the wrong key; every group zero-std; LR 0; model cannot follow the format | `rollout/raw_reward` distribution in a dump, `rollout/dynamic_filter/drop_zero_std_*`, `train/lr-pg_0`, first sample printed as `First rollout sample:` | Fix `--reward-key` / `--label-key`; use `python -m miles.utils.debug_utils.replay_reward_fn` on a dump; adjust data difficulty; check `--lr` |
| First rollout is gibberish | Weights did not load, bad parameter mapping, chat-template mismatch | the rollout dump, `rollout/log_probs` vs `rollout/ref_log_probs` at rollout 0 | See [Debugging](/developer/debug#is-the-first-rollout-coherent); `--check-weight-update-equal` |
| `rollout/log_probs` ≠ `rollout/ref_log_probs` at rollout 0 | Non-deterministic attention kernel, template mismatch (about 0.8/token), R3 enabled | same two keys, `train/ppo_kl` at step 0 | `--attention-backend flash`; `--ci-test` to turn it into an assertion |
| Reward rises then collapses | Policy divergence, often amplified by a growing train–inference gap | `train/grad_norm` spike, `train/kl_loss`, `train/train_rollout_kl`, `rollout/truncated_ratio` and `rollout/repetition_frac` rising together | Lower `--lr`; `--use-kl-loss --kl-loss-coef`; `--use-tis` or MIS; roll back to a checkpoint before the rise |
| `train_rollout_logprob_abs_diff` climbs steadily | Staleness (async) or precision drift (FP8, MoE routing) | `rollout/fully_async/avg_staleness`, `rollout/weight_version/mixed_version_ratio`, MoE or not | `--max-weight-staleness`; `--use-rollout-routing-replay`; `--use-tis`; `--true-on-policy-mode` where supported |
| `train_rollout_logprob_abs_diff` step change | Weight sync not landing, differing kernels or quantization | `perf/update_weights_time` at that step, engine logs | `--check-weight-update-equal`; `--ci-inject-rollout-data-path` replay (below) |
| `train/pg_clipfrac` always 0 | Single optimizer step per rollout without `--use-rollout-logprobs`: the ratio is identically 1 | `train/ppo_kl` ≈ 0 | Expected. Use the mismatch metrics instead, or `--use-rollout-logprobs` to anchor on SGLang |
| `train/pg_clipfrac` high on later steps | LR too high, too many steps per rollout | `train/ppo_kl`, `train/ess_ratio` | Lower `--lr`; fewer `--num-steps-per-rollout`; tighten `--eps-clip` / `--eps-clip-high` |
| Most groups zero-std | Data too easy (all-one) or too hard (all-zero) for the current policy | `rollout/dynamic_filter/drop_zero_std_1.0` vs `_0.0` (`_1` / `_0` for integer rewards), `rollout/raw_reward_unfiltered`, `rollout/zero_std/*` (filter off) | `--dynamic-sampling-filter-path miles.rollout.filter_hub.common_filters.apply_reward_nonzero_std_filter` with `--over-sampling-batch-size`; raise `--n-samples-per-prompt`; rebalance data |
| Entropy collapsing | Premature convergence | `rollout/entropy`, `rollout/zero_std/all_one_percentage` rising, `passrate/pass@k` for large k flattening toward `pass@1` | Raise `--rollout-temperature`; `--entropy-coef`; `--eps-clip-high` above `--eps-clip` (clip-higher); lower `--lr` |
| High truncation | Context or per-turn budget too small, or a looping policy | `rollout/truncated_ratio`, `rollout/response_len/max`, `rollout/repetition_frac`, `multi_turn/raw_response_length/response_length_clip_ratio` | Raise `--rollout-max-response-len`; in agentic runs the per-turn cap is often the binding one; if repetition rises with it, treat as divergence |
| `train/grad_norm` spikes or NaN | Bad batch, FP8 overflow, broken template | the dump for that rollout, `train/loss` | Lower `--lr`; `--clip-grad`; replay that rollout with `--load-debug-rollout-data` |
| `train/grad_norm` reads exactly 0 on a normal run | With Megatron's `--no-check-for-nan-in-loss-and-grad`, a step with inf/NaN gradients is skipped and the initial 0.0 is logged instead of failing | stdout `train op=train_step ... valid_step=false` | Find the batch; see the replay recipe. With the default NaN check Megatron raises instead |
| `perf/wait_time_ratio` near 1 | Rollout-bound | `perf/rollout_time`, `perf/tokens_per_gpu_per_sec`, `rollout/prefix_cache_hit_rate`, dashboard KV usage | More rollout GPUs; `train_async.py` or `--fully-async`; `--sglang-mem-fraction-static` if KV-bound |
| `perf/actor_train_mfu` low | Small micro-batches, heavy recompute, communication | `perf/actor_train_tok_per_s`, dashboard lanes | `--use-dynamic-batch-size --max-tokens-per-gpu`; `--balance-data` |
| Step 2 OOM under colocate | Offload/onload colliding with engine static memory | `perf/cpu_memory_after_offload_train_gb` | Lower `--sglang-mem-fraction-static` |
| Fully-async throughput high, reward stalls | Training on stale data | `rollout/fully_async/avg_staleness`, `stale_groups_filtered`, `train_rollout_kl` | `--max-weight-staleness`; `--async-data-buffer-capacity-factor`; `--use-tis` |
| Eval curve has gaps | Snapshot missing or eval skipped | `eval/skipped_<reason>` | `--eval-overflow-policy`; `--eval-max-in-flight` |
| Eval lower than train reward | Reward hacking, or `None` rewards counted as 0 | `eval/<name>-none_reward_ratio`, read samples | Fix the eval harness or the reward loophole |

## Debugging tooling

[Debugging](/developer/debug) covers cutting the loop in half and the CI assertion
harness. This section covers the recording and inspection tools that page does not.

### What stdout contains

| Line | Emitted by |
|---|---|
| `rollout 12: {'rollout/raw_reward': ...}` and `multi_turn 12: {...}` | train-side rollout logger (`log_utils.py:125`) |
| `perf 12: {...}` | both the train-side perf logger and the rollout-side logger; the rollout-side one also carries the rollout-side `rollout/*` keys (`metrics.py:94`) |
| `step 37: {'train/loss': ...}` | train-step logger; printed twice per step on Megatron (`log_utils.py:554`, `model.py:885`) |
| `eval 12: {...}` | eval logger |
| `Timer actor_train start` / `Timer actor_train end (elapsed: 41.2s)` | every timer, on global rank 0 (`timer.py:23`) |
| `train op=train_step rollout=12 step=0 attempt=0 outcome=NORMAL valid_step=true` | structured logfmt lines (`structured_log.py`), one per optimizer step |
| `First rollout sample: text_preview=...` / `Finish rollout: ...` | rollout functions, once per rollout |

Set `RAY_DEDUP_LOGS=0` to see every rank's line.

### Recording a run

| Flag | Writes |
|---|---|
| `--save-debug-rollout-data PATH` | `PATH.format(rollout_id=...)`: `torch.save(dict(rollout_id, metadata, samples=[Sample.to_dict()]))` of the filtered, trimmed samples, before conversion to training tensors (`ray/rollout/debug_data.py:117`). `PATH` must contain `{rollout_id}`. Eval rollouts use `eval_<id>`, and multi-policy runs `<model_id>_<id>`. Also writes a parquet token mirror to `../dashboard_columns/`. |
| `--save-debug-trajectory-data PATH` | one JSONL row per sample with a recorded conversation (`metadata["messages"]`), then strips the messages from the `.pt` (`debug_data.py:86`) |
| `--save-debug-train-data PATH` | `PATH.format(rollout_id=..., rank=...)`: each TP-rank-0, last-PP-stage rank's `rollout_data` after log-probs and advantages, with its CP rank and size (`miles/backends/training_utils/metrics/train_dump.py:11`) |
| `--dump-details DIR` | sets the three above to `DIR/rollout_data/{rollout_id}.pt`, `DIR/train_data/{rollout_id}_{rank}.pt`, `DIR/trajectory/{rollout_id}.jsonl`, sets `--save-debug-event-data DIR/events` (`arguments.py:3721`), and dumps every policy-loss call's per-token `train_log_probs`, `old_log_probs`, `rollout_log_probs`, advantages, masks, `ppo_kl`, `pg_loss` and finiteness flags to `DIR/policy_loss_debug/rank_<r>_call_<n>.pt` (`training_utils/metrics/debug_dump.py:13`) |
| `--dumper-enable` (+ `--dumper-inference`, `--dumper-fwd-only`, `--dumper-fwd-bwd`, `--dumper-dir`) | SGLang's tensor dumper in the inference engine and in both Megatron phases, for layer-by-layer comparison; forces a single rollout and disables eval, save and fault tolerance (`arguments.py:4276`) |
| `--use-miles-dashboard` | live telemetry streams under `DIR/dashboard/` |
| `MILES_LOGPROB_BACKWARD_DEBUG_DIR=...` | gradient statistics of the log-prob tensors on backward (`math_utils.py:91`) |
| `MILES_DEBUG_PYSPY_DUMP_INTERVAL=SECONDS` | periodic `py-spy dump` of every Python, Ray and SGLang process on the node (`debug_utils/periodic_py_spy.py:19`), for hangs |
| `--save-local-weight-checksum`, `--ci-save-model-hash` | per-rank weight checksums and per-layer hashes |

### Inspecting saved rollouts

A rollout dump is a plain `torch.save`, so the quickest inspection is a Python session:

```python
import torch
from miles.utils.types import Sample

pack = torch.load("dump/rollout_data/57.pt", weights_only=False)
samples = [Sample.from_dict(s) for s in pack["samples"]]
by_group = {}
for s in samples:
    by_group.setdefault(s.group_index, []).append(s.reward)
zero_std = sum(len(set(r)) == 1 for r in by_group.values())
print(len(samples), "samples,", zero_std, "/", len(by_group), "zero-std groups")
worst = max(samples, key=lambda s: s.response_length)
print(worst.status, worst.response_length, worst.response[-2000:])
```

The train dump carries the per-token tensors that the metrics were reduced from, so any
`train/*` or `rollout/*` value can be recomputed by hand from `rollout_data["log_probs"]`,
`["rollout_log_probs"]`, `["advantages"]` and `["loss_masks"]`. Remember that under CP
the log-prob tensors are the rank's zig-zag chunk while `loss_masks` is the full mask.

Command-line helpers in `miles/utils/debug_utils/`:

| Module | Does |
|---|---|
| `replay_reward_fn --rollout-data-path P --custom-rm-path M.fn` | re-scores every sample of a dump with a reward function and prints index, status, reward and truncated text; the fastest way to test a reward change |
| `send_to_sglang --prompt-data F --url U` | sends a prompt file to a running SGLang server with Miles-style sampling args, outside training |
| `run_megatron` (`python -m miles.utils.debug_utils.run_megatron run ...`, `run-and-compare`, `compare`) | runs a standalone Megatron forward (and optional backward) on a prompt, with the dumper and routing replay, and compares per-token log-probs between two runs (`logprob_comparator.py`) |
| `display_debug_rollout_data --load-debug-rollout-data P` | intended to print dump metrics and samples; currently fails to import (see known issues) |

The dashboard (`python -m miles.dashboard.serve --dump-details DIR [--follow]`) is the
visual way to do the same: per-step trajectory tables, zero-std groups, a per-sample
conversation view and a per-token strip colored by log-prob gap or entropy.

### Profiling

`--use-pytorch-profiler --profile-step-start A --profile-step-end B --tensorboard-dir D`
writes a PyTorch trace on every backend. `--profile-target` picks the region:
`train_overall` works everywhere, while `train_actor` and `train_log_probs` only wrap loops
on FSDP and torchtitan (`miles/utils/profile_utils.py:48`, used at
`training_utils/torch_native/actor.py:235`, `:252`); Megatron rejects them at startup
(`megatron_utils/actor.py:163`).
`--record-memory-history` with `--memory-snapshot-dir`, `--memory-snapshot-path` and
`--memory-snapshot-num-steps` dumps a CUDA memory snapshot; `--memory-recorder memray`
swaps in memray.

### Reproducing one step offline

To investigate rollout 57, from a run that recorded with `--dump-details DIR` and saved a
checkpoint at rollout 56:

1. **Freeze the inputs.** Relaunch with the same model and parallelism, `--load` pointing
   at the rollout-56 checkpoint (this sets the start rollout id to 57),
   `--load-debug-rollout-data DIR/rollout_data/{rollout_id}.pt` and `--num-rollout 58`.
   Loading implies `--debug-train-only`, so no engines start. The rollout-side `rollout/*`
   and `perf/rollout_time` keys are not logged in this mode (`metrics.py:83`), but every
   train-side key is.
2. **Pin the reductions.** Add `--debug-deterministic-collective` when comparing two
   topologies bit-for-bit.
3. **Compare.** Add `--save-debug-train-data` to a new path and diff the per-token tensors
   against the original `DIR/train_data/57_*.pt`, or add `--ci-save-grad-norm` on the
   original configuration and `--ci-load-grad-norm` on the modified one.
4. **Shrink.** `--load-debug-rollout-data-subsample 0.1` keeps the first and last 5% of
   the samples.
5. **Isolate the optimizer.** `--debug-disable-optimizer` keeps forward and backward but
   skips the step; `--debug-exit-after-rollout 1` stops after the replayed rollout.

To test whether the engines hold the right weights instead, keep them running with
`--ci-inject-rollout-data-path` (with `--ci-inject-rollout-data-start-rollout-id`), which trains on the recording while still generating and
comparing, as described in [Debugging](/developer/debug#replaying-without-killing-the-engines).

## Mapping from open-instruct

For readers coming from open-instruct's
[monitoring guide](https://github.com/allenai/open-instruct/blob/main/docs/algorithms/monitoring_and_debugging_runs.md).
"Not logged" means no built-in equivalent.

| open-instruct | Miles | Notes |
|---|---|---|
| `scores` | `rollout/raw_reward`, `rollout/episode_raw_reward` | Sample- vs rollout-weighted |
| `val/avg_group_performance_pre_filter` | `rollout/raw_reward_unfiltered` | Sample mean before the dynamic filter; equals the solve rate for binary rewards |
| `val/avg_group_performance_post_filter` | `passrate/pass@1` | Needs `--log-passrate`; binary rewards only |
| `val/advantages_mean` | `rollout/advantages` | Mean only |
| `val/advantages_min/max/hist`, `val/solve_rate_hist` | Not logged | Use a dump or the dashboard |
| `batch/filtered_prompts_zero/_solved/_nonzero` | `rollout/dynamic_filter/drop_zero_std_<r>` | Keyed by rounded reward |
| `batch/total_prompts`, `real_batch_size_ratio` | `rollout/num_training_samples` | |
| `unsolved_batch_size_ratio` | `1 − rollout/zero_std/all_one_percentage` (group-level, approximate) | |
| `packed_ratio` | Not logged | |
| `val/sequence_lengths` | `rollout/response_lengths` (full), `rollout/response_len/mean` (model tokens) | |
| `val/sequence_lengths_solved/_unsolved` | Not logged | `--log-correct-samples` was meant to cover it but does not log |
| `val/stop_rate`, `val/truncated_completion_fraction` | `rollout/truncated_ratio` | |
| `val/non_submitting_completion_fraction` | Not logged | Environment-specific; return it from a custom rollout function |
| `objective/kl0/1/2/3_avg` | `train/kl_loss` | One estimator, chosen by `--kl-loss-type` |
| `loss/kl_avg` | Not logged as a product | `kl_loss_coef × train/kl_loss` |
| `val/ratio` | `train/ois` | Needs `--use-tis` or `--get-mismatch-metrics` |
| `val/ratio_var` | Not logged | `train/ess_ratio` is the closest concentration measure |
| `policy/clipfrac_avg` | `train/pg_clipfrac` | |
| `val/tis_ratio`, `val/tis_clipfrac` | `train/tis`, `train/tis_clipfrac` | |
| `debug/tis_mask_frac_kept` | `train/mis_tis_mask_fraction_*`, `train/mis_rs_mask_fraction_*` | Via `mis.py`; reported as the rejected fraction |
| `debug/dppo_mask_frac_kept`, `debug/tvpo_mask_frac_kept` | Not applicable | No DPPO/TVPO loss; `train/opsm_clipfrac` is the nearest mask metric |
| `debug/vllm_vs_local_logprob_diff_mean` | `train/train_rollout_logprob_abs_diff` | |
| `debug/vllm_vs_local_logprob_diff_max/std` | Not logged | The dashboard has per-sample `max_abs_lp_diff` |
| `debug/vllm_local_reverse_kl` | `train/train_rollout_kl` | Miles' is a proper non-negative k3 estimate |
| `policy/entropy_avg` | `rollout/entropy`, `train/entropy_loss` | Flags as described above |
| `loss/policy_avg`, `loss/total_avg` | `train/pg_loss`, `train/loss` | Miles' `loss` is per rollout, not a fraction of the global loss |
| `optim/grad_norm`, `lr` | `train/grad_norm`, `train/lr-pg_0` | |
| `model_step_min/max/mean` | `rollout/weight_version/*` | Absolute versions; optimizer steps on Megatron, syncs on FSDP |
| `stale_results_dropped` | `rollout/fully_async/stale_groups_filtered` | |
| `time/getting_response` | `perf/rollout_time` | Buffer drain time, not generation, under `--fully-async` |
| `time/training` | `perf/train_time` | |
| `time/weight_sync` | `perf/update_weights_time` | One step late |
| `time/trainer_idle_waiting_for_inference` | `perf/train_wait_time`, `perf/wait_time_ratio` | Also includes save and weight sync |
| `time/generation_idle_waiting_for_trainer` | Not logged | `rollout/fully_async/queue_size` at capacity is the symptom |
| `learner_mfu` | `perf/actor_train_mfu` | Fraction, not percent; excludes wait time |
| `actor_mfu`, `actor_mbu` | Not logged | Dashboard SGLang series |
| `learner_tokens_per_second_*` | `perf/actor_train_tok_per_s` | |
| `val/actor_tokens_per_second` | `perf/tokens_per_gpu_per_sec` | Per GPU |
| `tools/*`, `env/*` | Not logged | Custom rollout metrics; see `agent/*` in the SWE example (not under `--fully-async`) |
| `eval/pass_at_k` | `eval/<name>-pass@k` | `--log-passrate` |
| `value/*` | `train/critic-*` | |

## What is not logged

These are covered by the open-instruct guide or would be expected by someone reading an
RL dashboard, and Miles does not emit them. Each names where the data already is.

| Gap | Why it matters | Where the data is |
|---|---|---|
| Advantage spread (std, min, max, histogram) | Distinguishes "no signal" from "signal, no learning" | `rollout_data["advantages"]` in `log_rollout_data` |
| Pre-filter per-group stats (pass@k, zero-std fractions) | Kept-batch metrics are pinned by the zero-std filter, and `rollout/raw_reward_unfiltered` is only a mean | `all_samples` in `inference_rollout_train.py:187`, passed to `--rollout-all-samples-process-path` |
| Lengths split by solved vs unsolved | Budget-bound failures show as unsolved lengths at the cap | `raw_reward` and `response_lengths` on the train side |
| Per-token mismatch max and quantiles | A mean hides a handful of catastrophic tokens | `abs_diff` in `losses.py:353` before reduction; `mis.py` veto metrics cover part of it |
| Mismatch bucketed by weight-version lag | Separates staleness from numerics in async runs | `weight_versions` per sample |
| Tool calls, tool errors, turns | The agentic health signal | agent metadata; only examples aggregate it |
| Engine abort rate on wandb | Aborted samples discard whole groups | dashboard counters `sglang_num_aborted_requests_total` |
| Per-rank token counts and step times | Stragglers and uneven packing | per-rank timers (only rank 0 is logged), dashboard lanes |
| Token-weighted versions of `train/*` | Comparability with token-mean frameworks | `--calculate-per-token-loss` changes the loss too; no metric-only switch |

## Known issues and doc corrections

Found while verifying this page against the code.

**Bugs**

- `--log-correct-samples` computes nothing visible. The block at `log_utils.py:307` writes
  `correct_length/*` and `correct_entropy` into `rollout_data` after the rollout metrics
  were already gathered, so they are never logged. It also indexes the rank-local
  `response_lengths` with positions from `raw_reward`, which holds the whole batch
  (`train_data_conversion.py:499-510`), so with DP > 1 it can raise `IndexError` or attribute
  lengths to the wrong samples. What it calls entropy is the negative log-prob.
- `rollout/ref_entropy` and `rollout/teacher_entropy` fall through to the generic tensor
  branch (`log_utils.py:249`): they are unmasked token means (tool tokens included) and,
  under context parallelism, multiplied by the CP size.
- `--kl-coef` is silently ignored by `grpo` and `gspo`: `compute_advantages` broadcasts the
  reward and discards the KL tensor (`advantages.py:53`), while the reference model is still
  loaded and run (`ray/specs/train.py:59`). Use `--use-kl-loss --kl-loss-coef` with GRPO.
- `python -m miles.utils.debug_utils.display_debug_rollout_data` fails with `ImportError`:
  it imports `compute_perf_metrics_from_samples` from `miles.ray.rollout`, which does not
  exist (the function is the private `_compute_perf_metrics_from_samples` in
  `miles/ray/rollout/metrics.py`, and it also needs `rollout_time`).
- Fixed since this page was first written: `rollout/zero_std/all_zero_percentage` and
  `all_one_percentage` used to match the strings `"0.0"` and `"1.0"`, so integer rewards
  read 0. They now compare the reward numerically (`metrics.py:329`). They are still
  computed on the post-filter batch, so they read 0 under a zero-std filter.
- `train/ess_ratio` and `train/opsm_clipfrac` are per-sample sums divided by the rollout
  count and can exceed 1 when a rollout produces several training samples.

**Surprises worth knowing**

- `train/pg_clipfrac` and `train/ppo_kl` are 0 (up to kernel nondeterminism) on every first
  optimizer step of a rollout unless `--use-rollout-logprobs` is set, including in async runs
  where the data is stale. `--keep-old-actor` is the other exception: after the first sync it
  anchors on `old_actor`, a different weight version than the one being trained.
- `train/entropy_loss` is always present and always 0 unless entropy is requested.
- `train/lr-pg_*` is the post-step learning rate; `perf/update_weights_time` and other
  post-train timers land on the next rollout's step; `perf/*` is one rank only.
- Under `--fully-async`, `perf/rollout_time` is the buffer drain, so the `perf/*tokens_per*`
  rates read far too high and the `_without_non_generation` rates can go negative.
- Megatron's weight version counts optimizer steps, not syncs, so `rollout/weight_version/*`,
  every fully-async staleness value and the `--max-weight-staleness` bound are in optimizer
  steps and move by the number of optimizer steps per rollout.
- `rollout/repetition_frac` never flags responses shorter than 10,000 characters.
- `--recompute-logprobs-via-prefill` replaces $\ell^\mu$ with SGLang `input_token_logprobs`.
  In the pinned SGLang those are a raw `log_softmax` with no temperature
  (`sglang/python/sglang/srt/layers/logprob_processor.py:619`; still so at `:734` in SGLang
  0.5.21), while the trainer divides by
  `--rollout-temperature`. The two agree only at temperature 1, which is what the shipped
  true-on-policy recipes use. This comes from reading the code, not from a run.
- In `check_kl` the MLA branch asserts the signed `train/ppo_kl < 1e-8` and skips the clip
  fraction (`backends/training_utils/metrics/checks.py:17`), a slightly different check than
  [Debugging](/developer/debug#the-assertion-harness-ci-uses) describes. That page also
  scopes the trainer-vs-engine (0.03) and entropy (0, 0.7) log-prob checks to rollout 0;
  only the `log_probs` vs `ref_log_probs` check is limited to rollout 0, the other two run
  on every rollout (`log_utils.py:290`).

**Stale statements in other pages**

- [Monitoring and Logging](/user-guide/monitoring): the stdout block
  (`[trainer] iter 12/3000 | loss=... reward=...`) is not a format Miles prints; the real
  lines are listed above. The "What to watch" table uses bare names (`raw_reward`,
  `kl_loss`, `entropy_loss`, `rollout_time`, `train_time`) whose actual keys are
  `rollout/raw_reward`, `train/kl_loss`, `train/entropy_loss`, `perf/rollout_time`,
  `perf/train_time`; `entropy_loss` is 0 by default; `pg_clipfrac > 0.5` cannot happen on a
  single-step recipe. The profiling section implies Megatron honors all three
  `--profile-target` values, but Megatron rejects `train_actor` and `train_log_probs` at
  startup, and `train_overall` still needs `--use-pytorch-profiler`. The TensorBoard, MLflow and Prometheus backends and the
  remaining wandb flags (`--wandb-team`, `--wandb-mode`, `--wandb-dir`,
  `--wandb-always-use-train-step`, `--disable-wandb-random-suffix`) are not mentioned.
- `miles/dashboard/README.md` points to `docs/user-guide/dashboard.md`, which no longer
  exists; `docs.json` redirects that URL to Monitoring and Logging, which does not cover
  the dashboard.
- [Rollout Correction Methods](/examples/infra-features/train-infer-mismatch-helper):
  `train/ois` is emitted whenever `--use-tis` or `--get-mismatch-metrics` is set, not only
  for decoupled PPO with `--use-tis`; the page omits `train/train_rollout_kl` and
  `train/ess_ratio`; and `train/ppo_kl`, `train/pg_clipfrac` and
  `train/train_rollout_logprob_abs_diff` are reduced with the post-rejection mask, unlike
  the `mis_*` metrics.
