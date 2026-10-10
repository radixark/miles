# FlashREINFORCE

[FlashREINFORCE](https://www.alphaxiv.org/abs/2609.flashreinforce-asynchronous-rl-agentic-models) is
critic-free RL with **one rollout per prompt**, built for asynchronous training where trajectories
finish at irregular times and come from several policy versions. Without sibling rollouts there is no
group baseline, so it centers rewards across the batch instead, and it controls policy drift with a
sequence-level trust region rather than PPO clipping. This example trains Qwen2.5-Math-1.5B with it on
one 8-GPU node, fully async.

## Files

* `run_qwen2_5_math_1_5b_flash_reinforce.py`: single-node launcher following the paper's
  Qwen2.5-Math-1.5B setting (128 prompts x 1 rollout per update, 4k responses, lr 1e-6, delta 3e-3).

## Quick Start

```bash
cd miles
python examples/flash_reinforce/run_qwen2_5_math_1_5b_flash_reinforce.py
```

`prepare` downloads the model, DAPO-Math-17k and AIME-2024, and converts the checkpoint to Megatron
`torch_dist`. Two GPUs train; the other six serve rollout. Flags passed through `--extra-args` come
last and override the recipe, e.g. `--extra-args "--tis-binary-kl-threshold inf"` for the
importance-weighted PG ablation without the trust region.

## The algorithm as flags

FlashREINFORCE is not a new loss. Each of its pieces is one setting:

| Paper component | Miles |
|---|---|
| One-Batch REINFORCE: `A_i = R_i - mean_j R_j` over the batch, no std, no groups | `--advantage-estimator flash_reinforce` |
| One update per fresh batch, REINFORCE gradient | `--num-steps-per-rollout 1 --skip-actor-forward-only` |
| Token IS `rho = pi / mu` against the sampler, unclipped | `--use-tis --custom-tis-function-path miles.backends.training_utils.loss.hub.corrections.binary_kl_trust_region_function` |
| Sequence trust region `m_i = 1[mean_t binKL(mu_it, pi_it) <= delta]` | `--tis-binary-kl-threshold delta` (default `5e-3`) |
| Sample-mean loss: average within each trajectory, then across the batch | the default reduction; do not set `--calculate-per-token-loss` |

Put together, the loss is `-1/B sum_i m_i A_i / T_i sum_t stopgrad(rho_it) log pi_it`, with no
critic, no reference model, and no ratio clipping.

**Why the PPO loss becomes REINFORCE.** The policy loss is still Miles' PPO surrogate. With
`--skip-actor-forward-only` the old log-probs are the detached training forward, so the PPO ratio is
exactly 1, nothing clips, and the gradient is `-A grad log pi`. Without the flag, one optimizer step per
batch gives the same result at the cost of an extra forward pass.

**The trust region.** For each sampled token, the binary KL collapses the sampler and trainer
distributions to "this token vs. everything else" and needs only the log-probs already stored for IS.
A sequence whose mean exceeds delta gets weight 0 but stays in the batch denominator `B`, and its
entropy and KL terms are untouched. Every other token carries its unclipped ratio `pi / mu`, bounded to
`exp(+-30)` for numerical safety. With context parallelism, each rank holds a slice of every sequence;
the per-sequence sums are all-reduced over the CP group before the mean is taken.

**What counts as one trajectory.** A rollout that Miles trains as several samples (context compaction,
sub-agents) shares one `rollout_id`. The batch baseline counts each rollout once, and the sample-mean
loss divides by the token count of the whole rollout, so the trajectory weighs `1/B` like any other.
The trust region, however, is decided per training sample: each segment of a split trajectory is gated
on its own.

## Choosing delta

The paper runs `1e-3` to `1e-2`: `3e-3` on Qwen2.5-Math-1.5B, where it rejects about 0.13% of
trajectories, `5e-3` on DeepSeek-R1-Distill-Qwen-1.5B at policy lag 4, and `1e-2` on a 30B MoE at lag
8. Treat delta as a screen for drift outliers, not a knob that should reject often. Watch
`train/tis_seq_reject_frac`, the fraction of rejected sequences; `train/tis_binary_kl` is the mean
token proxy, and `train/tis` / `train/tis_abs` report the IS ratio. With `--get-mismatch-metrics` and
without `--use-tis`, the function only logs these metrics and leaves the loss alone.

## Constraints worth knowing before you debug

* **Do not whiten.** `--normalize-advantages` divides the centered rewards by their std, which the
  algorithm deliberately omits.
* **Group filters drop everything.** With `--n-samples-per-prompt 1`, a dynamic-sampling filter that
  rejects zero-variance prompt groups rejects every group.
* **The policy lag comes from the async schedule.** The launcher keeps 512 trajectories in flight
  against 128 per update, about four updates of lag. `--pause-generation-mode in_place` lets in-flight
  requests finish on new weights, so one response can mix policy versions; the per-token ratios correct
  for that.
* **Context length.** The paper's 2k prompt + 4k response budget exceeds the 4k `max_position_embeddings`
  of Qwen2.5-Math-1.5B, so the launcher sets `--sglang-context-length 6144` and
  `SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1`.
* **A batch with one outcome does not move.** If every rollout of a batch scores the same, every
  advantage is 0. The launcher grades the last `\boxed{}` answer with 0/1 (`--rm-type math`), which the
  prompts ask for; a stricter format check such as `--rm-type dapo` scores the base model's first batches
  all wrong, and training never starts.
