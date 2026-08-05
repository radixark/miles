---
title: Debugging
description: Split a misbehaving run into rollout and training halves, then debug the side that is actually wrong.
---
When a Miles run misbehaves, the first question is always: **rollout or training?** The
flags on this page exist to answer it by cutting the loop into pieces you can run one at a
time. Once you know which side is wrong, it becomes an ordinary debugging session.

## Cut the loop in half

| Flag | What it does |
|---|---|
| `--debug-rollout-only` | Run generation only. The training backend (Megatron or FSDP) is never initialized. |
| `--debug-train-only` | Run training only. No SGLang engines are started. |
| `--save-debug-rollout-data <path>` | Pickle every rollout to `path.format(rollout_id)`. |
| `--load-debug-rollout-data <path>` | Train from those recordings instead of generating. Implies `--debug-train-only`, since it does not start engines. |

The two `--debug-*-only` flags are mutually exclusive and argument validation rejects
setting both.

The pattern worth internalizing:

```bash
# 1. capture a few well-formed rollouts, no trainer involved
--debug-rollout-only --save-debug-rollout-data /tmp/rollout_{rollout_id}.pt

# 2. iterate on training with those exact inputs, no engines involved
--load-debug-rollout-data /tmp/rollout_{rollout_id}.pt
```

Step 2 removes sampling randomness from the loop, which is what makes a bisect or an A/B
of a loss change trustworthy.

### Replaying without killing the engines

`--load-debug-rollout-data` gives up the engines, so it cannot tell you whether the
*weights that reached the engines* are right. `--ci-inject-rollout-data-path` is the
variant that keeps them:

```bash
--ci-inject-rollout-data-path /tmp/rollout_{rollout_id}.pt \
--ci-inject-rollout-data-start-rollout-id 2 \
--ci-inject-rollout-data-min-match-ratio 0.9
```

From the start rollout id onwards, generation still runs and its output is still compared,
then discarded, and training consumes the recording instead. The comparison is the point:
if the mean response-token match ratio between the fresh generation and the recording
falls below `--ci-inject-rollout-data-min-match-ratio` (default `0.9`), the engines are
holding the wrong weights. Legitimate ulp-level drift only flips the occasional sampled
token, so the ratio stays high when the sync is correct.

## Cut a single step into pieces

| Flag | What it removes |
|---|---|
| `--debug-skip-weight-update` | The actor-to-rollout weight update itself, while keeping the offload and onload schedule around it. Separates "the sync is wrong" from "the schedule around the sync is wrong". |
| `--debug-disable-optimizer` | Optimizer and LR-scheduler construction, and the optimizer step. Rollout, log-prob forward and actor forward/backward still run, so this isolates optimizer-state memory and update behavior from the rest. |
| `--debug-exit-after-rollout <n>` | Everything after rollout `n`. Built for exercising checkpoint resume with consistent scheduler state. |

## Make two runs comparable

`--debug-deterministic-collective` runs the training world on the `det_nccl` backend from
`miles.utils.test_utils.det_process_group`, which folds order-sensitive SUM and AVG
reductions in a fixed tree order. That is what makes two different reduction topologies
bitwise comparable, and it is how the Megatron-versus-FSDP alignment test can assert
equality at all. It is slow; never enable it in production.

For end-to-end reproducibility of a whole run, including the sampling path, see the
[Reproducibility recipe](https://github.com/radixark/miles/tree/main/examples/experimental/reproducibility).

## The assertion harness CI uses

An e2e test in Miles is mostly a normal training run with `--ci-test` added. That flag
turns on a set of in-process assertions, so a violated invariant fails the run where it
happens instead of showing up as a wrong number hours later. You can use the same flags on
your own runs, and each checker has an escape hatch for the case where your change makes it
legitimately not hold.

| Checker | What it asserts | Turn off with |
|---|---|---|
| KL | At step 0, `abs(train/ppo_kl) < 1e-9` and `train/pg_clipfrac < 1e-10`, and `train/kl_loss` is zero. MLA relaxes to `1e-8`; LoRA relaxes because the Megatron-to-HF adapter conversion is not bit-exact | `--ci-disable-kl-checker` |
| Log-probs | At rollout 0, trainer `rollout/log_probs` matches `rollout/ref_log_probs` within `1e-8` (`5e-3` under R3, whose reference does not replay routing); trainer versus engine log-probs within `0.03`; rollout entropy in `(0, 0.7)`. Under `--true-on-policy-mode` the two must be exactly equal | `--ci-disable-logprobs-checker` |
| Weight update | Sets `check_weight_update_equal`, comparing trainer and engine weights after a sync. Skipped automatically under either `--debug-*-only` | `--ci-disable-weight-update-checker` |

Three more take values rather than switching off:

**Accuracy gate.** `--ci-metric-checker-key <key> --ci-metric-checker-threshold <x>` asserts
that at least one eval during the run reported `key >= x`. The verdict lands when the
checker is disposed at the end of the run, so a single good eval is enough to pass and a
run with no eval at all fails.

**Gradient-norm comparison.** `--ci-save-grad-norm <path>` writes the grad norm, and
`--ci-load-grad-norm <path>` asserts a later run matches within `rel_tol=abs_tol=0.03`.
Both accept `{role}`, `{rollout_id}` and `{step_id}` in the path, and only rank 0
participates. This is how the Megatron-versus-FSDP alignment test compares two backends
that cannot share a process.

**Per-layer weight hashes.** `--ci-save-model-hash` and `--ci-check-model-hash` compute
SHA256 over each layer's parameter bytes, including name, shape and dtype, and write one
JSON per rank under `iter_<iteration>/model_hash_tp*_pp*_dp*_cp*.json`. Layer granularity
is deliberate: a mismatch names the layer instead of just saying the model differs.

**Fault injection.** `--ci-ft-test-actions` takes a JSON array of actions such as
`{"at_rollout": 3, "action": "stop_cell_at_end", "cell_id": "trainer-actor-0"}`, with
`stop_cell_at_end`, `start_cell_at_end` and `crash_before_allreduce` available and `cell_id`
naming the target cell. It is how the fault-tolerance suite kills things on purpose. See
[Fault Tolerance](/advanced/fault-tolerance).

## Aligning precision

The most common class of bug. Walk these checks before anything else.

### Is the first rollout coherent?

If the very first rollout is gibberish:

* **Parameters didn't load.** Megatron logs a clear load line; if it is absent, fix
  `--load` / `--ref-load`.
* **Parameter mapping is wrong.** With `pp_size > 1`, second-stage layer IDs are a common
  offset bug. Dump the parameters in the SGLang model's `load_weights` and compare against
  the checkpoint.
* **SGLang dropped buffers.** Some buffers can be released during the parameter-release
  path; check they are re-loaded after weight sync.
* **Pretrained versus instruct.** If the instruct model of the same architecture works,
  your base model plus chat template combination is wrong.

### Are `log_probs` and `ref_log_probs` equal at step 1?

They should be, because the actor and the reference are the same weights. This is exactly
what the log-probs checker above asserts, so running with `--ci-test` turns the question
into a hard failure. If KL is non-zero:

* **Non-deterministic kernels.** Some Transformer Engine versions need
  `--attention-backend flash` to force deterministic Flash Attention under context
  parallelism.
* **KL below `1e-4`.** Kernel-level jitter, acceptable.
* **KL above 1.** A configuration error; re-check parallelism and precision.
* **About 0.8 per token on an instruct model.** Almost always a chat-template mismatch. Run
  the [chat template verifier](/user-guide/agentic-rollout).

### Is `grad_norm` reasonable?

Step 1 with `num_steps_per_rollout=1` should produce a tiny gradient. If it does not, look
at MoE fusion first (`--moe-permute-fusion`), then at whether the reward is being computed
from the right key (a `--label-key` typo is the usual cause).

### Does step 2 OOM under colocate?

The trainer's offload and reload cycle is colliding with the engine's static memory. Lower
`--sglang-mem-fraction-static` to 0.7 or 0.6. See
[Training Backends](/user-guide/training-backend) for the layout and offload knobs.

## Common kernel pitfalls

| Symptom | Likely culprit |
|---|---|
| Garbled rollout, parameters loaded fine | Chat-template mismatch, or a dropped buffer in SGLang |
| KL non-zero at step 1 | Non-deterministic fused attention; force `--attention-backend flash` |
| MoE training collapses after tens of steps | Routing not preserved; check `--use-rollout-routing-replay` |
| Gradient NaN or Inf | Bad chat template, or activation overflow in FP8 |
| `illegal memory access` in SGLang | OOM in disguise; lower `--sglang-mem-fraction-static` |
| `JSONDecodeError` from inductor | Corrupt compile cache; set `TORCHINDUCTOR_FORCE_DISABLE_CACHES=1` |
| NCCL hang during weight sync | Raise the NCCL timeout and re-run with `NCCL_DEBUG=INFO` |

## Reading logs

| Component | Where |
|---|---|
| Trainer stdout | Wherever you redirected `ray job submit` |
| Ray workers | `~/.ray/session_latest/logs/worker-*.{out,err}` |
| SGLang | Inside the Ray worker logs; verbosity via `--sglang-log-level` |
| NCCL | `NCCL_DEBUG=INFO NCCL_DEBUG_FILE=/tmp/nccl_%h_%p.log` |

Useful environment variables while debugging:

```bash
RAY_DEDUP_LOGS=0                       # every rank's line, not one deduplicated line
NCCL_DEBUG=INFO
NCCL_DEBUG_SUBSYS=COLL,P2P
TORCHINDUCTOR_FORCE_DISABLE_CACHES=1
```

The per-step metrics are the real signal, and they are the same numbers the checkers above
assert on: `train/ppo_kl`, `train/pg_clipfrac`, `train/kl_loss`, `rollout/log_probs`,
`rollout/ref_log_probs`, `rollout/rollout_log_probs`, `rollout/entropy`. Trainer steps log
as `step <n>: {...}`, and the rollout side logs one reduced dict per rollout. See
[Monitoring and Logging](/user-guide/monitoring) for the full metric surface.

## When all else fails

* Drop to a tiny model on a known-good recipe (the
  [Reproducibility](https://github.com/radixark/miles/tree/main/examples/experimental/reproducibility) one) to separate framework from model.
* `git bisect` between a known-good commit and HEAD, with the record-and-replay pattern
  above pinning the inputs and `--debug-deterministic-collective` pinning the reductions.
* Open a GitHub issue with the launch script, `pip freeze`, the first 200 lines of trainer
  stdout, and what you have already ruled out.
