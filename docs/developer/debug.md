---
title: Debugging
description: Aligning precision, separate train/rollout debugging, common kernel pitfalls.
---

# Debugging

When something is wrong with a Miles run, the question is almost always: **rollout or
training?** Once you've isolated which side is misbehaving, the rest is a normal
debugging session.

## Aligning precision

The single most common class of bug. Walk through these checks before anything else.

### First training step

#### Is the rollout coherent?

If the very first rollout is gibberish:

* **Parameters didn't load.** Check Megatron's logs — there should be a clear
  "loaded checkpoint from ..." line. If absent, fix `--load` / `--ref-load`.
* **Parameter mapping is wrong.** When `pp_size > 1`, the second-stage layer IDs are a
  common offset bug. Save all parameters in `load_weights` of the corresponding model
  in SGLang and verify they match the loaded checkpoint exactly.
* **SGLang dropped buffers.** Some special buffers can be released during the parameter
  release process. Check if they're re-loaded correctly after weight sync.
* **Pretrained vs. instruct.** Try the instruct version of the same architecture. If
  that works, your pretrained model + chat template combination is wrong.

#### Are `log_probs` and `ref_log_probs` exactly equal in step 1?

They should be: at step 1 the actor and reference are the same weights. If KL > 0:

* **Non-deterministic kernels.** Some Transformer Engine versions need
  `--attention-backend flash` to force deterministic Flash Attention under context
  parallel.
* **Slightly off values (KL < 1e-4).** Acceptable — kernel-level numerical jitter.
* **Large values (KL > 1).** Configuration error. Re-check parallelism and precision.
* **Slightly elevated logp on instruct (~0.8 per token).** Almost always a chat-template
  mismatch — your prompts don't match the format the model was trained on. Run the
  [chat template verifier](../user-guide/agentic-chat-template.md).

#### Is `grad_norm` reasonable?

Step 1 with `num_steps_per_rollout=1` should produce a tiny gradient. If it doesn't:

* **Megatron / TE bug.** MoE in particular requires `--moe-permute-fusion`. Check
  release notes for known fixes.
* **Reward signal is broken.** Confirm rewards are computed from the right key (often a
  `--label-key` typo).

### Second training step

If step 2 OOMs in colocate mode, Megatron's offload→reload cycle is hitting the SGLang
side. Lower `--sglang-mem-fraction-static` to 0.7 (or 0.6).

## Separate train/rollout debugging

Miles ships flags to isolate one side:

| Flag | What |
|---|---|
| `--debug-rollout-only` | Don't init Megatron — only spin up SGLang. |
| `--debug-train-only` | Don't init SGLang — only spin up Megatron. |
| `--save-debug-rollout-data /path/data_{rollout_id}.pt` | Pickle every rollout to disk. |
| `--load-debug-rollout-data /path/data_{rollout_id}.pt` | Replay rollouts from disk; auto-sets `--debug-train-only`. |

A typical workflow:

1. Run with `--debug-rollout-only --save-debug-rollout-data` to capture a few
   well-formed rollouts.
2. Switch to `--debug-train-only --load-debug-rollout-data` to iterate on training
   changes (parallelism, optimizer, custom loss) with **fixed inputs**. Removes
   rollout randomness from your bisect.

This is the single most useful pattern in the Miles workflow. Use it.

## Determinism for bisecting

When you need to A/B test a code change, bit-wise reproducibility is your friend. See
the [Reproducibility recipe](../examples/reproducibility.md) for the exact flag set
and env vars. The 25% throughput cost is worth it during development.

## Common kernel pitfalls

| Symptom | Likely culprit |
|---|---|
| Garbled rollout, parameters loaded fine | Chat template mismatch, or buffer drop in SGLang |
| KL ≠ 0 in step 1 | Non-det fused attention; force `--attention-backend flash` |
| MoE training collapses after ~50 steps | R3 not on, or routing not preserved |
| Gradient NaN/Inf | Bad chat template, or activation overflow in FP8 |
| `illegal memory access` in SGLang | OOM in disguise — lower `--sglang-mem-fraction-static` |
| `JSONDecodeError` from inductor | Cache corrupt; set `TORCHINDUCTOR_FORCE_DISABLE_CACHES=1` |
| NCCL hang during weight sync | `NCCL_TIMEOUT=900`; check `NCCL_DEBUG=INFO` |

## Reading logs

Where to look:

| Component | Path |
|---|---|
| Trainer stdout | wherever you redirected `ray job submit` |
| SGLang | `/tmp/sglang/*.log` (or `--sglang-log-dir`) |
| Ray workers | `~/.ray/session_latest/logs/worker-*.{out,err}` |
| NCCL | `NCCL_DEBUG=INFO NCCL_DEBUG_FILE=/tmp/nccl_%h_%p.log` |

`grep` for these signal phrases:

* `Loaded checkpoint from` — Megatron load OK.
* `weight_sync` — P2P / broadcast events.
* `Continuous async rollout worker started` — async worker alive.
* `WorkerCrashed` / `RaySystemError` — ranks died.
* `forward_logp_max_diff` — FP8 train/inference alignment.

## When all else fails

* Drop to a tiny model (Qwen2.5-0.5B) on a known-good recipe (Reproducibility) to
  isolate framework vs. model.
* `git bisect` between a known-working commit and HEAD. Determinism makes this
  trustworthy.
* Open a GitHub issue with: launch script, `pip freeze`, the first 200 lines of
  trainer stdout, and a short description.

## Useful debugging knobs

```bash
--debug-determinism                      # log per-step hashes
--debug-weight-sync 1                    # verbose P2P transfer logs
--debug-rollout-print-every 1            # dump every Nth rollout to stdout
NCCL_DEBUG=INFO
NCCL_DEBUG_SUBSYS=COLL,P2P
RAY_DEDUP_LOGS=0
```
