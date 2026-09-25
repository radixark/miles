---
title: Deep Dive
sidebarTitle: Overview
description: Code-grounded internals of Miles RL — one training step, every metric's formula, async/off-policy mechanics, and agentic rollouts.
---
These pages trace the actual code (with `path:line` references as of commit `b2c9e4632`;
line numbers drift; SGLang claims verified against `sglang-miles-v0.5.19-final` `c9660a7e11`, Harbor against `harbor-miles-v0.20.0` `43e944d6`) rather than describing intended behavior. They complement the
[User Guide](../user-guide/index.md) and [Architecture](../developer/architecture.md)
pages, and each one lists places where those pages are stale.

| Page | Read it when you want to know… |
|---|---|
| [One RL step, end to end](./rl-step-walkthrough.md) | Which actor does what, how `train.py` runs a step, who computes which logprob, advantage/loss formulas, loss normalization, weight sync, the `Sample` → train-dict contract. |
| [Metrics & debugging](./metrics-and-debugging.md) | The exact formula and emitter of every logged metric, step axes and reduction pitfalls, a symptom → cause → check → lever playbook, dump/replay tooling. |
| [Async & off-policy](./async-and-off-policy.md) | Timelines for sync / `train_async.py` / `--fully-async`, how staleness arises and is bounded, which logprobs anchor the PPO ratio, weight-sync protocols, fault tolerance. |
| [Agentic rollout internals](./agentic-rollout-internals.md) | The generate-function contract, the multi-turn token loop, the TITO session server, sandbox lifecycle and failure handling, one Harbor/Terminal episode traced, porting Terminal RL. |

## Coming from open-instruct

| open-instruct doc | Closest Miles material |
|---|---|
| `terminal_rl_walkthrough.md` | [rl-step-walkthrough](./rl-step-walkthrough.md) + [agentic-rollout-internals §6](./agentic-rollout-internals.md) |
| `grpo_fast_internals.md` | [rl-step-walkthrough](./rl-step-walkthrough.md), [async-and-off-policy](./async-and-off-policy.md) |
| `rollout_loop_internals.md` | [agentic-rollout-internals](./agentic-rollout-internals.md), [Agentic Rollout](../user-guide/agentic-rollout.md) |
| `monitoring_and_debugging_runs.md` | [metrics-and-debugging](./metrics-and-debugging.md), [Monitoring](../user-guide/monitoring.md), [Debug](../developer/debug.md) |
| `rl_with_environments.md` / sandboxed envs | [Environments](../user-guide/environments.md), [Sandbox providers](../user-guide/sandbox-providers.md), [Adding a sandbox provider](../developer/adding-a-sandbox-provider.md), [Harbor](../user-guide/harbor.md) |
| `tmax_4b_script_reference.md` (arg reference) | [CLI reference](../user-guide/cli-reference.md), [Argument groups](../user-guide/argument-groups.md), [Launch script](../user-guide/launch-script.md) |
| train/inference mismatch discussion | [True on-policy](../examples/infra-features/true-on-policy.md), [Mismatch helper](../examples/infra-features/train-infer-mismatch-helper.md) |
| On-policy distillation | [On-policy distillation](../advanced/on-policy-distillation.md) |

## Findings worth knowing before you launch

Collected from the pages above; each is traced to code there.

- **`--tito-model` does not set SGLang's tool-call or reasoning parser.** Both engine flags default to `None`;
  without them tool calls come back as plain text and agent episodes end after one turn. Pass
  `--sglang-reasoning-parser` / `--sglang-tool-call-parser` explicitly (for Qwen3.5: `qwen3` / `qwen3_coder`).
- **`--kl-coef` is a no-op for `grpo`/`gspo`** (`get_grpo_returns` only uses the KL tensor's
  shape) but the reference model is still loaded and run. Use `--use-kl-loss` for a KL term.
- **Async GRPO has no staleness guard.** `validate_async_off_policy_correction` returns early
  without `--use-critic`, so `train_async.py` / `--fully-async` GRPO anchors the ratio to the
  current actor unless you pass `--use-rollout-logprobs`, `--use-tis`, or `--keep-old-actor`.
- **`train/pg_clipfrac` and `train/ppo_kl` are 0 on the first optimizer step per rollout** (up to
  kernel nondeterminism) unless `--use-rollout-logprobs` or `--keep-old-actor`; in async runs only `train/train_rollout_kl` /
  `train/train_rollout_logprob_abs_diff` reveal staleness (mixed with kernel mismatch).
- **Default `train/*` reduction is a mean of per-rollout token-means**, and all-masked samples
  still count in the divisor.
- **Harbor failures split three ways.** Sandbox failures before the first model call (pull/build,
  container start, agent install) drop the whole group. Failures after it (verifier crash/timeout,
  `max_seq_len`) come back as HTTP 200 with `reward: 0.0` and train as 0 — returning `None` from the
  reward function does not catch them; key on `exit_status` + an empty `eval_report`. Agent
  timeouts / non-zero exits still run the verifier and are graded normally.
- **Harbor timeouts are multiplied by `HARBOR_TIMEOUT_MULTIPLIER` (default 2.0)**, so the example's
  agent timeout exceeds the Miles client timeout; `/flush` and the heartbeat kill trials without
  teardown, leaving Docker containers running.
- **`--rollout-max-response-len` never reaches the engine for Harbor's mini-swe-agent** (Harbor drops
  its `sampling_params`); terminus-2 forwards it.
- **`--pause-generation-mode in_place` keeps old-weight KV in the radix cache** across the update,
  and later requests sharing that prefix reuse it. The default `retract` re-prefills under the new
  weights.
- **Dropping is per group**, and in sync mode dropped prompts are lost for the epoch.
- **`--rollout-max-response-len` is per turn** on multi-turn paths, and a length-truncated turn
  ends the training sample while the full-episode reward still applies.
- **`perf/update_weights*_time` lands on the next step**; `train.py` ignores
  `--update-weights-interval`.
- **`rollout/zero_std/all_{zero,one}_percentage`** string-match `"0.0"`/`"1.0"` and are computed
  post-filter.
