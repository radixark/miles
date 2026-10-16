---
title: Deep Dive
sidebarTitle: Overview
description: Code-grounded internals of Miles RL: one training step, every metric's formula, async mechanics, agentic rollouts, open-instruct differences.
---
These pages trace the actual code rather than describing intended behavior. Their `path:line` references were
re-verified at commit `edf2c7ae3e` (the TMax branch, rebased onto upstream `e30d9a8ec6`, October 2026); line
numbers drift, so search by function name when they do. SGLang claims were checked against the 0.5.21 wheel
(production images run 0.5.22.dev46), Harbor claims against `harbor-miles-v0.20.0` `43e944d6`. They complement the
[User Guide](/user-guide/index) and [Architecture](/developer/architecture) pages, and each one lists places where
those pages are stale.

| Page | Read it when you want to know… |
|---|---|
| [One RL step, end to end](/deep-dive/rl-step-walkthrough) | Which actor does what, how `train.py` runs a step, who computes which logprob, advantage/loss formulas, loss normalization, weight sync, the `Sample` → train-dict contract. |
| [Metrics & debugging](/deep-dive/metrics-and-debugging) | The exact formula and emitter of every logged metric, step axes and reduction pitfalls, a symptom → cause → check → lever playbook, dump/replay tooling. |
| [Async & off-policy](/deep-dive/async-and-off-policy) | Timelines for sync / `train_async.py` / `--fully-async`, how staleness arises and is bounded, which logprobs anchor the PPO ratio, weight-sync protocols, fault tolerance. |
| [Agentic rollout internals](/deep-dive/agentic-rollout-internals) | The generate-function contract, the multi-turn token loop, the TITO session server, sandbox lifecycle and failure handling, one Harbor/Terminal episode traced, porting Terminal RL. |
| [Miles vs open-instruct](/deep-dive/open-instruct-comparison) | How Miles and open-instruct differ as RL frameworks, as Terminal-RL harnesses and as Beaker infrastructure, traced through the TMax port, with the flag-by-flag mapping of its production recipe. |

## Coming from open-instruct

| open-instruct doc | Closest Miles material |
|---|---|
| Terminal-RL production scripts (`scripts/general_agent/terminal/rl/`) | [TMax Terminal-RL](/examples/tmax), [Miles vs open-instruct](/deep-dive/open-instruct-comparison) |
| `mason_reference.md`, `podman_sandboxes.md`, `registry_mirror.md`, `checkpoint_resumption.md` | [TMax on Beaker](/examples/tmax/beaker) |
| `terminal_rl_walkthrough.md` | [rl-step-walkthrough](/deep-dive/rl-step-walkthrough) + [agentic-rollout-internals §6](/deep-dive/agentic-rollout-internals) |
| `grpo_fast_internals.md` | [rl-step-walkthrough](/deep-dive/rl-step-walkthrough), [async-and-off-policy](/deep-dive/async-and-off-policy) |
| `rollout_loop_internals.md` | [agentic-rollout-internals](/deep-dive/agentic-rollout-internals), [Agentic Rollout](/user-guide/agentic-rollout) |
| `monitoring_and_debugging_runs.md` | [metrics-and-debugging](/deep-dive/metrics-and-debugging), [Monitoring](/user-guide/monitoring), [Debug](/developer/debug) |
| `rl_with_environments.md` / sandboxed envs | [Environments](/user-guide/environments), [Sandbox providers](/user-guide/sandbox-providers), [Adding a sandbox provider](/developer/adding-a-sandbox-provider), [Harbor](/user-guide/harbor) |
| `tmax_4b_script_reference.md` (arg reference) | [CLI reference](/user-guide/cli-reference), [Argument groups](/user-guide/argument-groups), [Launch script](/user-guide/launch-script) |
| train/inference mismatch discussion | [True on-policy](/examples/infra-features/true-on-policy), [Mismatch helper](/examples/infra-features/train-infer-mismatch-helper) |
| On-policy distillation | [On-policy distillation](/advanced/on-policy-distillation) |

## Findings worth knowing before you launch

Collected from the pages above; each is traced to code there.

- **`--tito-model` does not set SGLang's tool-call or reasoning parser.** Both engine flags default to `None`;
  without them tool calls come back as plain text and agent episodes end after one turn. Pass
  `--sglang-reasoning-parser` / `--sglang-tool-call-parser` explicitly (for Qwen3.5: `qwen3` / `qwen3_coder`); the
  TMax and Harbor launchers do.
- **`--kl-coef` is a no-op for `grpo`/`gspo`** (`get_grpo_returns` only uses the KL tensor's
  shape) but the reference model is still loaded and run. Use `--use-kl-loss` for a KL term.
- **Async GRPO has no off-policy-correction guard, and `--max-weight-staleness` is off by default.**
  `validate_async_off_policy_correction` returns early without `--use-critic`, so `train_async.py` /
  `--fully-async` GRPO anchors the ratio to the current actor unless you pass `--use-rollout-logprobs`, `--use-tis`,
  or `--keep-old-actor`, and staleness is unbounded unless you set `--max-weight-staleness`.
- **On Megatron a weight version is an optimizer step**, so weight-version metrics, staleness and
  `--max-weight-staleness` scale with `--num-steps-per-rollout` × `--update-weights-interval`. Data generated on
  the current weights has staleness 0; open-instruct counts the same data as 1. FSDP and torchtitan count syncs.
- **`train/pg_clipfrac` and `train/ppo_kl` are 0 on the first optimizer step per rollout** (up to
  kernel nondeterminism) unless `--use-rollout-logprobs` or `--keep-old-actor`; in async runs only `train/train_rollout_kl` /
  `train/train_rollout_logprob_abs_diff` reveal staleness (mixed with kernel mismatch). Score-centering runs emit
  neither.
- **On Megatron the default `train/*` reduction is a mean of per-rollout token-means**, and all-masked samples
  still count in the divisor. FSDP and torchtitan average per sample.
- **In fully-async runs `perf/rollout_time` times only the buffer drain**, so the `perf/tokens_per_*` rates are
  inflated and the `_without_non_generation` rates go negative.
- **`--fully-async` refuses `--rollout-function-path`**, so metrics a custom rollout function adds (such as an
  agent's `agent/*` episode statistics) never appear in fully-async runs.
- **Agent functions run in a child process per call by default** (`--custom-agent-function-mode subproc`):
  module-level state, locks and caches in an agent module last one episode, and the module's `abort` hook runs in
  the rollout process, not in the agents. Share limits through `miles.rollout.agentic.rollout_limits`.
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
- **`--pause-generation-mode in_place` keeps old-weight KV for the rest of the rollout call** it belongs to. Under
  `--fully-async` it turns on `--namespaced-radix-cache` by default (session server v2 when one is used), so later
  calls cannot reuse that prefix; `rollout/weight_version/prefill_*` measures it. The default `retract`
  re-prefills under the new weights.
- **Dropping is per group**, and in sync mode dropped prompts are lost for the epoch (`--partial-rollout` returns
  aborted groups to the buffer; filtered ones are still lost).
- **`--rollout-max-response-len` is per turn** on multi-turn paths, and a length-truncated turn
  ends the training sample while the full-episode reward still applies.
- **`perf/update_weights*_time` lands on the next step** (disk-delta metrics too); `train.py` ignores
  `--update-weights-interval`.
- **`rollout/zero_std/all_{zero,one}_percentage` are computed after the filter**, so they read 0 under a zero-std
  filter.
- **A leftover `MILES_ROUTER_EXTERNAL_HOST` now fails the launch**; use `--session-server-external-host` or a
  per-node `MILES_NODE_EXTERNAL_IP`.
