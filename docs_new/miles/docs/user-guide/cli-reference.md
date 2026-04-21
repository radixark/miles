---
title: CLI Reference
description: Every command-line flag Miles accepts, grouped by subsystem. The essentials live up top; the complete catalogue follows.
---

# CLI Reference

Miles is configured entirely through command-line flags passed to `train.py` (or
`train_async.py`). There is no YAML file, no config object, no plugin registry — what
you type on the command line *is* the configuration.

This page is structured in two passes:

1. **Essentials.** The flags you will actually touch, grouped by intent.
2. **Complete reference.** Every flag Miles accepts, with types and defaults.

To regenerate the canonical list from source code:

```bash
python -m miles.args --help
```

---

# Part 1 — Essentials

The knobs that matter when you're sizing a run, picking an algorithm, or tuning
throughput.

## Cluster topology

| Flag | Default | What |
|---|---|---|
| `--actor-num-nodes` | `1` | Total nodes for the actor. |
| `--actor-num-gpus-per-node` | `8` | GPUs per actor node. |
| `--rollout-num-gpus` | derived | GPUs for SGLang rollout (ignored when `--colocate`). |
| `--rollout-num-gpus-per-engine` | `2` | TP size of each SGLang engine. |
| `--colocate` | off | Share GPUs between actor and rollout. |

Decide colocate vs. disaggregate first; everything else follows from that choice. See
[Training Script Walkthrough: Colocation](training-script-walkthrough.md#colocation-share-gpus-or-dont).

## Batch sizing

The four-knob invariant:

```
rollout_batch_size × n_samples_per_prompt
  = global_batch_size × num_steps_per_rollout
```

| Flag | Typical | What |
|---|---|---|
| `--rollout-batch-size` | `16 – 256` | Prompts per rollout. |
| `--n-samples-per-prompt` | `4 – 16` | Responses per prompt (GRPO group size). |
| `--global-batch-size` | derived | Samples per optimiser step. |
| `--num-steps-per-rollout` | `1` | Optimiser steps per rollout. |
| `--num-rollout` | `1000 – 10000` | Total rollout iterations. |

## Memory & throughput

| Flag | Default | What |
|---|---|---|
| `--use-dynamic-batch-size` | recommended on | Pack varlen samples into micro-batches. |
| `--max-tokens-per-gpu` | `4096 – 16384` | Token budget per micro-batch per GPU. |
| `--context-parallel-size` | `1` | Spread a single sample across N CP ranks. |
| `--recompute-granularity` | `full` | `full` or `selective`. |
| `--recompute-method` | `uniform` | `uniform` or `block`. |
| `--recompute-num-layers` | `1` | Layers per recompute chunk. |

Rule of thumb: start with `max_tokens_per_gpu = rollout_max_response_len / cp_size`,
then push up until you OOM.

## RL algorithm

| Flag | Default | What |
|---|---|---|
| `--advantage-estimator` | `grpo` | `grpo` / `gspo` / `ppo` / `reinforce++` / `reinforce++_baseline` / `on_policy_distillation` |
| `--use-kl-loss` | off | Compute KL against the reference model. |
| `--kl-loss-coef` | `0.0` | Weight of KL in the loss (0 = monitor only). |
| `--kl-loss-type` | `low_var_kl` | `kl`, `abs_kl`, `low_var_kl`, `mse_kl`. |
| `--entropy-coef` | `0.0` | Entropy bonus weight. |
| `--eps-clip` | `0.2` | PPO/GRPO low clip. |
| `--eps-clip-high` | `0.28` | DAPO-style asymmetric high clip. |
| `--use-tis` | off | Truncated Importance Sampling — turn on when train ≠ inference. |

## Sampling

| Flag | Default | What |
|---|---|---|
| `--rollout-temperature` | `1.0` | Sampling temperature. |
| `--rollout-top-p` | `1.0` | Top-p truncation. |
| `--rollout-max-response-len` | `8192` | Max tokens per response. |
| `--rollout-stop-token-ids` | model default | Stop token IDs. Override when generations don't stop. |
| `--apply-chat-template` | off | Apply the tokenizer's chat template. |
| `--rollout-shuffle` | off | Shuffle prompts each rollout. |

## Optimiser

| Flag | Default | What |
|---|---|---|
| `--optimizer` | `adam` | `adam`, `sgd`. |
| `--lr` | `1e-6` | Learning rate. **Start small for post-training.** |
| `--lr-decay-style` | `constant` | `constant`, `linear`, `cosine`. |
| `--weight-decay` | `0.1` | L2 weight decay. |
| `--adam-beta1`, `--adam-beta2` | `0.9, 0.98` | Adam moments. |

## Logging

| Flag | Default | What |
|---|---|---|
| `--use-wandb` | off | Log to Weights & Biases. |
| `--wandb-project` | — | wandb project name. |
| `--log-interval` | `1` | Stdout log cadence (rollouts). |
| `--save-interval` | `100` | Checkpoint cadence (rollouts). |

## SGLang passthrough

Anything accepted by `python -m sglang.launch_server` is accepted by Miles prefixed
with `--sglang-`:

```bash
--sglang-log-level INFO
--sglang-mem-fraction-static 0.8
--sglang-enable-overlap-schedule
--sglang-enable-ep-moe
--sglang-enable-dp-attention
```

See [SGLang docs](https://docs.sglang.io) for the full list.

## Environment variables

Set these in Ray's `env_vars` for multi-node runs:

| Variable | Effect |
|---|---|
| `MILES_DEBUG=1` | Verbose internal logging. |
| `TORCHINDUCTOR_FORCE_DISABLE_CACHES=1` | Workaround for torch-compile JSONDecodeError. |
| `RAY_DEDUP_LOGS=0` | Don't deduplicate worker logs. |
| `NCCL_DEBUG=INFO` | NCCL diagnostics. |
| `PYTHONPATH=/root/Megatron-LM` | Required when using the Megatron backend. |

---

# Part 2 — Complete reference

Every flag Miles accepts. Section headings mirror the launch-script argument groups.

## Cluster

| Flag | Type | Default | Notes |
|---|---|---|---|
| `--actor-num-nodes` | int | `1` | Total nodes for actor training. |
| `--actor-num-gpus-per-node` | int | `8` | GPUs per actor node. |
| `--rollout-num-gpus` | int | derived | Ignored under `--colocate`. |
| `--rollout-num-gpus-per-engine` | int | `2` | TP size of each SGLang engine. |
| `--colocate` | flag | off | Share GPUs between actor and rollout. |
| `--placement-group-strategy` | str | `STRICT_PACK` | Ray placement-group strategy. <!-- TODO(verify): not found in miles source via grep (2026-04); may have been renamed or removed. --> |

## Model & checkpoints

| Flag | Type | Default | Notes |
|---|---|---|---|
| `--train-backend` | enum | `megatron` | `megatron` or `fsdp`. |
| `--hf-checkpoint` | path | – | HF model dir — tokenizer + config, and the weights FSDP loads. |
| `--ref-load` | path | – | Reference model in `torch_dist` format (Megatron). |
| `--load` | path | – | Actor checkpoint to resume from. |
| `--save` | path | – | Actor checkpoint write directory. |
| `--save-interval` | int | `100` | Rollouts between saves. |
| `--ckpt-step` | int | latest | Force-load a specific iteration. |
| `--ckpt-format` | enum | `torch_dist` | `torch` or `torch_dist`. |
| `--model-name` | str | – | Set in multi-node to avoid `transformers` file-system race. |
| `--spec` | `<module> <fn>` | – | Plugin spec for custom architectures (e.g. `miles_plugins.models.qwen3_5 get_qwen3_5_spec`). |

## Rollout — data & batching

| Flag | Type | Default | Notes |
|---|---|---|---|
| `--prompt-data` | str+ | – | One or more `name path` pairs. |
| `--input-key` | str | `prompt` | JSONL key → `Sample.prompt`. |
| `--label-key` | str | `label` | JSONL key → `Sample.label`. |
| `--metadata-key` | str | – | JSONL key → `Sample.metadata`. |
| `--apply-chat-template` | flag | off | Apply tokenizer chat template. |
| `--rollout-shuffle` | flag | off | Shuffle prompts each rollout. |
| `--num-rollout` | int | `1000` | Total rollout iterations. |
| `--rollout-batch-size` | int | – | Prompts per rollout. |
| `--n-samples-per-prompt` | int | `1` | Responses per prompt. |
| `--global-batch-size` | int | derived | Samples per optimiser step. |
| `--num-steps-per-rollout` | int | `1` | Optimiser steps per rollout. |
| `--over-sampling-batch-size` | int | – | Oversample size for dynamic sampling (DAPO). |
| `--balance-data` | flag | off | Balance per-rank token count. |

## Rollout — sampling

| Flag | Type | Default | Notes |
|---|---|---|---|
| `--rollout-max-response-len` | int | `8192` | Max tokens per response. |
| `--rollout-temperature` | float | `1.0` | Sampling temperature. |
| `--rollout-top-p` | float | `1.0` | Top-p truncation. |
| `--rollout-top-k` | int | `-1` | Top-k truncation (-1 = off). |
| `--rollout-stop` | str+ | – | Stop strings. |
| `--rollout-stop-token-ids` | int+ | – | Stop token IDs. |

## Eval

| Flag | Type | Default | Notes |
|---|---|---|---|
| `--eval-prompt-data` | str+ | – | `name path` pairs. |
| `--eval-interval` | int | `5` | Rollouts between eval runs. |
| `--n-samples-per-eval-prompt` | int | `1` | Responses per eval prompt. |
| `--eval-max-response-len` | int | `16384` | Max eval response length. |
| `--eval-temperature` | float | `0.0` | Eval temperature. |
| `--eval-top-p` | float | `1.0` | Eval top-p. |

## Performance

| Flag | Type | Default | Notes |
|---|---|---|---|
| `--tensor-model-parallel-size` | int | `1` | TP. |
| `--pipeline-model-parallel-size` | int | `1` | PP. |
| `--context-parallel-size` | int | `1` | CP. |
| `--expert-model-parallel-size` | int | `1` | EP (MoE). |
| `--expert-tensor-parallel-size` | int | `1` | TP within experts. |
| `--sequence-parallel` | flag | off | Enable Megatron sequence parallel. |
| `--use-dynamic-batch-size` | flag | off | Pack varlen samples. **Recommend on.** |
| `--max-tokens-per-gpu` | int | `4096` | Token budget per micro-batch per GPU. |
| `--micro-batch-size` | int | `1` | Ignored when dynamic batching is on. |
| `--recompute-granularity` | enum | `full` | `full` or `selective`. |
| `--recompute-method` | enum | `uniform` | `uniform` or `block`. |
| `--recompute-num-layers` | int | `1` | Recompute chunk size. |
| `--gradient-checkpointing` | flag | off | FSDP equivalent of recompute flags. |
| `--fsdp-cpu-offload` | flag | off | FSDP: offload params / grads / optimiser state to CPU. |
| `--fsdp-cpu-backend` | str | – | FSDP: CPU backend for hybrid offload. |
| `--attn-implementation` | enum | – | FSDP: `flash_attention_2`, `sdpa`, `eager`. |

## RL algorithm

| Flag | Type | Default | Notes |
|---|---|---|---|
| `--advantage-estimator` | enum | `grpo` | `grpo`, `gspo`, `ppo`, `reinforce++`, `reinforce++_baseline`, `on_policy_distillation` |
| `--use-kl-loss` | flag | off | Compute KL vs. reference. |
| `--kl-loss-coef` | float | `0.0` | KL weight in loss (0 = monitor). |
| `--kl-loss-type` | enum | `low_var_kl` | `kl`, `abs_kl`, `low_var_kl`, `mse_kl`. |
| `--entropy-coef` | float | `0.0` | Entropy bonus weight. |
| `--eps-clip` | float | `0.2` | PPO/GRPO low clip. |
| `--eps-clip-high` | float | – | Asymmetric high clip. |
| `--use-tis` | flag | off | Truncated Importance Sampling. |
| `--use-routing-replay` | flag | off | Forward-backward routing consistency. |
| `--use-rollout-routing-replay` | flag | off | R3 (requires `--use-miles-router`). |
| `--calculate-per-token-loss` | flag | off | Per-token loss reduction. |
| `--no-check-for-nan-in-loss-and-grad` | flag | off | Skip NaN/Inf guard (debug only). |
| `--true-on-policy-mode` | flag | off | Strict on-policy: reject samples from a prior policy. |

## Optimiser

| Flag | Type | Default | Notes |
|---|---|---|---|
| `--optimizer` | enum | `adam` | `adam`, `sgd`. |
| `--lr` | float | `1e-6` | Learning rate. |
| `--lr-decay-style` | enum | `constant` | `constant`, `linear`, `cosine`. |
| `--lr-warmup-iters` | int | `0` | Warmup steps. |
| `--min-lr` | float | `0` | Lower LR bound for decay schedules. |
| `--weight-decay` | float | `0.1` | L2 weight decay. |
| `--adam-beta1` | float | `0.9` | – |
| `--adam-beta2` | float | `0.98` | – |
| `--clip-grad` | float | `1.0` | Grad clipping. |
| `--optimizer-cpu-offload` | flag | off | Megatron CPU Adam. |
| `--overlap-cpu-optimizer-d2h-h2d` | flag | off | Overlap D2H/H2D with compute. |
| `--use-precision-aware-optimizer` | flag | off | Precision-aware optimiser path. |

## Reward / filters

| Flag | Type | Default | Notes |
|---|---|---|---|
| `--rm-type` | enum | – | Built-in reward: `math`, `dapo`, `deepscaler`, `f1`, `gpqa`, `ifbench`, `remote_rm`. |
| `--rm-url` | str | – | Endpoint when `--rm-type remote_rm`. |
| `--group-rm` | flag | off | Batched reward computation. |
| `--custom-rm-path` | str | – | Custom reward function (see [Customization](customization.md)). |
| `--dynamic-sampling-filter-path` | str | – | Group filter (DAPO-style). |
| `--buffer-filter-path` | str | – | Buffer dequeue filter. |
| `--rollout-sample-filter-path` | str | – | Per-sample filter. |

## SGLang

| Flag | Type | Default | Notes |
|---|---|---|---|
| `--sglang-router-ip` | str | – | External router IP (Miles starts its own if unset). |
| `--sglang-router-port` | int | – | External router port. |
| `--use-miles-router` | flag | off | Use the Miles router (preserves routing metadata for R3). |
| `--sglang-*` | — | — | Any flag accepted by `python -m sglang.launch_server` works with this prefix. |
| `--router-*` | — | — | Any flag accepted by `sgl-router` works with this prefix. |

Common `--sglang-*` flags:

```bash
--sglang-mem-fraction-static 0.8
--sglang-context-length 32768
--sglang-log-level INFO
--sglang-enable-ep-moe
--sglang-enable-dp-attention
--sglang-enable-deepep
--sglang-enable-overlap-schedule
--sglang-enforce-piecewise-cuda-graph     # off by default in colocate
```

## MTP / speculative decoding

| Flag | Type | Default | Notes |
|---|---|---|---|
| `--mtp-num-layers` | int | `0` | Number of MTP layers in the checkpoint. |
| `--enable-mtp-training` | flag | off | Train MTP alongside the policy. |
| `--mtp-loss-scaling-factor` | float | `0.2` | Weight of MTP loss. |

## Fault tolerance

| Flag | Type | Default | Notes |
|---|---|---|---|
| `--use-fault-tolerance` | flag | off | Enable rank-level recovery + heartbeats. |
| `--rollout-health-check-first-wait` | int | `0` | Grace period before heartbeats start. |
| `--rollout-health-check-interval` | int | `30` | Seconds between heartbeats. |
| `--rollout-health-check-timeout` | int | `30` | Heartbeat timeout. |
| `--p2p-weight-sync-retries` | int | `3` | Retry count before falling back to broadcast. <!-- TODO(verify): not found in miles source via grep (2026-04); may be named differently. --> |

## Async / partial rollout

| Flag | Type | Default | Notes |
|---|---|---|---|
| `--fully-async-rollout` | flag | off | Continuous background worker. <!-- TODO(verify): exact flag name not confirmed via grep; may be `--async-rollout` or similar. --> |
| `--partial-rollout` | flag | off | Resume aborted rollouts in the next iteration. |

## Logging

| Flag | Type | Default | Notes |
|---|---|---|---|
| `--use-wandb` | flag | off | Enable wandb. |
| `--wandb-project` | str | – | Project name. |
| `--wandb-group` | str | – | Group name. |
| `--log-interval` | int | `1` | Stdout log cadence (rollouts). |
| `--custom-rollout-log-function-path` | str | – | Custom train logger. |
| `--custom-eval-rollout-log-function-path` | str | – | Custom eval logger. |

## Debugging

| Flag | Type | Default | Notes |
|---|---|---|---|
| `--debug-rollout-only` | flag | off | Skip Megatron; only spin up SGLang. |
| `--debug-train-only` | flag | off | Skip SGLang; only spin up Megatron. |
| `--save-debug-rollout-data` | path | – | Pickle every rollout to disk. |
| `--load-debug-rollout-data` | path | – | Replay rollouts from disk (implies `--debug-train-only`). |
| `--debug-determinism` | flag | off | Log per-step hashes. <!-- TODO(verify): not found in miles source via grep (2026-04). --> |
| `--debug-weight-sync` | int | `0` | Verbose P2P transfer logs. <!-- TODO(verify): not found in miles source via grep (2026-04). --> |
| `--deterministic-mode` | flag | off | Megatron deterministic mode. |

## Customisation

See the [Customization](customization.md) page for the full catalogue of `--*-path`
flags that replace or extend Miles's behaviour.
