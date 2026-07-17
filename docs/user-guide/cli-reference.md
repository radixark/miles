---
title: CLI Reference
description: Every command-line flag Miles accepts, grouped by subsystem.
---
Miles is configured through command-line flags passed to `train.py` or
`train_async.py`. The Megatron flags (such as `--num-layers`, `--rotary-base`,
`--recompute-granularity`) are inherited via Megatron's argument parser; Miles adds
its own flags through an `extra_args_provider`. Run `python3 train.py --help` against
your installed Megatron source for the canonical list.

This page has two passes.

1. **Essentials** lists the flags most runs actually touch.
2. **Complete reference** lists every Miles flag with type and default.

---

## Essentials

### Cluster topology

| Flag | Default | What |
|---|---|---|
| `--actor-num-nodes` | `1` | Total nodes for the actor. |
| `--actor-num-gpus-per-node` | `8` | GPUs per actor node. |
| `--rollout-num-gpus` | derived | GPUs for SGLang rollout (ignored when `--colocate`). |
| `--rollout-num-gpus-per-engine` | `1` | TP size of each SGLang engine. |
| `--colocate` | off | Share GPUs between actor and rollout. |

See [Training Script Walkthrough: Colocation](/user-guide/training-script-walkthrough#colocation-share-gpus-or-dont)
for what `--colocate` flips on under the hood.

### Batch sizing

The four-knob invariant:

```
rollout_batch_size × n_samples_per_prompt
  = global_batch_size × num_steps_per_rollout
```

| Flag | Typical | What |
|---|---|---|
| `--rollout-batch-size` | `16 – 256` | Prompts per rollout. |
| `--n-samples-per-prompt` | `4 – 16` | Responses per prompt (GRPO group size). |
| `--global-batch-size` | derived | Samples per optimizer step. |
| `--num-steps-per-rollout` | `1` | Optimizer steps per rollout. |
| `--num-rollout` | `1000 – 10000` | Total rollout iterations. |

### Memory and throughput

| Flag | Default | What |
|---|---|---|
| `--use-dynamic-batch-size` | off | Pack varlen samples into micro-batches. |
| `--max-tokens-per-gpu` | `–` | Token budget per micro-batch per GPU. Required when dynamic batching is on. |
| `--context-parallel-size` | `1` | Spread a single sample across N CP ranks. |
| `--recompute-granularity` | Megatron default | `full` or `selective`. |
| `--recompute-method` | Megatron default | `uniform` or `block`. |
| `--recompute-num-layers` | Megatron default | Layers per recompute chunk. |

Rule of thumb: start with `max_tokens_per_gpu = rollout_max_response_len / cp_size`,
then push up until you OOM.

### RL algorithm

| Flag | Default | What |
|---|---|---|
| `--advantage-estimator` | `grpo` | `grpo`, `gspo`, `ppo`, `reinforce_plus_plus`, `reinforce_plus_plus_baseline`, `on_policy_distillation` |
| `--use-kl-loss` | off | Compute KL against the reference model. |
| `--kl-loss-coef` | `0.0` | Weight of KL in the loss (0 means monitor only). |
| `--kl-loss-type` | `k1` | `k1`, `k2`, `k3`, `low_var_kl`. |
| `--entropy-coef` | `0.0` | Entropy bonus weight. |
| `--observe-training-entropy` | off | Log training entropy even when `--entropy-coef` is `0.0`; detached from backward when the coefficient is zero. |
| `--eps-clip` | `0.2` | PPO/GRPO low clip. |
| `--eps-clip-high` | `–` | Asymmetric high clip (DAPO-style). |
| `--use-tis` | off | Truncated Importance Sampling for train/inference precision mismatch. |

### Sampling

| Flag | Default | What |
|---|---|---|
| `--rollout-temperature` | `1.0` | Sampling temperature. |
| `--rollout-top-p` | `1.0` | Top-p truncation. |
| `--rollout-max-response-len` | `–` | Max tokens per response. |
| `--rollout-stop-token-ids` | model default | Stop token IDs. Override when generations don't stop. |
| `--apply-chat-template` | off | Apply the tokenizer's chat template. |
| `--rollout-shuffle` | off | Shuffle prompts each rollout. |

### Optimizer

| Flag | Default | What |
|---|---|---|
| `--optimizer` | `adam` | `adam`, `sgd`. |
| `--lr` | `1e-6` | Learning rate. Post-training is sensitive to large updates; recipes typically stay near `1e-6`. |
| `--lr-decay-style` | `constant` | `constant`, `linear`, `cosine`. |
| `--weight-decay` | `0.1` | L2 weight decay. |
| `--adam-beta1`, `--adam-beta2` | `0.9, 0.98` | Adam moments. |

### Logging

| Flag | Default | What |
|---|---|---|
| `--use-wandb` | off | Log to Weights and Biases. |
| `--wandb-project` | – | wandb project name. |
| `--log-interval` | `1` | Stdout log cadence (rollouts). |
| `--save-interval` | – | Checkpoint cadence (rollouts). Recipes typically set 20 to 100. |

### SGLang passthrough

Any flag accepted by `python -m sglang.launch_server` is accepted by Miles with the
`--sglang-` prefix:

```bash
--sglang-log-level INFO
--sglang-mem-fraction-static 0.8
--sglang-enable-overlap-schedule
--sglang-enable-ep-moe
--sglang-enable-dp-attention
```

See [SGLang docs](https://docs.sglang.io) for the full list.

### Environment variables

Set these in Ray's `env_vars` for multi-node runs:

| Variable | Effect |
|---|---|
| `TORCHINDUCTOR_FORCE_DISABLE_CACHES=1` | Workaround for torch-compile JSONDecodeError. |
| `RAY_DEDUP_LOGS=0` | Don't deduplicate worker logs. |
| `NCCL_DEBUG=INFO` | NCCL diagnostics. |
| `PYTHONPATH=/root/Megatron-LM` | Required when using the Megatron backend. |

---

## Complete reference

Sections mirror the launch-script argument groups.

### Cluster

| Flag | Type | Default | Notes |
|---|---|---|---|
| `--actor-num-nodes` | int | `1` | Total nodes for actor training. |
| `--actor-num-gpus-per-node` | int | `8` | GPUs per actor node. |
| `--rollout-num-gpus` | int | derived | Ignored under `--colocate`. |
| `--rollout-num-gpus-per-engine` | int | `1` | TP size of each SGLang engine. |
| `--colocate` | flag | off | Share GPUs between actor and rollout. Implicitly enables `--offload-train`, `--offload-rollout`, and defaults `--sglang-cuda-graph-backend-prefill=disabled`. |

### Model and checkpoints

| Flag | Type | Default | Notes |
|---|---|---|---|
| `--train-backend` | enum | `megatron` | `megatron` or `fsdp`. |
| `--hf-checkpoint` | path | – | HF model dir. Provides tokenizer, config, and the weights FSDP loads. |
| `--ref-load` | path | – | Reference model in `torch_dist` format (Megatron). |
| `--load` | path | – | Actor checkpoint to resume from. |
| `--save` | path | – | Actor checkpoint write directory. |
| `--save-interval` | int | – | Rollouts between saves. |
| `--custom-megatron-post-save-hook-path` | `<module>.<fn>` | – | Rank-0 callback after each checkpoint save. |
| `--model-name` | str | – | Set in multi-node to avoid `transformers` file-system race. |
| `--spec` | `<module> <fn>` | – | Plugin spec for custom architectures (e.g. `miles_plugins.models.qwen3_5 get_qwen3_5_spec`). |

### Rollout: data and batching

| Flag | Type | Default | Notes |
|---|---|---|---|
| `--prompt-data` | str | – | Path to a single JSONL file. |
| `--input-key` | str | `input` | JSONL key to `Sample.prompt`. |
| `--label-key` | str | `None` | JSONL key to `Sample.label`. |
| `--metadata-key` | str | `metadata` | JSONL key to `Sample.metadata`. |
| `--apply-chat-template` | flag | off | Apply tokenizer chat template. |
| `--rollout-shuffle` | flag | off | Shuffle prompts each rollout. |
| `--num-rollout` | int | – | Total rollout iterations. If unset, derived from dataset size. |
| `--rollout-batch-size` | int | – | Prompts per rollout. |
| `--n-samples-per-prompt` | int | `1` | Responses per prompt. |
| `--global-batch-size` | int | derived | Samples per optimizer step. |
| `--num-steps-per-rollout` | int | – | Optimizer steps per rollout. Alternative to `--global-batch-size`; setting one derives the other. |
| `--over-sampling-batch-size` | int | – | Oversample size for dynamic sampling (DAPO). |
| `--balance-data` | flag | off | Balance per-rank token count. |

### Rollout: sampling

| Flag | Type | Default | Notes |
|---|---|---|---|
| `--rollout-max-response-len` | int | – | Max tokens per response. |
| `--rollout-temperature` | float | `1.0` | Sampling temperature. |
| `--rollout-top-p` | float | `1.0` | Top-p truncation. |
| `--rollout-top-k` | int | `-1` | Top-k truncation (-1 disables). |
| `--rollout-stop` | str+ | – | Stop strings. |
| `--rollout-stop-token-ids` | int+ | – | Stop token IDs. |

### Eval

| Flag | Type | Default | Notes |
|---|---|---|---|
| `--eval-prompt-data` | str+ | – | One or more `name path` pairs. |
| `--eval-interval` | int | – | Rollouts between eval runs. |
| `--n-samples-per-eval-prompt` | int | `1` | Responses per eval prompt. |
| `--eval-max-response-len` | int | – | Max eval response length. Inherits from rollout if unset. |
| `--eval-temperature` | float | – | Eval temperature. Inherits from rollout if unset. |
| `--eval-top-p` | float | – | Eval top-p. Inherits from rollout if unset. |

### Performance

| Flag | Type | Default | Notes |
|---|---|---|---|
| `--tensor-model-parallel-size` | int | `1` | TP. |
| `--pipeline-model-parallel-size` | int | `1` | PP. |
| `--context-parallel-size` | int | `1` | CP. |
| `--expert-model-parallel-size` | int | `1` | EP (MoE). |
| `--expert-tensor-parallel-size` | int | `1` | TP within experts. |
| `--sequence-parallel` | flag | off | Enable Megatron sequence parallel. |
| `--use-dynamic-batch-size` | flag | off | Pack varlen samples. Recommended for varlen workloads. |
| `--max-tokens-per-gpu` | int | – | Token budget per micro-batch per GPU. Required when dynamic batching is on. |
| `--micro-batch-size` | int | `1` | Ignored when dynamic batching is on. |
| `--recompute-granularity` | enum | Megatron default | `full` or `selective`. |
| `--recompute-method` | enum | Megatron default | `uniform` or `block`. |
| `--recompute-num-layers` | int | Megatron default | Recompute chunk size. |
| `--gradient-checkpointing` | flag | off | FSDP equivalent of recompute flags. |
| `--fsdp-cpu-offload` | flag | off | FSDP: offload params, grads, optimizer state to CPU. |
| `--fsdp-cpu-backend` | str | `gloo` | FSDP: CPU backend for hybrid offload. |
| `--attn-implementation` | enum | `flash_attention_2` | FSDP only: `flash_attention_2`, `sdpa`, `eager`. |

### RL algorithm

| Flag | Type | Default | Notes |
|---|---|---|---|
| `--advantage-estimator` | enum | `grpo` | `grpo`, `gspo`, `ppo`, `reinforce_plus_plus`, `reinforce_plus_plus_baseline`, `on_policy_distillation` |
| `--use-kl-loss` | flag | off | Compute KL vs. reference. |
| `--kl-loss-coef` | float | `0.0` | KL weight in loss (0 means monitor). |
| `--kl-loss-type` | enum | `k1` | `k1`, `k2`, `k3`, `low_var_kl`. |
| `--entropy-coef` | float | `0.0` | Entropy bonus weight. |
| `--observe-training-entropy` | flag | off | Log detached training entropy when entropy bonus weight is zero. |
| `--eps-clip` | float | `0.2` | PPO/GRPO low clip. |
| `--eps-clip-high` | float | – | Asymmetric high clip. |
| `--use-tis` | flag | off | Truncated Importance Sampling. |
| `--use-routing-replay` | flag | off | Forward/backward routing consistency. |
| `--use-rollout-routing-replay` | flag | off | R3 — capture inference-side expert routing and replay it during training. |
| `--calculate-per-token-loss` | flag | off | Per-token loss reduction. |
| `--no-check-for-nan-in-loss-and-grad` | flag | off | Skip NaN/Inf guard (Megatron flag, debug only). |
| `--true-on-policy-mode` | flag | off | Strict on-policy: reject samples from a prior policy. |

### Optimizer

| Flag | Type | Default | Notes |
|---|---|---|---|
| `--optimizer` | enum | `adam` | `adam`, `sgd`. |
| `--lr` | float | `1e-6` | Learning rate. |
| `--lr-decay-style` | enum | `constant` | `constant`, `linear`, `cosine`. |
| `--lr-warmup-iters` | int | `0` | Warmup steps (Megatron flag). |
| `--min-lr` | float | `0` | Lower LR bound for decay schedules (Megatron flag). |
| `--weight-decay` | float | `0.1` | L2 weight decay. |
| `--adam-beta1` | float | `0.9` | |
| `--adam-beta2` | float | `0.98` | |
| `--clip-grad` | float | `1.0` | Grad clipping (Megatron flag). |
| `--optimizer-cpu-offload` | flag | off | Megatron CPU Adam (Megatron flag). |
| `--overlap-cpu-optimizer-d2h-h2d` | flag | off | Overlap D2H/H2D with compute (Megatron flag). |
| `--use-precision-aware-optimizer` | flag | off | Precision-aware optimizer path (Megatron flag). |

### Reward and filters

| Flag | Type | Default | Notes |
|---|---|---|---|
| `--rm-type` | enum | – | Built-in reward: `math`, `dapo`, `deepscaler`, `f1`, `gpqa`, `ifbench`, `remote_rm`, `random`. |
| `--rm-url` | str | – | Endpoint when `--rm-type remote_rm`. |
| `--group-rm` | flag | off | Batched reward computation. |
| `--custom-rm-path` | str | – | Custom reward function (see [Customization](/user-guide/customization)). |
| `--dynamic-sampling-filter-path` | str | – | Group filter (DAPO-style). |
| `--buffer-filter-path` | str | – | Buffer dequeue filter. |
| `--rollout-sample-filter-path` | str | – | Per-sample filter. |

### SGLang and router

| Flag | Type | Default | Notes |
|---|---|---|---|
| `--sglang-router-ip` | str | – | External router IP. Miles starts its own router if unset. |
| `--sglang-router-port` | int | – | External router port. |
| `--sglang-*` | passthrough | | Any flag accepted by `python -m sglang.launch_server` works with this prefix. |
| `--router-*` | passthrough | | Any flag accepted by the active router works with this prefix. |

Common `--sglang-*` flags:

```bash
--sglang-mem-fraction-static 0.8
--sglang-context-length 32768
--sglang-log-level INFO
--sglang-enable-ep-moe
--sglang-enable-dp-attention
--sglang-enable-deepep
--sglang-enable-overlap-schedule
--sglang-cuda-graph-backend-prefill       # prefill graphs default to disabled in colocate mode
```

### MTP / speculative decoding

| Flag | Type | Default | Notes |
|---|---|---|---|
| `--mtp-num-layers` | int | `0` | Number of MTP layers in the checkpoint. |
| `--enable-mtp-training` | flag | off | Train MTP alongside the policy. |
| `--mtp-loss-scaling-factor` | float | `0.2` | Weight of MTP loss. |

### Fault tolerance

| Flag | Type | Default | Notes |
|---|---|---|---|
| `--use-fault-tolerance` | flag | off | Enable rank-level recovery and heartbeats. |
| `--rollout-health-check-first-wait` | int | `0` | Grace period before heartbeats start. |
| `--rollout-health-check-interval` | int | `30` | Seconds between heartbeats. |
| `--rollout-health-check-timeout` | int | `30` | Heartbeat timeout. |

### Async / partial rollout

| Flag | Type | Default | Notes |
|---|---|---|---|
| `--partial-rollout` | flag | off | Resume aborted rollouts in the next iteration. |

### Logging

| Flag | Type | Default | Notes |
|---|---|---|---|
| `--use-wandb` | flag | off | Enable wandb. |
| `--wandb-project` | str | – | Project name. |
| `--wandb-group` | str | – | Group name. |
| `--log-interval` | int | `1` | Stdout log cadence (rollouts). |
| `--custom-rollout-log-function-path` | str | – | Custom train logger. |
| `--custom-eval-rollout-log-function-path` | str | – | Custom eval logger. |

### Profiling

| Flag | Type | Default | Notes |
|---|---|---|---|
| `--profile-target` | enum+ | `[train_overall]` | Which sub-loop to profile: `train_overall`, `train_actor`, `train_log_probs`. |
| `--use-pytorch-profiler` | flag | off | FSDP only: enable PyTorch profiler. |
| `--profile-step-start` | int | `10` | FSDP only: first step to profile. |
| `--profile-step-end` | int | `12` | FSDP only: last step to profile. |
| `--memory-snapshot-path` | str | `snapshot.pickle` | FSDP only: memory snapshot output. |
| `--tensorboard-dir` | str | – | FSDP only: TensorBoard output dir. |

### Debugging

| Flag | Type | Default | Notes |
|---|---|---|---|
| `--debug-rollout-only` | flag | off | Skip Megatron, only spin up SGLang. |
| `--debug-train-only` | flag | off | Skip SGLang, only spin up Megatron. |
| `--save-debug-rollout-data` | path | – | Pickle every rollout to disk. |
| `--load-debug-rollout-data` | path | – | Replay rollouts from disk (implies `--debug-train-only`). |
| `--deterministic-mode` | flag | off | Megatron deterministic mode. |

### Customization

See [Customization](/user-guide/customization) for the full catalog of `--*-path` flags
that replace or extend Miles's behavior.
