---
title: Argument Groups
description: The launch-script argument groups used by Miles recipes, with links to the flags that belong in each group.
---

# Argument Groups

Miles launch scripts are bash arrays. The grouping is deliberately boring: each array
owns one operational concern, then the script expands all arrays into `train.py` or
`train_async.py`.

Use this page to decide where a flag belongs. Use the [CLI Reference](cli-reference.md)
when you need the full default and type for an individual flag.

| Group | Owns | Typical source |
|---|---|---|
| [`MODEL_ARGS`](#model-args) | Architecture constants and plugin specs | `scripts/models/<family>.sh` |
| [`CKPT_ARGS`](#ckpt-args) | Actor, reference, HF tokenizer/config, save paths | Launch script |
| [`ROLLOUT_ARGS`](#rollout-args) | Prompt data, sampling, reward, train/eval batch flow | Launch script |
| [`EVAL_ARGS`](#eval-args) | Evaluation datasets and eval-only sampling overrides | Launch script |
| [`PERF_ARGS`](#perf-args) | Parallelism, recomputation, dynamic batching | Recipe defaults |
| [`GRPO_ARGS`](#grpo-args) | RL objective, KL, clipping, entropy, advantage estimator | Recipe defaults |
| [`OPTIMIZER_ARGS`](#optimizer-args) | Learning rate, schedule, weight decay, Adam betas | Recipe defaults |
| [`SGLANG_ARGS`](#sglang-args) | Rollout engine topology and `--sglang-*` passthrough | Deployment shape |

<a id="model-args"></a>
## MODEL_ARGS - architecture constants

`MODEL_ARGS` tells Megatron what model it is instantiating. Megatron cannot infer all
architecture details from a HuggingFace checkpoint, so each recipe sources a matching
file from `scripts/models/`.

Common entries:

| Flag family | Example |
|---|---|
| Transformer shape | `--num-layers`, `--hidden-size`, `--num-attention-heads` |
| Tokenizer/model dimensions | `--seq-length`, `--max-position-embeddings`, `--vocab-size` |
| Rotary and attention variants | `--rotary-base`, `--rotary-percent`, `--kv-channels` |
| MoE architecture | `--num-experts`, `--moe-router-topk`, `--moe-grouped-gemm` |
| Plugin specs | `--spec miles_plugins.models.qwen3_5 get_qwen3_5_spec` |

Keep these values aligned with the checkpoint's `config.json`. If one checkpoint in a
family changes rotary base, vocab padding, or normalization epsilon, override the
sourced defaults in the launch script.

<a id="ckpt-args"></a>
## CKPT_ARGS - checkpoint paths

`CKPT_ARGS` wires the three model roles in a run:

| Role | Flag |
|---|---|
| HuggingFace directory for tokenizer, config, and SGLang boot | `--hf-checkpoint` |
| Frozen reference model for KL anchoring | `--ref-load` |
| Actor resume point | `--load` |
| Actor output directory | `--save` |

`--load` and `--save` usually point to the same directory. If `--load` has no
`latest_checkpointed_iteration.txt`, Miles warm-starts the actor from `--ref-load`.

<a id="rollout-args"></a>
## ROLLOUT_ARGS - sampling and reward

`ROLLOUT_ARGS` controls data entering the loop and how many samples each rollout
produces.

| Concern | Flags |
|---|---|
| Prompt data | `--prompt-data`, `--input-key`, `--label-key`, `--apply-chat-template` |
| Rollout volume | `--rollout-batch-size`, `--n-samples-per-prompt`, `--num-rollout` |
| Training consumption | `--global-batch-size`, `--num-steps-per-rollout` |
| Sampling | `--rollout-temperature`, `--rollout-top-p`, `--rollout-max-response-len` |
| Reward | `--rm-type`, `--custom-rm-path` |
| Filtering | `--over-sampling-batch-size`, `--dynamic-sampling-filter-path` |

The rollout volume and training consumption must satisfy the
[four-knob invariant](concepts.md#the-four-knob-invariant).

<a id="eval-args"></a>
## EVAL_ARGS - evaluation overrides

Evaluation reuses the rollout stack but usually runs with a different dataset and more
deterministic sampling.

Common entries:

| Concern | Flags |
|---|---|
| Cadence | `--eval-interval` |
| Dataset | `--eval-prompt-data` |
| Eval group size | `--n-samples-per-eval-prompt` |
| Eval-only generation | `--eval-max-response-len`, `--eval-top-p`, `--eval-temperature` |

Flags not set in `EVAL_ARGS` inherit from `ROLLOUT_ARGS`.

<a id="perf-args"></a>
## PERF_ARGS - parallelism and memory

`PERF_ARGS` controls how training is sharded and how activation memory is managed.

| Concern | Flags |
|---|---|
| Tensor parallelism | `--tensor-model-parallel-size`, `--sequence-parallel` |
| Pipeline parallelism | `--pipeline-model-parallel-size` |
| Context parallelism | `--context-parallel-size` |
| Expert parallelism | `--expert-model-parallel-size`, `--expert-tensor-parallel-size` |
| Recomputation | `--recompute-granularity`, `--recompute-method`, `--recompute-num-layers` |
| Dynamic batching | `--use-dynamic-batch-size`, `--max-tokens-per-gpu` |

Megatron exposes TP, PP, CP, EP, and ETP, but not every product of those dimensions is
valid or worth using for every model. Start from the recipe's tested combination and
see [parallelism compatibility](usage.md#parallelism-compatibility) before changing
more than one dimension.

<a id="grpo-args"></a>
## GRPO_ARGS - RL objective

`GRPO_ARGS` controls the policy-gradient objective and the stability terms around it.

| Concern | Flags |
|---|---|
| Algorithm | `--advantage-estimator` |
| KL | `--use-kl-loss`, `--kl-loss-coef`, `--kl-loss-type` |
| Clipping | `--eps-clip`, `--eps-clip-high` |
| Entropy | `--entropy-coef` |
| Loss reduction | `--calculate-per-token-loss` |
| Precision/off-policy safety | `--use-tis` |

Zero-weight KL is recipe-specific. `--use-kl-loss --kl-loss-coef 0.00` still loads the
reference and logs KL; it does not remove the reference model.

<a id="optimizer-args"></a>
## OPTIMIZER_ARGS - optimizer schedule

`OPTIMIZER_ARGS` carries the optimizer choice and scalar schedule.

Common entries:

| Concern | Flags |
|---|---|
| Optimizer | `--optimizer` |
| Learning rate | `--lr`, `--min-lr`, `--lr-decay-style` |
| Adam | `--adam-beta1`, `--adam-beta2`, `--adam-eps` |
| Regularization | `--weight-decay`, `--clip-grad` |

Post-training is sensitive to large updates. Most recipes start near `1e-6` and use a
constant schedule unless the model page says otherwise.

<a id="sglang-args"></a>
## SGLANG_ARGS - rollout engine passthrough

`SGLANG_ARGS` configures the inference side. Miles owns
`--rollout-num-gpus-per-engine`; everything prefixed with `--sglang-` is forwarded to
`python -m sglang.launch_server` after removing the prefix.

Common entries:

| Concern | Flags |
|---|---|
| Engine tensor parallelism | `--rollout-num-gpus-per-engine` |
| Engine memory | `--sglang-mem-fraction-static` |
| Context length | `--sglang-context-length` |
| MoE serving | `--sglang-enable-ep-moe`, `--sglang-enable-dp-attention` |
| Debugging | `--sglang-log-level` |

SGLang parallelism is separate from trainer parallelism. For example,
`--rollout-num-gpus-per-engine` maps to the SGLang server's TP size, not Megatron's
`--tensor-model-parallel-size`.
