---
title: Argument Groups
description: The launch-script argument groups used by Miles recipes, with links to the flags that belong in each group.
---
Miles launch scripts are Python (`scripts/run_*.py`). The grouping is deliberately
boring: each script builds one string per operational concern, concatenates them into
`train_args`, and hands that to `execute_train`, which submits `train.py` or
`train_async.py` as a Ray job.

Use this page to decide where a flag belongs. Use the [CLI Reference](/user-guide/cli-reference)
when you need the full default and type for an individual flag. For how a script is laid
out and how to override one, see [Launch Script](/user-guide/launch-script).

| Group | Owns | Where it comes from |
|---|---|---|
| [model args](#model-args) | Architecture constants and plugin specs | `scripts/models/<megatron_model_type>.py`, spliced in by `execute_train` |
| [`ckpt_args`](#ckpt-args) | Actor, reference, HF tokenizer/config, save paths | Launch script |
| [`rollout_args`](#rollout-args) | Prompt data, sampling, reward, train/eval batch flow | Launch script |
| [`eval_args`](#eval-args) | Evaluation datasets and eval-only sampling overrides | Launch script |
| [`perf_args`](#perf-args) | Parallelism, recomputation, dynamic batching | Recipe defaults |
| [`grpo_args`](#grpo-args) | RL objective, KL, clipping, entropy, advantage estimator | Recipe defaults |
| [`optimizer_args`](#optimizer-args) | Learning rate, schedule, weight decay, Adam betas | Recipe defaults |
| [`sglang_args`](#sglang-args) | Rollout engine topology and `--sglang-*` passthrough | Deployment shape |
| [`misc_args`](#misc-args) | GPU layout, colocation, dropout, dashboard | Launch script |

The names above are the local variables every launcher uses; a script that needs no eval
simply leaves `eval_args` empty. Model args are the one group a launcher does not build:
it passes `megatron_model_type` to `execute_train`, which resolves the matching file
under `scripts/models/` and prepends those flags to the command line.

<a id="model-args"></a>
## Model args - architecture constants

Model args tell Megatron what model it is instantiating. Megatron cannot infer all
architecture details from a HuggingFace checkpoint, so each recipe loads a matching
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
## `ckpt_args` - checkpoint paths

`ckpt_args` wires the three model roles in a run:

| Role | Flag |
|---|---|
| HuggingFace directory for tokenizer, config, and SGLang boot | `--hf-checkpoint` |
| Frozen reference model for KL anchoring | `--ref-load` |
| Actor resume point | `--load` |
| Actor output directory | `--save` |

`--load` and `--save` usually point to the same directory. If `--load` has no
`latest_checkpointed_iteration.txt`, Miles warm-starts the actor from `--ref-load`.

<a id="rollout-args"></a>
## `rollout_args` - sampling and reward

`rollout_args` controls data entering the loop and how many samples each rollout
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
[four-knob invariant](/user-guide/concepts#the-four-knob-invariant).

<a id="eval-args"></a>
## `eval_args` - evaluation overrides

Evaluation reuses the rollout stack but usually runs with a different dataset and more
deterministic sampling.

Common entries:

| Concern | Flags |
|---|---|
| Cadence | `--eval-interval` |
| Dataset | `--eval-prompt-data` |
| Eval group size | `--n-samples-per-eval-prompt` |
| Eval-only generation | `--eval-max-response-len`, `--eval-top-p`, `--eval-temperature` |

Flags not set in `eval_args` inherit from `rollout_args`.

<a id="perf-args"></a>
## `perf_args` - parallelism and memory

`perf_args` controls how training is sharded and how activation memory is managed.

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
see [parallelism compatibility](/user-guide/training-backend#parallelism-compatibility) before changing
more than one dimension.

<a id="grpo-args"></a>
## `grpo_args` - RL objective

`grpo_args` controls the policy-gradient objective and the stability terms around it.

| Concern | Flags |
|---|---|
| Algorithm | `--advantage-estimator` |
| KL | `--use-kl-loss`, `--kl-loss-coef`, `--kl-loss-type` |
| Clipping | `--eps-clip`, `--eps-clip-high` |
| Entropy | `--entropy-coef`, `--observe-training-entropy` |
| Loss reduction | `--calculate-per-token-loss` |
| Precision/off-policy safety | `--use-tis` |

Zero-weight KL is recipe-specific. `--use-kl-loss --kl-loss-coef 0.00` still loads the
reference and logs KL; it does not remove the reference model.

<a id="optimizer-args"></a>
## `optimizer_args` - optimizer schedule

`optimizer_args` carries the optimizer choice and scalar schedule.

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
## `sglang_args` - rollout engine passthrough

`sglang_args` configures the inference side. Miles owns
`--rollout-num-gpus-per-engine`; everything prefixed with `--sglang-` is forwarded to
`python -m sglang.launch_server` after removing the prefix.

Common entries:

| Concern | Flags |
|---|---|
| Engine tensor parallelism | `--rollout-num-gpus-per-engine` |
| Engine memory | `--sglang-mem-fraction-static` |
| Context length | `--sglang-context-length` |
| MoE serving | `--sglang-ep-size`, `--sglang-moe-a2a-backend`, `--sglang-enable-dp-attention` |
| Debugging | `--sglang-log-level` |

SGLang parallelism is separate from trainer parallelism. For example,
`--rollout-num-gpus-per-engine` maps to the SGLang server's TP size, not Megatron's
`--tensor-model-parallel-size`.

<a id="misc-args"></a>
## `misc_args` - GPU layout and everything else

`misc_args` carries what the other groups do not: how many GPUs the actor gets, whether
it shares them with the rollout engines, the Megatron knobs a recipe pins once and never
tunes, and the optional dashboard.

Common entries:

| Concern | Flags |
|---|---|
| GPU layout | `--actor-num-nodes`, `--actor-num-gpus-per-node`, `--num-gpus-per-node` |
| Colocation | `--colocate` |
| Numerics pinned by the recipe | `--attention-dropout 0.0`, `--hidden-dropout 0.0`, `--attention-softmax-in-fp32`, `--accumulate-allreduce-grads-in-fp32` |
| Attention kernel | `--attention-backend` |
| Observability | `--use-miles-dashboard`, `--dump-details` |

Under `--colocate` the actor and the engines share the same GPUs and take turns, so
`--rollout-num-gpus` is ignored; see
[Training Backends](/user-guide/training-backend#3-choosing-the-gpu-layout).
