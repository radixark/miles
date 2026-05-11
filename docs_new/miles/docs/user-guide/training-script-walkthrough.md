---
title: Training Script Walkthrough
description: An annotated tour through every argument group in a Miles launch script, plus the feature modes you turn on when a recipe isn't enough.
---

# Training Script Walkthrough

A Miles launch script is plain bash — a sequence of `XXX_ARGS=( ... )` arrays handed
to `train.py` or `train_async.py`. This page walks through each group and then covers
the execution modes you turn on beyond the default recipe.

`scripts/run-glm4-9B.sh` is the reference script; other recipes follow the same shape.

## The eight argument groups

Every launch script assembles eight bash arrays, passes them as CLI flags, and hands
off to `train.py`:

| Array | Governs |
|---|---|
| [`MODEL_ARGS`](argument-groups.md#model-args) | Architecture constants (layers, hidden size, rotary base, ...) |
| [`CKPT_ARGS`](argument-groups.md#ckpt-args) | Filesystem paths for the actor / reference / save directory |
| [`ROLLOUT_ARGS`](argument-groups.md#rollout-args) | Prompt dataset, batch knobs, sampling parameters, reward type |
| [`EVAL_ARGS`](argument-groups.md#eval-args) | Eval dataset, cadence, sampling overrides for evaluation |
| [`PERF_ARGS`](argument-groups.md#perf-args) | Parallelism (TP/PP/CP/EP/ETP), recomputation, dynamic batching |
| [`GRPO_ARGS`](argument-groups.md#grpo-args) | RL algorithm, KL, clipping, entropy bonus, advantage estimator |
| [`OPTIMIZER_ARGS`](argument-groups.md#optimizer-args) | Learning rate, schedule, weight decay, Adam betas |
| [`SGLANG_ARGS`](argument-groups.md#sglang-args) | Engine TP, memory fraction, log level, `--sglang-*` passthrough |

---

## MODEL_ARGS — architecture constants

Megatron needs the model architecture hardcoded at launch because it cannot introspect
a HuggingFace checkpoint. Miles therefore sources a matching bash file from
`scripts/models/<family>.sh`:

```bash
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/models/glm4-9B.sh"
```

The sourced file sets `MODEL_ARGS=(--num-layers ... --hidden-size ... --rotary-base ...)`.

<Warning>

**Architecture parameters are not self-validating.** Two checkpoints from the same family can ship different `--rotary-base`, vocab
padding, or normalization epsilon. Diff the `config.json` against the file in
`scripts/models/` before you run, and override anything that drifts:

```bash
source "${SCRIPT_DIR}/models/glm4-9B.sh"
MODEL_ARGS+=(--rotary-base 10000)
```

</Warning>

## CKPT_ARGS — paths

The three roles — actor, frozen reference, HuggingFace directory — are defined in [Core Concepts](concepts.md#the-four-objects). Here they map to four flags:

```bash
CKPT_ARGS=(
   --hf-checkpoint /root/GLM-Z1-9B-0414          # tokenizer, config, SGLang init
   --ref-load      /root/GLM-Z1-9B-0414_torch_dist   # frozen ref (KL anchor)
   --load          /root/GLM-Z1-9B-0414_miles/   # actor resume point
   --save          /root/GLM-Z1-9B-0414_miles/   # where checkpoints are written
   --save-interval 20                            # rollouts between writes
)
```

`--load` and `--save` usually point at the same directory so the run is
restart-idempotent: the trainer reads whatever was last written there. When `--load` is
empty or missing `latest_checkpointed_iteration.txt`, the actor is warm-started from
`--ref-load` instead.

## ROLLOUT_ARGS — where data comes from and how much flows

Every Miles iteration alternates between **sampling** (rollout) and **consumption**
(training). Four knobs govern the balance.

**Sampling side**

- `--rollout-batch-size` — prompts drawn per rollout.
- `--n-samples-per-prompt` — responses generated per prompt (used as the GRPO group).

Their product is the total sample count produced each rollout.

**Consumption side**

- `--global-batch-size` — samples used per optimizer step.
- `--num-steps-per-rollout` — optimizer steps per rollout. Leave at `1` for strict
  on-policy behaviour; raise it for off-policy reuse of rollout batches.

Their product is the total sample count consumed each rollout.

These two products must be equal — that's the [four-knob invariant](concepts.md#the-four-knob-invariant). Set three sides; Miles fills in the fourth. Set all four and Miles validates the equation — inconsistent values abort early.

**Outer loop**

- `--num-rollout` — total sample/train iterations.

A compact example:

```bash
ROLLOUT_ARGS=(
   --prompt-data /root/dapo-math-17k/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle

   --rm-type deepscaler

   # the four-knob relationship
   --num-rollout 3000
   --rollout-batch-size 16
   --n-samples-per-prompt 8
   --num-steps-per-rollout 1
   --global-batch-size 128

   --rollout-max-response-len 8192
   --rollout-temperature 1

   --balance-data                  # equalize tokens across DP ranks
)
```

<Note>

**Optimizer step vs. weight sync.** `--num-steps-per-rollout` counts calls to `optimizer.step()`, not the weight
handshake between trainer and SGLang. The latter happens exactly once per rollout,
regardless of how many optimizer steps fired in between.

</Note>

## EVAL_ARGS — a strict subset of rollout

Evaluation reuses the rollout machinery but lets you override sampling behaviour so
that eval is deterministic and comparable across runs.

```bash
EVAL_ARGS=(
   --eval-interval 5                                # rollouts between eval runs
   --eval-prompt-data aime /root/aime-2024/aime-2024.jsonl
   --n-samples-per-eval-prompt 16                   # eval group size
   --eval-max-response-len 16384                    # longer than train rollout
   --eval-top-p 1                                   # disable nucleus
)
```

Flags not overridden here inherit their values from `ROLLOUT_ARGS`.

## PERF_ARGS — parallelism and memory

This group controls how the model is sharded across GPUs and how much activation
memory is recomputed vs. stored. Miles forwards Megatron's parallelism flags
untouched and adds two of its own.

```bash
PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 2
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 4608
)
```

A few notes worth committing to memory:

- **Always pair `--tensor-model-parallel-size > 1` with `--sequence-parallel`.** The
  sequence-parallel pass reclaims the activation memory TP leaves behind.
- **`--use-dynamic-batch-size` overrides `--micro-batch-size`.** When dynamic batching
  is active, Miles packs variable-length samples into the closest fit under
  `--max-tokens-per-gpu`. A sample whose length already exceeds the cap takes a whole
  micro-batch by itself — that batch still exceeds the cap and may OOM, so keep
  `--rollout-max-response-len` ≤ `--max-tokens-per-gpu`.
- **Under context parallel, the budget is shared.** A CP group of size `N` jointly
  processes up to `N × max_tokens_per_gpu` tokens per micro-batch. Size CP before
  tuning `max-tokens-per-gpu`.
- **Loss correctness is preserved.** Miles packs with proper attention masks and
  per-sample / per-token loss reductions — dynamic batching never changes the gradient
  value, only the throughput.

## GRPO_ARGS — the RL objective

`GRPO_ARGS` is the only group that carries RL semantics. The defaults encode vanilla
GRPO with a DAPO-style asymmetric clip.

```bash
GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)
```

A few design choices become visible here:

- **KL as a monitor vs. KL in the loss.** `--use-kl-loss` always loads the reference
  model and computes the divergence — its weight in the loss is controlled separately
  by `--kl-loss-coef`. Setting the coefficient to `0.0` turns KL into a pure
  observability signal, which is often what you want for early experiments.
- **`--advantage-estimator` covers more than GRPO.** `gspo`, `reinforce_plus_plus`,
  `reinforce_plus_plus_baseline`, `ppo`, and `on_policy_distillation` are all
  drop-in replacements.
- **Per-sample vs. per-token loss.** The default reduction is per-sample mean:
  `mean(sum(sample_i) / len(sample_i))`. Add `--calculate-per-token-loss` to switch to
  `sum(sum(sample_i)) / sum(len(sample_i))` — the correct choice for SFT-style loss
  or when you want length-proportional weighting.
- **`--use-tis` is the numerical safety belt.** Switch it on when rollout and trainer
  operate at different precisions or when you explicitly want off-policy reuse. See
  the R3 deep dive in [Rollout Routing Replay (R3)](../advanced/miles-router.md).

## OPTIMIZER_ARGS — nothing surprising

Post-training is unusually sensitive to optimizer settings: the model is already in a
good basin and large updates destabilize it.

```bash
OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)
```

A rule of thumb: start at `lr = 1e-6` with a constant schedule. If the loss plateaus
early, investigate the reward signal before raising the learning rate — in most cases
the reward pipeline collapsed (same score for every sample) rather than the optimizer
stalling.

## SGLANG_ARGS — passthrough to the rollout engine

The only Miles-owned flag here is `--rollout-num-gpus-per-engine`, which corresponds
loosely to SGLang's `tp_size`. Everything else prefixed with `--sglang-` is forwarded
verbatim.

```bash
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 2
)
```

Common additions that don't ship in the canonical recipe:

| Flag | When to add it |
|---|---|
| `--sglang-mem-fraction-static 0.7` | Colocated mode; Megatron needs headroom after init. |
| `--sglang-context-length 32768` | Rollout max length exceeds the model's `config.json`. |
| `--sglang-enable-ep-moe` | MoE models. |
| `--sglang-enable-dp-attention` | Long prompts on MoE. |
| `--sglang-log-level INFO` | Debugging. |

Miles multiplexes across multiple SGLang engines through a router. When
DP-attention is off, the effective `dp_size` is derived from
`rollout-num-gpus / rollout-num-gpus-per-engine`.

---

# Execution modes

The eight argument groups describe **what** you're training. The next set of sections
describe **how** the training runs — the execution modes that flip Miles from its
default one-rollout-then-one-train cadence into something more interesting.

## Synchronous vs. asynchronous rollout

In the default cadence the trainer blocks on rollout: `generate()` returns, then
`train_step()` fires, then the next rollout kicks off. Every iteration's wall-clock
time is the sum of the two.

Async rollout turns the cadence into two concurrent loops. A background worker keeps
`--rollout-batch-size` generations in flight at all times and pushes completed samples
into a queue; the trainer drains the queue, steps, and syncs weights. Per-iteration
wall time drops to roughly `max(rollout_time, train_time)`.

Enable it with two changes to the launch script:

```diff
- python3 train.py ...
+ python3 train_async.py ...
+   --rollout-function-path fully_async_rollout.generate_rollout_fully_async
```

| Mode | Per-iteration latency | Throughput | When to use |
|---|---|---|---|
| Sync *(default)* | Lower | Lower overall | Strict on-policy, debugging |
| Async | Higher | Up to 2× | Rollout-bound jobs, long runs |

See the [Fully Async Rollout example](../examples/fully-async.md) for the full
walkthrough including the worker implementation.

## Colocation: share GPUs or don't

In the default disaggregated layout, training and inference claim separate GPUs
through Ray. The simplest form is:

```bash
ray job submit ... -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 4 \
   --rollout-num-gpus 4 \
   ...
```

Eight GPUs, half training, half inference, running concurrently.

When GPUs are scarce, you can **colocate** actor and rollout onto the same set of
devices. Miles time-slices between them:

```bash
ray job submit ... -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 8 \
   --colocate \
   ...
```

`--rollout-num-gpus` is ignored under `--colocate`; the two phases always share the
entire allocation.

<Warning>

**Leave SGLang some headroom.** Megatron reserves VRAM during initialization and only releases it once the first
offload completes. If SGLang is sized to the full device capacity, the init
collision produces an OOM before training ever starts. Drop
`--sglang-mem-fraction-static` to **0.8** (further if memory is still tight).

</Warning>

<Note>

**What `--colocate` flips on.** Setting `--colocate` also enables (unless you've set them explicitly):

- `--offload-train` — train state offloads to CPU between phases
- `--offload-rollout` — rollout state offloads to CPU between phases
- `--sglang-disable-piecewise-cuda-graph` — avoids NVLS OOM in colocate mode

</Note>

## Dynamic sampling (DAPO-style filtering)

A common failure mode of GRPO is *reward homogeneity*: every trajectory in a group gets
the same score, the advantage is zero, and the gradient goes flat. DAPO addresses this
by **oversampling** and throwing away groups that lack reward variance.

Miles exposes the same capability through two flags:

```bash
ROLLOUT_ARGS+=(
   --over-sampling-batch-size 64
   --dynamic-sampling-filter-path \
     miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std
)
```

When `--over-sampling-batch-size` exceeds `--rollout-batch-size`, Miles draws the
larger batch, runs generation and reward scoring asynchronously, and applies the
filter function as results arrive. Groups that survive the filter enter the training
queue; groups that fail are discarded.

The shipped filter checks that reward standard deviation is strictly positive:

```python
def check_reward_nonzero_std(args, samples: list[Sample], **kwargs):
    rewards = [sample.reward for sample in samples]
    return torch.tensor(rewards, dtype=torch.float).std() > 0.0
```

The system watches `remaining_batch_size` — the count of usable groups still needed to
fill the rollout. Whenever the filter discards enough samples that the count drops
below `--rollout-batch-size`, Miles automatically kicks off another oversampling wave.
The mechanism is self-healing: a strict filter just means more oversampling rounds,
not a stuck trainer.

## Partial rollout: reclaim aborted work

Dynamic sampling implies that *some* in-flight generations will be abandoned. Without
care, the compute invested in those half-finished trajectories is lost.

`--partial-rollout` flips on a buffer that retains partial samples and resumes their
generation during the next rollout. The buffer dequeue policy is itself pluggable via
`--buffer-filter-path`; the default is a first-in-first-out pop:

```python
def pop_first(args, rollout_id, buffer, num_samples):
    num_to_pop = min(len(buffer), num_samples)
    samples = buffer[:num_to_pop]
    del buffer[:num_to_pop]
    return samples
```

Each entry in the buffer carries its original `sample.metadata` (including the
rollout ID that first launched it), which is usually enough to reason about staleness
if you want a stricter eviction policy.

<Tip>

**Partial rollout and off-policy correction.** Samples that re-enter the trainer from the buffer were generated under an older
policy than the one currently being trained. If you enable `--partial-rollout`
together with aggressive reuse (`--num-steps-per-rollout > 1`), pair it with
`--use-tis` to keep the gradient well-behaved.

</Tip>

## BF16 training with FP8 inference

The simplest way to exploit FP8 on Hopper-class hardware is to leave the trainer in
BF16 and serve FP8 weights to SGLang. Miles supports this path without any code
changes — only checkpoint pointers differ.

Download the FP8 weights alongside the BF16 originals:

```bash
hf download Qwen/Qwen3-4B-FP8 --local-dir /root/Qwen3-4B-FP8
```

Then swap the `--hf-checkpoint` pointer to the FP8 directory while leaving the
Megatron side unchanged:

```bash
CKPT_ARGS=(
   # SGLang picks up the FP8 weights from here
   --hf-checkpoint /root/Qwen3-4B-FP8

   # Megatron still trains from the BF16-derived torch_dist checkpoint
   --ref-load      /root/Qwen3-4B_torch_dist
)
```

The trainer continues to operate in BF16; SGLang will cast the BF16 weights it
receives via P2P sync into FP8 before running them. This is the lowest-friction way
to take the inference-side speedup, but it does introduce the precision mismatch
between rollout and trainer that R3 and TIS were designed to absorb.

<Warning>

**Do not point `--ref-load` at the FP8 directory.** The reference model must remain BF16. Replacing it with FP8 weights changes the
KL anchor silently and makes the loss curve incomparable to earlier runs.

</Warning>

For end-to-end FP8 (trainer and inference at bit-identical precision), see
[Low Precision RL](../advanced/fp8-low-precision.md). For INT4 quant-aware
training, see [INT4 QAT](../advanced/int4-qat.md).

---

## Next

- [Configuration](cli-reference.md) — the same material organized as a flag-by-flag
  reference.
- [Server Arguments](cli-reference.md) — the complete CLI surface.
- [Customization](customization.md) — the twenty-plus Python extension points.
- [Training Backends](usage.md) — Megatron vs FSDP and each one's plumbing.
