---
title: Training Backends
description: The contract Megatron-LM and FSDP both implement, which one to pick, and how to configure parallelism, GPU layout, offload, and checkpoints for each.
---
In Miles a training backend is one class: a `TrainRayActor` subclass that owns the model on
the GPU. `--train-backend` decides which one `miles/ray/train/actor_factory.py` instantiates
on every trainer rank, and there are two choices.

| Value | Class | What it is | Default |
|---|---|---|---|
| [`megatron`](#megatron-lm) | `MegatronTrainRayActor` | Megatron-LM: five parallel dimensions, `torch_dist` checkpoints | ✅ |
| [`fsdp`](#fsdp) | `FSDPTrainRayActor` | The model's own HuggingFace implementation under PyTorch FSDP2 | |

Whichever you pick, the rest of the job talks to it through the same handful of methods, and
that short list is the whole contract between a backend and everything else in Miles:

| Method | What the backend has to do |
|---|---|
| `init` | Build the model, optimizer and parallel layout after the shared base has set up the process group and the device |
| `train` | Consume one rollout's data and take the optimizer steps for it |
| `update_weights` | Push the freshly trained weights into the SGLang engines |
| `save_model` | Write a checkpoint, in whatever format this backend uses |
| `sleep` / `wake_up` | Move the model and optimizer off the GPU and back, so a colocated SGLang engine can use the memory in between |

That is also why switching backends does not touch the rest of your launch script. Rollout,
reward, eval, the RL algorithm and the SGLang engine all sit above this line, and so does
the GPU layout: **disaggregated** by default, where trainer and engines own separate GPUs,
or **colocated** with `--colocate`, where they share GPUs and `sleep` / `wake_up` hand the
memory back and forth.

What does change is everything below the line, which is what the rest of this page is about.

## Which one do you want?

**Use Megatron-LM for large models and for anything that needs real parallelism.** It is the
recommended backend, the one every recipe in [Models](/models/index) is tuned for, and the
only one that can split a model *inside* itself. If the model is a 100 B+ MoE, if the job
spans racks, or if fitting it at all depends on tensor / pipeline / expert parallelism, this
is the answer.

**Use FSDP when you want the HuggingFace implementation trained verbatim.** It loads a HF
directory as-is, with no conversion step and no architecture flags to write, which makes it
the fast path for bringing up a new architecture, for checking trainer numerics against the
HF reference, and for models that fit under data parallelism alone.

The rest follows from that split:

| | Megatron-LM | FSDP |
|---|---|---|
| Model splitting | TP × PP × CP × EP × ETP, plus DP | `dp_replicate` × `dp_shard` |
| Model input | `torch_dist` checkpoint (offline conversion step) | HF directory, loaded as-is |
| Architecture definition | `scripts/models/<megatron_model_type>.py` plus a Megatron spec for anything non-standard | HF `config.json`, plus an optional adaptation spec |
| Checkpoints written | Megatron `torch_dist` | PyTorch Distributed Checkpoint |
| Activation recompute | `--recompute-granularity / method / num-layers` | `--gradient-checkpointing` |
| Optimizer on CPU | `--optimizer-cpu-offload` | `--fsdp-cpu-offload` |
| Offload beyond host RAM | `--offload-train-target disk`, `--stream-optimizer-state-to-disk` | Not supported |
| Attention backend | Chosen by Megatron Core | `--attn-implementation` |
| LoRA | Supported | Not supported |

---

## Megatron-LM

Configuring this backend is a handful of decisions, in this order: what the architecture is,
how to split it, where it sits relative to the rollout engines, how to fit it in memory,
where the weights come from, and what you want to hook into.

### 1. Describing the architecture

You do not re-declare Megatron's flags to Miles. Miles imports Megatron's whole argument
surface at launch:

```python
from megatron.training.arguments import parse_args
```

so every Megatron flag your checkpoint needs (`--kv-channels`, `--rotary-base`,
`--moe-grouped-gemm`, and the rest) already works. Miles then threads its own flags in
through an `extra_args_provider` (`get_miles_extra_args_provider` in
`miles/utils/arguments.py`), which is why Miles and Megatron flags share one CLI.

That import is also why you export the Megatron source before launching:

```bash
export PYTHONPATH=/root/Megatron-LM
```

In a launch script the architecture flags come from `scripts/models/<family>.py`,
selected by the `megatron_model_type` the script hands to `execute_train`. Most models need nothing beyond the stock
`--num-layers / --hidden-size / ...`. For the ones that do, see
[bringing in a new architecture](#going-deeper-bringing-in-a-new-architecture) below.

### 2. Choosing the parallelism

<a id="parallelism-compatibility" />

Megatron exposes five useful parallel dimensions, but you can't combine them in arbitrary
ways. Only a subset of TP × PP × CP × EP × ETP combinations is actually supported, and some
legal combinations are slower than the recipe baseline. **Start from the model recipe's
tested combination, then change one dimension at a time.**

| Dimension | Use it for | Compatibility notes |
|---|---|---|
| TP | Shard dense matrix multiplications inside each layer | When `--tensor-model-parallel-size` is set above 1, also pass `--sequence-parallel` unless the recipe says otherwise. |
| PP | Split layers across pipeline stages | Combines with TP and CP, but changes micro-batch scheduling and checkpoint layout. |
| CP | Split long sequences across ranks | Useful for long context; size token budgets as `CP x max_tokens_per_gpu`. |
| EP | Distribute MoE experts across ranks | MoE-only. Keep trainer EP and SGLang EP as separate choices. |
| ETP | Tensor-parallelize expert MLPs | MoE-only. Use it only when the recipe enables it or when EP alone cannot fit the experts. |

Do not assume TP, CP, EP and ETP can all be raised independently for a new model. The exact
set of supported combinations depends on the Megatron Core kernels and model spec in use.
[Argument Groups](/user-guide/argument-groups#perf-args) lists the flags that belong in
`perf_args`.

### 3. Choosing the GPU layout

Parallelism says how the trainer splits the model. This says where the trainer sits relative
to the SGLang engines, and there are two answers.

**Disaggregated is the default.** The trainer takes `--actor-num-nodes` x
`--actor-num-gpus-per-node` GPUs, the engines take `--rollout-num-gpus` more, and the two
sets do not overlap. Nobody has to move: both halves stay resident on their own GPUs for the
whole run, so `--offload-train` / `--offload-rollout` default off and no phase pays an
offload cost. It is also the layout that lets the two halves actually run at the same time,
which is what [Fully Async Rollout](/user-guide/fully-async) and `train_async.py` are for.
Under the synchronous loop in `train.py` the phases still alternate, so each set of GPUs is
idle while the other works.

```bash
# 8 GPUs training, 8 more generating
--actor-num-nodes 1 --actor-num-gpus-per-node 8 \
--rollout-num-gpus 8 --rollout-num-gpus-per-engine 2
```

**Colocated shares one set of GPUs.** `--colocate` puts the engines on the training GPUs and
the two take turns: generate, offload the engine, train, offload the trainer, repeat. It is
the right default when GPUs are the scarce resource, since the same 8 GPUs do both jobs
instead of standing idle during the other phase.

```bash
--colocate \
--actor-num-nodes 1 --actor-num-gpus-per-node 8 \
--rollout-num-gpus-per-engine 2 \
--sglang-mem-fraction-static 0.8
```

Three things follow from `--colocate` that are worth knowing before you use it:

- `--rollout-num-gpus` is ignored and reconciled to `actor_num_gpus_per_node x
  actor_num_nodes`, since the engines are on the training GPUs by definition.
- `--offload-train` and `--offload-rollout` both turn on, which is what makes the taking of
  turns possible. That is the memory story in the next section.
- The trainer reserves HBM at init before SGLang starts, so `--sglang-mem-fraction-static`
  has to come down, typically to 0.8 or lower. Miles also defaults
  `--sglang-cuda-graph-backend-prefill=disabled` here to avoid an NVLS OOM.

The layout also decides how `update_weights` gets the weights across. Colocated, the engine
is on the same device, so the actor hands over CUDA IPC handles and nothing crosses the
network. Disaggregated, the weights have to travel, and `--update-weight-transfer-mode`
picks how: `broadcast` (the default) sends them over the training-to-engine process group,
`p2p` uses [RDMA point-to-point](/advanced/p2p-weight-transfer), and
[`disk-delta`](/advanced/disaggregated-rollout) publishes only the bytes that changed since
the last sync for each engine to pull.

On a node with fewer than 8 usable GPUs, set `--num-gpus-per-node` too, otherwise the
rollout side still assumes 8. And `--fully-async` cannot be colocated: its whole point is
that rollout keeps generating while the trainer steps, which requires separate GPUs.

### 4. Fitting it in memory

Parallelism decides how the model is divided; this decides what is allowed to sit in HBM at
all. Four things compete for it: parameters, gradients, optimizer state, and activations.
On bf16 training the optimizer state is the heavy one, at 12 bytes per parameter for the
fp32 master copy plus the two Adam moments, against 2 bytes for a bf16 parameter. Data
parallelism divides that state, so a run with GPUs to spare may need none of what follows,
and a run at DP=1 may need all of it.

Two of the knobs apply to any layout, and the rest exist only because a colocated engine
wants the GPU back.

#### Either layout

**Activations** are the first thing to trade, because recompute is cheap and predictable.
Every recipe passes some form of:

```bash
--recompute-granularity full --recompute-method uniform --recompute-num-layers 1
```

**The optimizer step can run on the CPU.** `--optimizer-cpu-offload` keeps the master
weights and moments in host memory and runs Adam there, and
`--overlap-cpu-optimizer-d2h-h2d` hides the copies behind compute. Recipes that use it
usually add `--use-precision-aware-optimizer`, which lets Megatron hold narrower optimizer
state. Note the interaction with rematerialization below: precision-aware on the GPU stores
masters as int16 remainders inside TE FusedAdam, so there is nothing standalone left to
rebuild from.

#### Colocated only

Everything from here down hangs off `--offload-train`, which is on precisely because the
engine needs the same HBM during generation. It is what `sleep` / `wake_up` do, it is turned
on for you by `--colocate`, and in a disaggregated run there is nothing to make room for, so
none of it applies.

| Flag | Effect |
|---|---|
| `--offload-train` / `--offload-rollout` | Which side is offloaded during the other's phase. Both implied by `--colocate`. |
| `--offload-train-target cpu` | Default: the paused actor is backed up in pinned host memory. |
| `--offload-train-target disk` | For when host RAM cannot hold that copy either: stream it to node-local NVMe instead, through a bounded pinned buffer (`--offload-train-disk-dir`, `--offload-train-disk-chunk-mb`). Megatron backend only. |
| `--rematerialize-param-from-master-weight` | Drop the actor's parameter backup during rollout and rebuild it from the optimizer's master weights on the next step. Saves 2 bytes per parameter per rank of host memory on bf16 training. Asserts `--colocate` plus the `cpu` target. |

**If the optimizer state does not fit while the step itself runs**, offloading the actor
cannot help, because pause and resume happen at phase boundaries and everything is resident
again by the time Adam launches. That case is what streaming addresses:

```bash
--offload-train --offload-train-target disk \
--stream-optimizer-state-to-disk \
--offload-train-disk-dir /scratch/miles_offload
```

The fp32 masters and Adam moments live in per-bucket files on NVMe, and the step brings in
one bucket at a time, so peak residency is one bucket instead of the whole state. At the
default `fp32` moment dtype it is bit-identical to keeping the state on the GPU and costs
disk traffic every step; `--stream-optimizer-state-moment-dtype bf16` cuts the volume by a
third. It requires the `disk` target and excludes `--optimizer-cpu-offload`.

[Disk Offload](/advanced/disk-offload) has the full picture for both, including the
same-topology resume limit, what checkpointing costs, and measured sleep / wake numbers.

### 5. Getting weights in and out

Megatron trains from its own `torch_dist` format: `.distcp` files that are
parallelism-agnostic, so you can change TP / PP / EP later without re-converting. Convert
once, up front:

```bash
MODEL_ARGS_LINE="$(python3 miles/utils/external_utils/model_args_utils.py <family>)" || exit 1
read -ra MODEL_ARGS <<< "${MODEL_ARGS_LINE}"
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
   ${MODEL_ARGS[@]} \
   --hf-checkpoint /root/<model> \
   --save          /root/<model>_torch_dist
```

For models larger than a single node, drive the converter with
`torchrun --nnodes=<N> --nproc-per-node=8 ...`. Each recipe page lists the exact command.

What the run then writes looks like this:

```text
/ckpt/
├── latest_checkpointed_iteration.txt
├── iter_0000100/
│   ├── _0_0.distcp
│   └── ...
├── iter_0000200/
└── ...
```

Always pass the **parent** directory to `--load`, never a specific `iter_*`. The loader
reads `latest_checkpointed_iteration.txt` to pick the step.

**Saving on demand.** `--save-trigger-sentinel <path>` forces a save from outside the
process, independent of `--save-interval`:

```bash
# trigger a save and wait until the checkpoint is on disk
touch /path/to/save_now && until [ ! -e /path/to/save_now ]; do sleep 5; done
```

A request fired at any moment during an iteration is consumed at that iteration's save
point. The checkpoint is written with `force_sync=True` (so async saves finalize first), and
only then is the sentinel deleted, which is why "file gone" means "checkpoint durable on
disk". If the job crashes mid-save the sentinel survives, so the request stays pending for
the next run. Requires `--save`.

### 6. Hooking into the loop

Three extension points override Megatron behavior without forking it:

| Flag | Runs |
|---|---|
| `--custom-megatron-init-path` | After Megatron initialization |
| `--custom-megatron-before-log-prob-hook-path` | Before every log-probability computation |
| `--custom-megatron-before-train-step-hook-path` | Before every training step |

Typical uses: mixing in an auxiliary loss, instrumenting per-step metrics, clipping weights
surgically. See [Customization](/user-guide/customization#megatron-hooks).

### Going deeper: bringing in a new architecture

Post-training runs on released checkpoints, so this is rarely your problem. When a model does
need a custom module, Miles embeds the model's official HuggingFace module inside Megatron's
scheduling rather than patching Megatron: a spec function under `miles_plugins/models/` is
selected with `--spec <module> <function>`, a bridge under `miles_plugins/mbridge/`
reconciles the parameter layouts, and parameters that must stay fp32 through Megatron's bf16
cast are tagged with `mark_param_dtype` from
`miles/backends/megatron_utils/fp32_param_utils.py`. The model configs in `scripts/models/`
that pass `--spec` are the worked examples.

---

## FSDP

The FSDP backend lives at `miles/backends/fsdp_utils/`. One idea explains the whole thing:
**nothing about the model is re-expressed for the trainer.** Architecture comes from the
HuggingFace `config.json`, weights load through `AutoModelForCausalLM.from_pretrained()`,
and sharding, the distributed optimizer and mixed precision all come from PyTorch FSDP2
rather than from Miles.

So there is no conversion step, no architecture flag block, and no spec to write for a model that
`transformers` already implements. The bill comes due on parallelism, which is why large
models and complex layouts belong on [Megatron-LM](#megatron-lm).

### 1. Pointing it at a model

```bash
--train-backend fsdp \
--hf-checkpoint /root/models/<model>
```

`--hf-checkpoint` is the whole model input: tokenizer, config and weights. Layer count is
read from the HF config, so Megatron's architecture flags (`--num-layers`, `--hidden-size`,
`--spec`, and the rest of `scripts/models/`) simply do not apply here.

### 2. Sharding it

This backend is pure data parallel. `miles/backends/fsdp_utils/parallel.py` builds a single
device mesh with two dimensions:

| Dimension | How you set it | What it does |
|---|---|---|
| `dp_replicate` | `--dp-replicate-size` | Replica count for FSDP2 hybrid sharding. Parameters are replicated across replicas, sharded within one. |
| `dp_shard` | derived | Whatever is left: `world_size / dp_replicate`. This is the dimension FSDP2 actually shards parameters, gradients and optimizer state over. |

The default, `dp_replicate=1`, means one flat shard group over every training rank. Tensor,
pipeline, context, expert and expert-tensor parallelism are all fixed at size 1 in the FSDP
`ParallelState`, so the model has to fit within those two dimensions.

<Note>

Context parallelism is not available here. `--context-parallel-size` above 1 is rejected in
argument validation (`miles/utils/arguments.py`); the mesh has no CP dimension to build.

</Note>

The mesh is checked before anything is built: `world_size` must divide by
`--dp-replicate-size`, otherwise the run fails in argument validation instead of deep inside
mesh construction.

Memory, once the layout is set:

| Flag | Effect |
|---|---|
| `--gradient-checkpointing` | Recompute activations. This backend's `--recompute-*`. |
| `--fsdp-cpu-offload` | Offload parameters, gradients and optimizer state to CPU. The optimizer step runs there. |
| `--fsdp-cpu-backend gloo` | CPU process-group backend used by the offload path. |

Under `--colocate` this backend also implements `sleep` / `wake_up` by moving the model and
the optimizer to host memory and back, gated on `--offload-train`. The deeper offload
targets are Megatron-only: both `--offload-train-target disk` and
`--stream-optimizer-state-to-disk` assert the Megatron backend.

### 3. Precision

- bf16 by default; `--fp16` switches the compute dtype.
- An fp32 master copy of the weights is kept by default, which is what makes the
  trainer to rollout weight sync bit-exact. `--no-keep-fp32-master` trades it for memory when
  you do not need that guarantee.
- `--attn-implementation` picks the `transformers` attention backend: `flash_attention_2` by
  default, with `flash_attention_3`, `sdpa` and `eager` passed straight through.
- An architecture with fussier numerics can register its own policy, see
  [when an HF model needs help](#going-deeper-when-an-hf-model-needs-help).

### 4. Checkpoints

`--save` writes PyTorch Distributed Checkpoint directories, one each for model, optimizer
and LR scheduler, plus a `latest_checkpointed_iteration.txt` tracker. So `--load` takes the
**parent** directory exactly like the Megatron backend does. These are FSDP-backend
checkpoints, not `torch_dist` ones, and the two formats are not interchangeable.

### Limits

<Warning>

**No TP / PP / CP / EP.** The model must fit under `dp_replicate` × `dp_shard`.

**No LoRA.** [LoRA](/advanced/lora) is Megatron-only.

</Warning>

For large models, multi-rack jobs, or any recipe whose fit depends on tensor, pipeline or
expert parallelism, use [Megatron-LM](#megatron-lm).

### Going deeper: when an HF model needs help

Any HuggingFace causal LM loads. Some need small corrections around the edges: a weight
layout SGLang does not expect, a stateful layer that must be reset per document, a class
that needs patching before construction. Those live in
`miles/backends/fsdp_utils/adaptations/specs/`, one file per architecture, and an
architecture that needs none of them registers nothing.

| Hook | What it fixes |
|---|---|
| `register_param_transform` | Train to rollout parameter rename / reshape at weight sync, for example unfusing batched experts into the per-expert names SGLang expects |
| `register_model_patch` | Config-time patch of a `transformers` class |
| `register_model_instance_patch` | Post-construction patch of one model instance |
| `register_packing_patch` | Per-document state reset under THD sequence packing, for stateful layers such as Gated-Delta-Net and Mamba2 hybrids |
| `register_post_load_fixup` | Repair weights `from_pretrained()` clobbered |
| `register_precision_policy` | Model-specific FSDP compute / autocast policy |

Specs ship today for `qwen3`, `qwen3_moe`, `qwen3_5`, `glm4_moe_lite` (GLM-4.7-Flash) and
`nemotron_h`; `adaptations/specs/__init__.py` is the source of truth for that list.

MoE is part of this backend rather than an exception to it: expert layers use the fused
Triton kernels in `fsdp_utils/kernels/`, the weight bridge unfuses batched experts at sync
time, and `--use-rollout-routing-replay` (R3) works through per-architecture routing
adapters.

### Try it

```bash
export WANDB_API_KEY=<key>

git clone https://github.com/radixark/miles.git && cd miles
pip install -e . --no-deps

# downloads model + datasets itself, no conversion step
python3 scripts/run_qwen3_0_6b_fsdp.py
```

Launchers with the same recipe shape: `scripts/run_qwen3_0_6b_fsdp.py`,
`scripts/run_qwen3_30b_a3b_fsdp.py`, `scripts/run_nemotron_3_nano_4b_fsdp.py`. To compare
the two backends on one model, `scripts/run_mcore_fsdp.py` takes `--train-backend` as a flag.

For profiling: `--use-pytorch-profiler` with `--profile-step-start` / `--profile-step-end`,
`--record-memory-history` with `--memory-snapshot-path`, and `--tensorboard-dir`. See
[Monitoring & Logging](/user-guide/monitoring).

---

## The other half: SGLang

SGLang is the inference engine no matter which training backend you picked. Three pieces of
configuration matter.

**HuggingFace pointer.** SGLang boots from `--hf-checkpoint`. Miles syncs the actor's
weights from the trainer before the first training step, so the checkpoint at that path does
**not** need to be current. The tokenizer and the `config.json`-derived context length are
all SGLang reads at init.

**Context length override.** SGLang takes max context from `config.json`. To serve beyond it
during training, set `--sglang-context-length`.

**Colocation memory.** Under `--colocate` the trainer reserves VRAM during init before
handing off to SGLang, so drop `--sglang-mem-fraction-static` to **0.8** or lower to let both
fit.

### Passthrough convention

Any flag `python -m sglang.launch_server` accepts, Miles accepts with a `--sglang-` prefix:

```bash
--sglang-ep-size 8
--sglang-enable-dp-attention
--sglang-dp-size 8
--sglang-mem-fraction-static 0.7
--sglang-log-level INFO
```

Two flags are **set by Miles** rather than by you:

- `--tp-size` from `--rollout-num-gpus-per-engine`
- `--model-path` from `--hf-checkpoint`

The integration lives at
[`miles/backends/sglang_utils/arguments.py`](https://github.com/radixark/miles/blob/main/miles/backends/sglang_utils/arguments.py).

### Router

A router sits in front of the SGLang workers. Router-side flags take a `--router-` prefix:

```bash
--router-balance-abs-threshold 0   # force uniform distribution (lowers prefix-cache hit rate)
```

Miles always starts its own router per model; external router mode
(`--sglang-router-ip`/`--sglang-router-port`) was removed and is expected to
return with the k8s-native mode.

---

## Further reading

- [Core concepts](/user-guide/concepts): the four objects that make up any Miles job.
- [Launch script](/user-guide/launch-script): the launch script,
  argument group by argument group.
- [Fully Async RL](/user-guide/fully-async): keep generation running continuously so rollout
  never waits on a training step.
- [Configuration](/user-guide/cli-reference): the flag taxonomy and defaults.
