---
title: Training Backend
description: Megatron-LM as the training backend — parameters, parallelism, checkpoints, and hooks.
---

# Training Backend

Miles decouples the **training backend** (how the model is sharded, checkpointed, and
stepped) from the **inference backend** (SGLang). The production training backend is
**Megatron-LM**.

## Megatron-LM

### Parameter discovery

Miles imports Megatron's entire argument surface at launch through Megatron's parser:

```python
from megatron.training.arguments import parse_args
```

That means every Megatron flag in your installed checkpoint works without Miles having
to re-declare it — `--kv-channels`, `--rotary-base`, `--moe-grouped-gemm`, and so on.

Export the Megatron source directory before you launch:

```bash
export PYTHONPATH=/root/Megatron-LM
```

Miles adds its own arguments by threading an `extra_args_provider` into Megatron's
`parse_args` (see `get_miles_extra_args_provider` in `miles/utils/arguments.py`),
so Miles flags and Megatron flags share a single CLI surface.

### Architecture specs

Most models work with stock `--num-layers / --hidden-size / ...` flags. For models that
need a custom module (Qwen3-Next's Gated-Delta-Net, Qwen3.5's attention-output gate,
GLM5's expert routing), Miles ships a plugin spec:

```bash
MODEL_ARGS=(
   --spec "miles_plugins.models.qwen3_5" "get_qwen3_5_spec"
   ...
)
```

The spec function replaces specific Megatron submodules with the HF implementation
without patching Megatron itself. Details:
[Backends Beyond Megatron](../advanced/architecture-support.md).

### Parallelism compatibility

Megatron exposes five useful parallel dimensions, but you can't combine them in
arbitrary ways — only a subset of TP × PP × CP × EP × ETP combinations is actually
supported, and some legal combinations are slower than the recipe baseline. Start from
the model recipe's tested combination, then change one dimension at a time.

| Dimension | Use it for | Compatibility notes |
|---|---|---|
| TP | Shard dense matrix multiplications inside each layer | When `--tensor-model-parallel-size` is set above 1, also pass `--sequence-parallel` unless the recipe says otherwise. |
| PP | Split layers across pipeline stages | Combines with TP and CP, but changes micro-batch scheduling and checkpoint layout. |
| CP | Split long sequences across ranks | Useful for long context; size token budgets as `CP x max_tokens_per_gpu`. |
| EP | Distribute MoE experts across ranks | MoE-only. Keep trainer EP and SGLang EP as separate choices. |
| ETP | Tensor-parallelize expert MLPs | MoE-only. Use it only when the recipe enables it or when EP alone cannot fit the experts. |

Do not assume TP, CP, EP, and ETP can all be raised independently for a new model — the
exact set of supported combinations depends on the Megatron Core kernels and model spec
being used. The [Argument Groups](argument-groups.md#perf-args) page lists the flags
that belong in `PERF_ARGS`.

### Checkpoint format

Miles uses Megatron's `torch_dist` format — `.distcp` files that are
parallelism-agnostic, so you can change TP / PP / EP without re-converting.

A checkpoint directory looks like:

```text
/ckpt/
├── latest_checkpointed_iteration.txt
├── iter_0000100/
│   ├── _0_0.distcp
│   └── ...
├── iter_0000200/
└── ...
```

Always pass the **parent** directory to `--load`, not a specific iteration. The
loader reads `latest_checkpointed_iteration.txt` to pick the step.

### HuggingFace → torch_dist

```bash
source scripts/models/<family>.sh
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
   ${MODEL_ARGS[@]} \
   --hf-checkpoint /root/<model> \
   --save          /root/<model>_torch_dist
```

For models larger than a single node, drive the converter with
`torchrun --nnodes=<N> --nproc-per-node=8 ...`. Each recipe page lists the exact
command.

### Hooks

Three extension points override Megatron behaviour without forking:

| Flag | Runs |
|---|---|
| `--custom-megatron-init-path` | After Megatron initialization |
| `--custom-megatron-before-log-prob-hook-path` | Before every log-probability computation |
| `--custom-megatron-before-train-step-hook-path` | Before every training step |

Typical use cases: mixing in an auxiliary loss, instrumenting per-step metrics, or
clipping weights surgically. See [Customization](customization.md#megatron-hooks).

---

## SGLang as the inference engine

SGLang is the fixed inference engine regardless of training backend. Three pieces of
configuration matter:

**HuggingFace pointer.** SGLang boots from `--hf-checkpoint`. Before the first training
step, Miles syncs the actor's weights from the trainer — so the checkpoint at that path
does **not** need to be current. The tokenizer and the `config.json`-derived context
length are the only things SGLang cares about at init time.

**Context length override.** SGLang reads max context from the model's `config.json`.
To serve beyond that during training, set `--sglang-context-length`.

**Colocation memory.** Under `--colocate`, Megatron reserves VRAM during init before
handing off to SGLang. Drop `--sglang-mem-fraction-static` to **0.8** (or lower) so
both can coexist.

### Passthrough convention

Any flag accepted by `python -m sglang.launch_server` is accepted by Miles prefixed
with `--sglang-`:

```bash
--sglang-enable-ep-moe
--sglang-enable-dp-attention
--sglang-dp-size 8
--sglang-mem-fraction-static 0.7
--sglang-log-level INFO
```

Conversely, two flags are **set by Miles** rather than the user:

- `--tp-size` ← `--rollout-num-gpus-per-engine`
- `--model-path` ← `--hf-checkpoint`

The integration lives at
[`miles/backends/sglang_utils/arguments.py`](https://github.com/radixark/miles/blob/main/miles/backends/sglang_utils/arguments.py).

### Router

A router sits in front of the SGLang workers. Pass router-side flags with the
`--router-` prefix:

```bash
--router-balance-abs-threshold 0   # force uniform distribution (lowers prefix-cache hit rate)
```

If `--sglang-router-ip` and `--sglang-router-port` are set, Miles treats them as an
**external** router and skips starting its own — engines register via `/add_worker`
at startup.

---

## Further reading

- [Core concepts](concepts.md) — the four objects that make up any Miles job.
- [Training script walkthrough](training-script-walkthrough.md) — the launch script,
  argument group by argument group.
- [Fully Async Rollout](fully-async.md) — decouple generation from trainer steps with
  a queue-backed rollout worker.
- [Configuration](cli-reference.md) — the flag taxonomy and defaults.
- [Backends beyond Megatron](../advanced/architecture-support.md) — wrapping new
  architectures without patching Megatron core.
- [Experimental Features → FSDP backend](../developer/experimental-features.md#fsdp-backend)
  — experimental PyTorch FSDP2 backend for fast iteration on small dense models.
