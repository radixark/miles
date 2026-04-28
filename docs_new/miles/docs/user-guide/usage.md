---
title: Training Backends
description: Picking between Megatron-LM and FSDP, and the backend-specific plumbing each one exposes.
---

# Training Backends

Miles decouples the **training backend** (how the model is sharded, checkpointed, and
stepped) from the **inference backend** (SGLang, which is fixed). Two training
backends are supported today:

| Backend | Flag |
|---|---|
| Megatron-LM *(default)* | `--train-backend megatron` |
| PyTorch FSDP2 | `--train-backend fsdp` |

Pick one before you touch anything else. Every other decision — checkpoint format,
parallelism strategy, conversion tooling — follows from this.

---

## Choosing a backend

| Concern | Megatron-LM | FSDP |
|---|---|---|
| Peak throughput at MoE scale | ✅ best-in-class | ❌ trailing |
| Expert / tensor / pipeline parallelism | TP · PP · CP · EP · ETP | CP only (TP/PP/EP coming) |
| Checkpoint format | `torch_dist` (custom) | HuggingFace native |
| HF → backend conversion step | **Required** | **Not needed** |
| New-architecture support | Via `miles_plugins` wrappers | Via `AutoConfig` / `AutoModelForCausalLM` |
| Gradient checkpointing | Fine-grained (`--recompute-granularity`) | Boolean (`--gradient-checkpointing`) |
| CPU offload | Distributed optimiser | `--fsdp-cpu-offload` |
| First-time setup effort | Higher | Lower |

**Rule of thumb.** Reach for FSDP when you want to iterate quickly on a new model or
run small-to-mid dense workloads; reach for Megatron when you're training a large MoE,
a multi-rack job, or anything where the parallelism story matters.

---

## Megatron-LM

### Parameter discovery

Miles imports Megatron's entire argument surface at launch via
`from megatron.training.arguments import parse_args`. That means every Megatron flag
in your installed checkpoint works without Miles having to re-declare it —
`--kv-channels`, `--rotary-base`, `--moe-grouped-gemm`, and so on.

Export the Megatron source directory before you launch:

```bash
export PYTHONPATH=/root/Megatron-LM
```

Miles locates flags defined outside `parse_args` the same way Megatron does — by
threading an `extra_args_provider` through:

```python
if __name__ == "__main__":
    try:
        from pretrain_gpt import extra_args_provider
    except ImportError:
        extra_args_provider = None
    args = parse_args(extra_args_provider)
    train(args)
```

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

### Checkpoint formats

Two on-disk formats are supported:

| Format | Flag | Layout | Parallelism-agnostic? |
|---|---|---|---|
| Legacy `torch` | `--ckpt-format torch` | `mp_rank_*/` per rank | ❌ |
| Recommended `torch_dist` | `--ckpt-format torch_dist` | `.distcp` files | ✅ |

Use `torch_dist` unless you have a specific reason not to — it lets you change
parallelism without re-converting.

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

Always pass the **parent** directory to `--load`, not a specific iteration. Use
`--ckpt-step <N>` to pin a particular step; the default reads
`latest_checkpointed_iteration.txt`.

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
| `--custom-megatron-init-path` | After Megatron initialisation |
| `--custom-megatron-before-log-prob-hook-path` | Before every log-probability computation |
| `--custom-megatron-before-train-step-hook-path` | Before every training step |

Typical use cases: mixing in an auxiliary loss, instrumenting per-step metrics, or
clipping weights surgically. See [Customization](customization.md#megatron-hooks).

---

## FSDP (PyTorch FSDP2)

FSDP trades maximum throughput for **zero conversion overhead**. There is no
`torch_dist` step: Miles reads architecture information from the HuggingFace
`config.json` and loads weights directly.

!!! tip "Why FSDP, summarised"
    Architecture discovery is automatic via `AutoModelForCausalLM.from_pretrained()`.
    The distributed optimiser is built-in. Mixed precision falls out of standard
    PyTorch. For small-to-mid dense models, the shortest path from `hf download` to a
    running RL loop is FSDP.

### Enabling it

```bash
--train-backend fsdp
```

### Flag mapping

Most RL-level flags carry over unchanged. Backend-specific differences:

| Concern | Megatron | FSDP |
|---|---|---|
| Model load | `--load` + architecture args | `--hf-checkpoint` *(single flag, required)* |
| Tensor parallel | `--tensor-model-parallel-size` | Coming soon |
| Pipeline parallel | `--pipeline-model-parallel-size` | Coming soon |
| Expert parallel | `--expert-model-parallel-size` | Coming soon |
| Context parallel | `--context-parallel-size` | `--context-parallel-size` |
| Optimiser | `--use-distributed-optimizer` *(opt-in)* | Built-in |
| Gradient checkpoint | `--recompute-granularity / method / num-layers` | `--gradient-checkpointing` *(boolean)* |
| CPU offload | Distributed optimiser | `--fsdp-cpu-offload` |
| CPU backend | *(in distributed optimiser)* | `--fsdp-cpu-backend` |
| Attention backend | Decided by Megatron Core | `--attn-implementation flash_attention_2 / sdpa / eager` |
| Mixed precision | `--fp16` / `--bf16` | `--fp16` *(bf16 inferred)* |
| Extra backend config | — | `--config <yaml>` |

### Quick start

```bash
# Optional: wandb
export WANDB_API_KEY=<key>

# Model + data
hf download Qwen/Qwen3-4B                          --local-dir /root/Qwen3-4B
hf download --repo-type dataset BytedTsinghua-SIA/DAPO-Math-17K --local-dir /root/dapo-math-17k
hf download --repo-type dataset zhuzilin/aime-2024     --local-dir /root/aime-2024

# Code
git clone https://github.com/radixark/miles.git && cd miles
pip install -e . --no-deps

# Launch — no conversion step
bash scripts/run-qwen3-4B-fsdp.sh
```

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

### Router passthrough

Miles talks to SGLang through [`sgl-router`](https://github.com/sgl-project/sglang/tree/main/sgl-model-gateway).
Pass router flags with a `--router-` prefix:

```bash
--router-balance-abs-threshold 0   # force uniform distribution (lowers prefix-cache hit rate)
```

If `--sglang-router-ip` and `--sglang-router-port` are set, Miles treats them as an
**external** router and skips starting its own. Miles-registered engines will call the
external router's `/add_worker` at startup. See
[Miles Router (R3)](../advanced/miles-router.md) for the Miles-specific proxy.

---

## Further reading

- [Core concepts](concepts.md) — the four objects that make up any Miles job.
- [Training script walkthrough](training-script-walkthrough.md) — the launch script,
  argument group by argument group.
- [Configuration](cli-reference.md) — the flag taxonomy and defaults.
- [Backends beyond Megatron](../advanced/architecture-support.md) — wrapping new
  architectures without patching Megatron core.
