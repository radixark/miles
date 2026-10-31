---
title: LoRA Training and Weight Sync
description: PEFT LoRA on FSDP diffusion actors and weight sync to sglang-diffusion rollout engines over CUDA IPC or NCCL.
---
Miles-diffusion trains LoRA adapters on the FSDP actor and syncs them to
sglang-diffusion rollout engines each iteration. The rollout engine has no PEFT
layers — weights arrive either as merged base tensors or as raw `lora_A` /
`lora_B` pairs for local merge.

## 1. Recommended flags

For colocated LoRA training, prefer **IPC merge** — push only `lora_A` /
`lora_B` and let the rollout engine merge locally:

```bash
--use-lora \
--lora-ipc-weight-sync \
--lora-rank 32 \
--lora-alpha 64 \
--colocate
```

`--lora-rank` / `--lora-alpha` vary by recipe (e.g. SD3 uses 32/64; some others
use 64/128). Without `--lora-ipc-weight-sync`, LoRA still trains but merges on
the train side and pushes full merged weights (§3). IPC merge requires
`--colocate`.

Without `--colocate` (separate train and rollout GPU pools, e.g. async
training), leave out `--lora-ipc-weight-sync`: LoRA always pushes only
`lora_A` / `lora_B` over NCCL, and the rollout engine merges them the same way.

## 2. Key flags

| Flag | Purpose |
|---|---|
| `--use-lora` | Enable PEFT LoRA on the FSDP actor |
| `--lora-ipc-weight-sync` | Push only `lora_A`/`lora_B` over CUDA IPC; rollout merges locally (requires `--colocate`) |
| `--lora-rank` | LoRA rank (recipe-specific; often 32 or 64) |
| `--lora-alpha` | LoRA alpha (typically 2× rank) |
| `--lora-target-modules` | Override family defaults (optional) |
| `--lora-init-weights` | Init scheme, e.g. `gaussian` |
| `--update-weight-buffer-size` | Sync bucket size in bytes (most recipes use 2 GB) |
| `--update-weight-target-module` | Component to sync (SD3 default: `transformer`) |
| `--colocate` | Train and rollout share GPU visibility and sync over CUDA IPC; without it, weights sync over NCCL |

<Warning>
`--lora-ipc-weight-sync` requires `--use-lora`, and `--colocate` whenever weights
are synced (any run without `--train-only` or `--debug-rollout-only`). Without
colocation, CUDA IPC handles cannot cross the train/rollout process boundary;
LoRA then always syncs `lora_A` / `lora_B` over NCCL, and the flag is rejected.
</Warning>

LoRA target modules default from the model family's
`TrainPipelineConfig.lora_target_modules`. For SD3, see
[SD3 model guide](/diffusion/models/sd3/sd3).

## 3. Weight-sync strategies

Selection logic in `miles/backends/fsdp_utils/actor.py`:

| Condition | Updater class | Behavior |
|---|---|---|
| No LoRA, `--colocate` | `DiffusionUpdateWeightFromTensor` | Full base-weight IPC |
| No LoRA, no `--colocate` | `DiffusionUpdateWeightFromDistributed` | Full base weights over NCCL |
| `--use-lora --colocate`, no IPC | `DiffusionUpdateWeightFromTensorLoRA` | Merge `W + αBA/r` on train side, push merged weights |
| `--use-lora --colocate --lora-ipc-weight-sync` | `DiffusionUpdateWeightFromTensorLoRAIPC` | Push only `lora_A`/`lora_B` over IPC (`weight_update_mode=lora_merge`); the rollout merges them or keeps them unmerged per `--sglang-lora-merge-mode` |
| `--use-lora`, no `--colocate` | `DiffusionUpdateWeightLoRADistributed` | Push only `lora_A`/`lora_B` over NCCL (`weight_update_mode=lora_merge`); the rollout merges them or keeps them unmerged per `--sglang-lora-merge-mode` |

`rollout_merges_lora(args)` in `miles/utils/arguments.py` is the single check
for the last two rows; the engine's LoRA target modules, the merge precision and
the adapter checks (§5) all follow it.

Implementation: `miles/backends/fsdp_utils/diffusion_update_weight_utils.py`.

### Full-weight sync (no LoRA)

FSDP shards are all-gathered into `FlattenedTensorBucket` objects, serialized
via CUDA IPC, and sent to the rollout engine's
`update_weights_from_tensor(load_format="flattened_bucket")`.

### Train-side merge (LoRA, no IPC)

For each base layer with adapters, the updater computes:

```
W_merged = W_base + (B @ A) * scaling
```

on the fly, strips PEFT key prefixes, and pushes standard weight names that
sglang-d expects (e.g. `transformer_blocks.0.attn.to_q.weight`).

### NCCL sync (no `--colocate`)

Trainer rank 0 and every rollout engine GPU join one NCCL group (trainer rank
0, engine *i* from rank `1 + i × --rollout-num-gpus-per-engine`). For each
bucket, rank 0 broadcasts the tensors while the engines receive them through
`update_weights_from_distributed`. This needs an sglang build with distributed
weight updates for diffusion engines.

### Rollout-side LoRA merge (IPC or NCCL)

`DiffusionUpdateWeightFromTensorLoRAIPC` (IPC) and
`DiffusionUpdateWeightLoRADistributed` (NCCL) share this path:

1. `FSDPHfWeightIterator.iter_hf_adapter_weights()` turns each LoRA module into one unit,
   its **lora_A and lora_B**, named by `to_hf_name()`
   (e.g. `transformer_blocks.0.attn.to_q.lora_A`). Fused families such as H3
   keep these diffusers names; sglang-d `lora_merge` applies
   `param_names_mapping` and the disk-load FFN swap.
2. FSDP shard all-gather → `pack_units_by_size()` packs units into buckets of up to
   **`--update-weight-buffer-size`** (most recipes use 2 GB; a unit is never split, so one larger than that ships
   as its own bucket) → CUDA IPC or NCCL
   broadcast.
3. Rollout engine receives `weight_update_mode="lora_merge"` with
   `lora_alpha` and `lora_rank`.

**Bucket packing:** every updater packs **units**, not individual tensors. When
adding the next unit would reach or exceed `--update-weight-buffer-size`, the current
bucket is flushed first; the whole unit then starts the next bucket. A unit is
never split across buckets, so a `lora_A` / `lora_B` pair always arrives together.
For a family whose rollout fuses projections into one layer (H3 `to_q/k/v`, Qwen-Image and
Cosmos3 `add_q/k/v_proj`), the adapters of those projections form one **atomic update group**
and share a bucket too: sglang-d's `lora_merge` replaces a fused layer's LoRA with the sections
one request carries.

Constant: `LORA_WEIGHT_UPDATE_MODE = "lora_merge"`.

Rollout-side merge precision is controlled by environment variable
`SGLANG_DIFFUSION_LORA_MERGE_FP32`:

- `"1"` when `--diffusion-forward-dtype fp32`
- `"0"` otherwise (no fp32 upcast: the engine merges in its own weight dtype, e.g. bf16)

Set automatically in `RolloutManager` when spawning engines that merge LoRA.

On the first few syncs, rank 0 logs lines like:

```text
LoRA weight sync v1 [transformer]: pushed N lora tensors in K buckets
```

After FSDP all-gather, IPC buckets are collected on the **gather-src rank**
only, and that rank calls the rollout engine; over NCCL, trainer rank 0
broadcasts. If sync stalls or VRAM grows across rollouts, check trainer logs
for `LoRA weight sync` lines and Ray worker stderr under
`~/.ray/session_latest/logs/`.

## 4. Internals

| File | Role |
|---|---|
| `miles/backends/fsdp_utils/diffusion_update_weight_utils.py` | Updater classes: send buckets over CUDA IPC or NCCL |
| `miles/backends/fsdp_utils/adaptations/weight_bridge.py` | `ParamTransform` registry (train -> rollout name/shape), as in miles |
| `miles/backends/fsdp_utils/hf_weight_iterator.py` | `FSDPHfWeightIterator`: FSDP shards -> HF-named units (full, LoRA-merged, LoRA adapters) |
| `miles/backends/training_utils/weight_update/hf_weight_iterator/` | `HfWeightIteratorBase` and bucketing (atomic groups, size-bounded packing), as in miles |
| `miles/backends/training_utils/weight_update/hf_weight_iterator/atomic_groups.py` | Per-family atomic groups for weights and adapters (today: LoRA adapters the rollout fuses into one layer) |
| `miles/backends/fsdp_utils/actor.py` | Updater selection |
| `miles/backends/fsdp_utils/model_loader.py` | LoRA apply via PEFT (`apply_lora`, `load_lora_adapter`) |
| `miles/backends/sglang_diffusion_utils/sglang_diffusion_engine.py` | HTTP `update_weights_from_tensor` (IPC) / `update_weights_from_distributed` (NCCL) to rollout |
| `miles/utils/arguments.py` | `rollout_merges_lora(args)` and LoRA weight-sync validation |
| `miles/ray/rollout.py` | Engine env vars (`SGLANG_DIFFUSION_LORA_MERGE_FP32`) |

## 5. Limitations

- **Rollout-side merge expresses only uniform standard LoRA** — adapters from
  `--lora-adapter-path` with rsLoRA or per-layer rank/alpha are rejected for it.
  They need `--colocate` without `--lora-ipc-weight-sync` (train-side merge),
  which has no NCCL path yet.
- **One adapter per trained component** — every component in `--update-weight-target-module` gets its own adapter,
  all from one set of `--lora-*` flags per job.
- **FSDP backend** — LoRA weight sync is implemented for the FSDP diffusion
  actor; Megatron LLM LoRA (in upstream Miles) uses a separate path.
