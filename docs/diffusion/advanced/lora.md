---
title: LoRA Training and Weight Sync
description: PEFT LoRA on FSDP diffusion actors and CUDA-IPC weight sync to sglang-diffusion rollout engines.
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

## 2. Key flags

| Flag | Purpose |
|---|---|
| `--use-lora` | Enable PEFT LoRA on the FSDP actor |
| `--lora-ipc-weight-sync` | Push only `lora_A`/`lora_B`; rollout merges locally |
| `--lora-rank` | LoRA rank (recipe-specific; often 32 or 64) |
| `--lora-alpha` | LoRA alpha (typically 2× rank) |
| `--lora-target-modules` | Override family defaults (optional) |
| `--lora-init-weights` | Init scheme, e.g. `gaussian` |
| `--update-weight-buffer-size` | IPC bucket size in bytes (recipes use 2 GB) |
| `--update-weight-target-module` | Component to sync (SD3 default: `transformer`) |
| `--colocate` | **Required** — train and rollout share GPU visibility |

<Warning>
`--lora-ipc-weight-sync` requires both `--use-lora` and `--colocate`. Without
colocation, CUDA IPC handles cannot cross the train/rollout process boundary.
</Warning>

LoRA target modules default from the model family's
`TrainPipelineConfig.lora_target_modules`. For SD3, see
[SD3 model guide](/diffusion/models/sd3/sd3).

## 3. Three weight-sync strategies

Selection logic in `miles/backends/fsdp_utils/actor.py`:

| Condition | Updater class | Behavior |
|---|---|---|
| No LoRA | `DiffusionUpdateWeightFromTensor` | Full base-weight IPC |
| `--use-lora`, no IPC | `DiffusionUpdateWeightFromTensorLoRA` | Merge `W + αBA/r` on train side, push merged weights |
| `--use-lora --lora-ipc-weight-sync` | `DiffusionUpdateWeightFromTensorLoRAIPC` | Push only `lora_A`/`lora_B`; rollout merges via `weight_update_mode=lora_merge` |

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

### LoRA IPC merge (recommended)

`DiffusionUpdateWeightFromTensorLoRAIPC`:

1. `collect_lora_layer_groups()` groups state-dict entries by layer prefix so
   **lora_A and lora_B for the same layer always stay together**.
2. `PeftLoRAKeyMapper.to_sgld_name()` maps PEFT keys to sglang-d names
   (e.g. `transformer_blocks.0.attn.to_q.lora_A`).
3. FSDP shard all-gather → pack into buckets capped by
   **`--update-weight-buffer-size`** (recipes use 2 GB) → CUDA IPC.
4. Rollout engine receives `weight_update_mode="lora_merge"` with
   `lora_alpha` and `lora_rank`.

**Bucket packing:** the IPC updater iterates **layer groups**, not individual
tensors. When adding the next group would exceed `--update-weight-buffer-size`,
the current bucket is flushed first; the whole group (both `lora_A` and
`lora_B`) then starts the next bucket. Pairs are never split across buckets.

Constant: `LORA_IPC_WEIGHT_UPDATE_MODE = "lora_merge"`.

Rollout-side merge precision is controlled by environment variable
`SGLANG_DIFFUSION_LORA_MERGE_FP32`:

- `"1"` when `--diffusion-forward-dtype fp32`
- `"0"` otherwise (fp16 merge)

Set automatically in `RolloutManager` when spawning engines.

On the first few syncs, rank 0 logs lines like:

```text
LoRA IPC weight sync v1 [transformer]: pushed N lora tensors, M layer prefixes in K buckets (unmapped=0)
```

After FSDP all-gather, serialized buckets are collected on the **gather-src
rank** only; that rank calls the rollout engine. If sync stalls or VRAM grows
across rollouts, check trainer logs for `LoRA IPC weight sync` lines and Ray
worker stderr under `~/.ray/session_latest/logs/`.

## 4. Internals

| File | Role |
|---|---|
| `miles/backends/fsdp_utils/diffusion_update_weight_utils.py` | Three updater classes + `PeftLoRAKeyMapper` |
| `miles/backends/fsdp_utils/actor.py` | Updater selection, LoRA apply via PEFT |
| `miles/backends/sglang_diffusion_utils/sglang_diffusion_engine.py` | HTTP `update_weights_from_tensor` to rollout |
| `miles/ray/rollout.py` | Engine env vars (`SGLANG_DIFFUSION_LORA_MERGE_FP32`) |

## 5. Limitations

- **Colocate only** — disaggregated train/rollout is not supported for LoRA IPC.
- **Single adapter per run** — one set of `--lora-*` flags per job.
- **FSDP backend** — LoRA weight sync is implemented for the FSDP diffusion
  actor; Megatron LLM LoRA (in upstream Miles) uses a separate path.
