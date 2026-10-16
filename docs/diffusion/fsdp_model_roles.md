# FSDP model roles and sleep/wake

The FSDP trainer owns one trainable actor, optional independent frozen reference
and teacher models, and an optional EMA optimizer. Each model contains the same
configured components; Wan2.2 can therefore have both `transformer` and
`transformer_2` in every role.

```text
train worker
  actor components -------- AdamW + scheduler
         |
         +----------------- EMAOptimizer.state[parameter]["ema"]
  reference components ---- frozen, eval, independent checkpoint
  teacher components ------ frozen, eval, independent checkpoint
```

Only actor parameters belong to AdamW. EMA stores averages for trainable actor
parameters in a separate `torch.optim.Optimizer`; frozen base parameters and all
buffers are excluded from averaging.

## Model sources

Use an independent reference with:

```bash
--hf-checkpoint /models/student \
--ref-mode ref --ref-load /models/reference \
--teacher-load /models/teacher \
--custom-loss-function-path my_losses.distill_loss
```

All three sources use the same family backend, component names, input preparation,
and device mesh. This provides separate model instances, not cross-family
distillation or a new DMD2 loss. The existing loss receives the reference prediction.
Custom prepare/loss hooks can access `ctx.reference_models` and `ctx.teacher_models`
by component name. The teacher has no built-in consumer, so `--teacher-load` requires
one of these hooks. Teacher inference should run inside `torch.no_grad()`.

Each source may load pretrained PEFT LoRA adapters, one plain PEFT directory (what
`save_pretrained` writes) per `--update-weight-target-module` entry, in the same order:

```bash
--use-lora --lora-adapter-path /adapters/student --lora-rank 64 --lora-alpha 128 \
--ref-mode ref --ref-load /models/reference \
--ref-lora-adapter-path /adapters/reference \
--teacher-load /models/teacher \
--teacher-lora-adapter-path /adapters/teacher
```

Wan2.2, for example, takes two directories, one per expert; any other number is rejected:

```bash
--update-weight-target-module transformer,transformer_2 \
--use-lora --lora-adapter-path /adapters/high_noise /adapters/low_noise
```

Without a role's adapter path, that frozen role loads ordinary full weights, even
when the actor uses LoRA. Full weights and base-only weights use the same loader.
`--lora-rank` and `--lora-alpha` must match the actor adapter's `r` and `lora_alpha`. Actor adapter weights are loaded
before a training checkpoint is restored. Independent reference/teacher weights
are always reconstructed from their configured sources, so retain those sources
and their adapter checkpoints when resuming training.

`--ref-mode lora_base` keeps the existing low-memory alternative: temporarily
disable the actor adapter for a gradient-free base forward. It does not construct
another base model. FSDP's lazy init runs at load time, while the adapters are
trainable, so the gradient dtype does not depend on which forward runs first. FSDP is
resharded before the adapters are re-enabled, so PEFT restores adapter gradients on
the shards rather than on gathered copies.

## EMA and checkpointing

`--use-ema` creates the EMA optimizer. Under `--loss-type nft` it is π_old: the rollout
samples with it (`--rollout-weights ema`) and the loss trains against its prediction.
`--ref-mode` only selects the KL reference (`lora_base` or `ref`), so NFT can use both.
EMA updates once after a rollout's training completes, and its decay follows the actor's
optimizer step count, as in DiffusionNFT. Publishing or retrying a weight update never advances
EMA. EMA evaluation and publication swap the EMA values into the actor's shards and
swap them back afterward, dropping FSDP's gathered copies on both sides so no forward
reads stale weights; the actor's backward re-gathers its restored shards.

EMA state lives on the same device as its actor parameter. Sleep moves it to
pinned CPU memory together with AdamW state, and wake moves both back.

Rollout publication pushes the current actor weights unless
`--rollout-weights ema` asks for the EMA weights.

```text
iter_0000001/
  model/         actor model state (adapter-only for LoRA)
  optimizer/     AdamW state
  lr_scheduler/  scheduler state
  ema/           EMA optimizer state and decay schedule
```

EMA uses the existing distributed optimizer-state checkpoint wrapper, including
parameter names and DTensor sharding. `--no-save-optim` and `--no-load-optim` apply
to AdamW; EMA remains independently saved and restored. Loading a legacy checkpoint
without EMA seeds EMA from the loaded actor and logs that its averaging history
starts over. Disabling EMA simply omits this state from loading.

## Sleep/wake and native CPU offload

| Setting | Parameter residence during training | Transfer boundary |
|---|---|---|
| No offload | GPU | None |
| `--offload-train` | GPU while awake | Training phase sleep/wake |
| `--fsdp-cpu-offload` | Actor shards on CPU | FSDP unshard |
| `--ref-cpu-offload` | Reference shards on CPU | FSDP unshard |
| `--teacher-cpu-offload` | Teacher shards on CPU | FSDP unshard |

Reference and teacher CPU offload are opt-in. Sleep covers actor, reference,
teacher, their buffers, AdamW state, and EMA state. Sleep clears completed-step
gradients instead of transferring them. Transfers use pinned CPU
storage and enqueue asynchronous copies before synchronization; Parameter objects
and optimizer bindings are preserved. A natively CPU-offloaded model never sleeps,
as in miles: its parameter shards stay on CPU and its buffers stay on GPU.

Sleep retains the model's own CPU tensors, without a second actor backup. Reference and
teacher weights never change, so waking keeps their pinned host copies and each
later sleep points the parameters back at them instead of copying them off the GPU.
The helper lets FSDP rebuild its shard views and padding through `Module._apply`;
uneven shards remain pinned. This requires a small FSDP-internal adaptation of its
pinning flag and is covered by the registered CUDA test on supported PyTorch.

## Focused validation

CPU CI covers EMA averaging, DCP resume, model-source selection, loaded LoRA,
and argument validation. CUDA CI covers real two-rank FSDP sleep/wake with tensor
identity, uneven shards, reshard settings, frozen/trainable models, native
CPU offload, EMA residence, and checkpoint state.

```bash
python -m pytest tests/fast/backends/fsdp_utils/test_ema_optimizer.py \
  tests/fast/backends/fsdp_utils/test_reference_models.py \
  tests/fast/utils/test_reference_arguments.py
python -m pytest tests/fast-gpu/backends/fsdp_utils/test_device_move.py \
  tests/fast-gpu/backends/fsdp_utils/test_reference_models.py
```
