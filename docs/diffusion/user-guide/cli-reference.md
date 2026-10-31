---
title: CLI Reference
description: Every flag train_diffusion.py accepts, grouped by subsystem.
---
miles-diffusion is configured entirely through flags on `train_diffusion.py`, coming from three
places:

| Source | Where | Notes |
|---|---|---|
| Training-backend flags | `FSDPArgs` in `miles/backends/fsdp_utils/arguments.py` | Every dataclass field becomes `--field-name` automatically |
| miles flags | `miles/utils/arguments.py` | Added on top as an `extra_args_provider` |
| sglang-d passthrough | The rollout engine's own CLI | Re-registered with a `--sglang-` prefix; a short skip list covers what miles sets itself (`model_path`, ports, `base_gpu_id`, `random_seed`, …) |

`--custom-config-path <file.yaml>` loads YAML keys into the namespace at the very end of argument
validation. A key that collides with an existing argument **overrides it** (with a logged warning) —
the YAML wins over the command line, but the override skips every check and derived value computed
before it (a YAML `global_batch_size`, for example, does not re-derive `num_steps_per_rollout`).

`--config <file.yaml>` is different: it only adds YAML keys that are not arguments at all, so it
cannot change any flag.

Reading the prefixes:

| Prefix | Marks |
|---|---|
| `diffusion-` | Diffusion-specific settings — denoising, SDE, CFG, latents, frames, and the diffusion variants of loss knobs (`--diffusion-clip-range`, `--diffusion-kl-beta`). Shared knobs (EMA, LoRA, batching) do not take it |
| `fsdp-` | The training side — compare `--fsdp-flow-shift` (SFT training sigma grid) with `--diffusion-flow-shift` (the engine's generation schedule) |
| `rollout-` / `sglang-` | The engine side |

The prefix does not tell you which argument group a flag lives in — groups follow concern, not
name. `python3 train_diffusion.py --help` is always the ground truth.

---

## Essentials

### The required flags

| Flag | What |
|---|---|
| `--hf-checkpoint` | The diffusers pipeline to train, as an HF repo id or a local directory. |
| `--diffusion-task` | The task to train on that family, e.g. `t2i` for SD3, `t2v` for Wan2.2. |

One value serves three readers: training loads components from it, the sglang-d
engine serves it, and the **model family** is matched from its name. Add
`--diffusion-model-family` when the name carries no family hint — which local weights usually
do not.

### Cluster topology

| Flag | Default | What |
|---|---|---|
| `--actor-num-nodes` | `1` | Nodes for the training actor. |
| `--actor-num-gpus-per-node` | `8` | GPUs per actor node. Train world size is the product. |
| `--rollout-num-gpus` | – | GPUs for rollout-side work. Forced to the train world size under `--colocate`. |
| `--rollout-num-gpus-per-engine` | `1` | GPUs per sglang-d engine (its `num_gpus`). By default they all go to TP (`--sglang-tp-size`); SP is opt-in via `--sglang-sp-degree`, with `tp × sp` equal to this value (checked when both are set). CFG-parallel (`--sglang-enable-cfg-parallel`) is not supported for training — the trainer's CFG combine does not match it. |
| `--num-gpus-per-node` | `8` | Total GPUs the job may use per node. **Set this when using fewer than 8.** |
| `--colocate` | off | Train and rollout share GPUs and sync weights over CUDA IPC. Without it, rollout runs on its own `--rollout-num-gpus` and weights sync over NCCL (required by `train_diffusion_async.py`). |

### Batch sizing

| Flag | Default | What |
|---|---|---|
| `--rollout-batch-size` | unset | Prompts per rollout. **Required** — the parser rejects a run without it. |
| `--n-samples-per-prompt` | `1` | Samples per prompt (GRPO group size). |
| `--global-batch-size` | derived | **Samples** per optimizer step. Must divide by `dp_size`. |
| `--num-steps-per-rollout` | derived | Optimizer steps per rollout. |
| `--micro-batch-size` | `1` | **Train pairs** per DiT forward (flat cut). |
| `--micro-batch-size-sample` × `--micro-batch-size-tstep` | – | 2D tile per DiT forward. Set together; overrides `--micro-batch-size` with their product. |
| `--num-rollout` | – | Total rollout iterations. Set it or `--num-epoch`. |
| `--num-epoch` | – | Alternative to `--num-rollout`; ignored if both are set. |

For how these batch arguments relate to each other, see
[the batch-knob invariant](/diffusion/user-guide/concepts#the-batch-knob-invariant).

### Diffusion sampling

| Flag | Default | What |
|---|---|---|
| `--diffusion-num-steps` | `10` | Denoising steps per rollout sample. |
| `--diffusion-num-sde-steps` | `0` | How many steps become train pairs; read by the step strategy. |
| `--diffusion-step-strategy-path` | – | Which steps, e.g. `sde_window`. Flow-GRPO training needs one. |
| `--diffusion-noise-level` | `0.7` | SDE noise injected during rollout. |
| `--diffusion-guidance-scale` | `4.0` | CFG scale. |
| `--diffusion-height` / `--diffusion-width` | `512` | Output resolution. |
| `--diffusion-output-num-frames` | `1` | Frames — `1` for images. |
| `--rollout-microgroup-size` | `1` | Samples per prompt sent in one `POST /rollout/generate`. |

### Precision

| Flag | Default | What |
|---|---|---|
| `--diffusion-forward-dtype` | `bf16` | Training-side DiT compute — FSDP `param_dtype` and the input cast. It reaches the engine only as the LoRA-merge precision (`fp32` merges in fp32, see [LoRA](/diffusion/advanced/lora)); the engine's DiT precision is `--sglang-dit-precision`, which is not derived from it — set both to the same value. |
| `--fsdp-master-dtype` | `fp32` | Sharded master weights and optimizer state. |
| `--fsdp-reduce-dtype` | `fp32` | Gradient reduce-scatter. |

See [Dtype Control](/diffusion/advanced/dtype-control).

### Algorithm

| Flag | Default | What |
|---|---|---|
| `--loss-type` | `policy_loss` | `policy_loss` (Flow-GRPO), `nft`, `sft_loss`, `custom_loss`. |
| `--advantage-estimator` | `grpo` | Only `grpo` today. |
| `--diffusion-clip-range` | `1e-4` | PPO ratio clip. |
| `--diffusion-adv-clip-max` | `5.0` | Advantage clamp. |
| `--diffusion-kl-beta` | `0.0` | Reference-KL coefficient; `> 0` turns on a reference forward. |
| `--globalize-reward-std` | off | Batch-wide std instead of per-group. |
| `--globalize-reward-mean` | off | Batch-wide mean instead of per-prompt. |

---

## Complete reference

### Cluster

| Flag | Type | Default | Notes |
|---|---|---|---|
| `--actor-num-nodes` | int | `1` | |
| `--actor-num-gpus-per-node` | int | `8` | |
| `--rollout-num-gpus` | int | – | For train-only SFT, unset colocates encoders with training; set it to reserve dedicated encoder GPUs, which also lets encoding overlap training. |
| `--rollout-num-gpus-per-engine` | int | `1` | sgl-d `num_gpus` per engine; defaults to all-TP. |
| `--num-gpus-per-node` | int | `8` | |
| `--colocate` | flag | off | Train and rollout share GPUs and sync weights over CUDA IPC; without it, rollout gets its own GPUs and weights sync over NCCL. Async training requires it off. |
| `--offload` | flag | off | `--offload-train` + `--offload-rollout`. |
| `--offload-train` / `--no-offload-train` | tri-state | – | On by default under `--colocate`. |
| `--offload-rollout` / `--no-offload-rollout` | tri-state | – | On by default under `--colocate`. |
| `--skip-train-actor-gc-collect` | flag | off | The train actor's `clear_memory` after each rollout's training skips `gc.collect()` and only releases the CUDA cache; saves 0.2-0.5 s per rollout on a large actor process. |
| `--distributed-backend` | str | `nccl` | |
| `--distributed-timeout-minutes` | int | `10` | |

### Training backend

| Flag | Type | Default | Notes |
|---|---|---|---|
| `--train-backend` | enum | `fsdp` | Only value. |
| `--fsdp-master-dtype` | enum | `fp32` | `fp32` / `bf16` / `fp16`. Load, shard, and optimizer-state precision. |
| `--fsdp-reduce-dtype` | enum | `fp32` | `bf16` matches flow_grpo's all-bf16 policy but adds cross-rank add-noise. |
| `--diffusion-forward-dtype` | enum | `bf16` | `bf16` / `fp16` / `fp32`. |
| `--fsdp-cpu-offload` | flag | off | Offloads params, grads, optimizer state; the optimizer then runs on CPU. |
| `--fsdp-cpu-backend` | str | `gloo` | CPU process group for the above. |
| `--dp-replicate-size` | int | `1` | Hybrid sharding: replica count. `dp_shard` takes the ranks left over from this and SP. |
| `--sequence-parallel-size` | int | `1` | USP = Ulysses × Ring. |
| `--ulysses-degree` | int | `0` | `0` = auto (Ulysses fills SP). Ring degree > 1 needs torch ≥ 2.11 and a ring-capable attention backend. |
| `--fsdp-attention-backend` | str | – | diffusers `set_attention_backend` value. |
| `--fsdp-flow-shift` | str | – | SFT training sigma grid shift per latent stream, e.g. `visual=12,audio=3`; a bare number is the visual shift. RL pairs carry their rollout sigmas instead. Distinct from `--diffusion-flow-shift`. |
| `--fsdp-supervised-streams` | str | `visual,audio` | Comma-separated latent streams the SFT loss supervises, among those the pair carries; must include `visual`. A stream left out is still noised and fed to the DiT but adds no loss. RL trajectories carry only the visual stream. |
| `--gradient-checkpointing` | flag | off | |
| `--deterministic-mode` | flag | off | See [Deterministic Training](/diffusion/advanced/deterministic). |
| `--train-env-vars` | JSON | `{}` | Extra env for the training processes. |

### Optimizer and schedule

| Flag | Type | Default | Notes |
|---|---|---|---|
| `--optimizer` | str | `adam` | AdamW. The only value the actor accepts. |
| `--lr` | float | `1e-6` | |
| `--adam-beta1` / `--adam-beta2` | float | `0.9` / `0.999` | β₂ matches flow_grpo, not the LLM-side `0.95`. |
| `--adam-eps` | float | `1e-8` | |
| `--weight-decay` | float | `0.0` | |
| `--clip-grad` | float | `1.0` | |
| `--lr-decay-style` | str | `constant` | `constant` / `linear` / `cosine` / `inverse-square-root` / `WSD`. |
| `--min-lr` / `--lr-warmup-init` | float | `0.0` | |
| `--lr-warmup-iters` | int | `0` | |
| `--lr-warmup-fraction` | float | – | |
| `--lr-decay-iters` | int | – | |
| `--lr-wsd-decay-iters` / `--lr-wsd-decay-style` | int / str | – | For `--lr-decay-style WSD`; style is `linear` / `cosine` / `exponential` / `minus_sqrt`. |
| `--use-checkpoint-lr-scheduler` | flag | off | No effect in the FSDP backend, which always restores a saved LR-scheduler state from `--load`. |
| `--override-lr-scheduler` | flag | off | No effect in the FSDP backend; rejected together with `--use-checkpoint-lr-scheduler`. |
| `--seed` | int | `1234` | |

### Rollout

| Flag | Type | Default | Notes |
|---|---|---|---|
| `--hf-checkpoint` | str | – | **Required.** Pipeline to train and to serve; also the family source. |
| `--diffusion-model-family` | str | – | Registered family key: `sd3`, `wan2_2`, `ltx`, `qwen_image`, `qwen_image21`, `cosmos3`, `krea2`, `h3`. Overrides name matching. The longest pattern wins, so `Qwen-Image-2.1` resolves to `qwen_image21`. |
| `--diffusion-task` | str | – | **Required** unless the config declares no `tasks`. The task to train, one of the family config's `tasks`: `t2i` (sd3, krea2, qwen_image, qwen_image21, cosmos3), `t2v` (wan2_2, ltx), `t2va` (h3, the only H3 rollout task) or `ref2va` (h3 joint video/audio SFT with ordered references on `transformer_ref`). |
| `--rollout-function-path` | str | `miles.rollout.sglang_rollout.generate_rollout` | Inherited Miles default; that module is not shipped in miles-diffusion, so set it — diffusion recipes pass `miles.rollout.sglang_diffusion_rollout.generate_rollout`. |
| `--train-pipeline-config-path` | str | – | Your own `TrainPipelineConfig` for an unregistered family. Mutually exclusive with `--diffusion-model-family`. |
| `--model-backend-path` | str | – | Override the family's model loader. |
| `--diffusion-num-steps` | int | `10` | |
| `--diffusion-flow-shift` | float | – | Generation-schedule shift, sent to the engine. |
| `--rollout-microgroup-size` | int | `1` | |
| `--diffusion-fps` | float | – | Video only. |
| `--diffusion-output-num-frames` | int | `1` | |
| `--rollout-video-dtype` | enum | `keep` | `keep` / `uint8`: dtype of the decoded video in rollout responses. `uint8` quantizes engine-side by truncation (×255, then clamp), cutting the response ~4x. That matches the PickScore reward path; HPS, OCR and the OpenAI-API reward round instead, so keep `keep` for them and for any consumer that needs the float tensor. |
| `--diffusion-guidance-scale` | float | `4.0` | |
| `--diffusion-guidance-scale-2` | float | – | Wan2.2 low-noise expert; **required** when training it. |
| `--diffusion-true-cfg-scale` | float | – | |
| `--diffusion-negative-prompt` | str | – | Unset sends none, so the engine's per-model default applies; flow_grpo recipes pass `" "`. |
| `--diffusion-noise-level` | float | `0.7` | |
| `--diffusion-quality` | enum | `exact` | `exact` / `lossless` / `high`: sglang-diffusion request quality level. `exact` is the bit-identical reference path; the looser levels mount rounding-changing fusions. |
| `--diffusion-height` / `--diffusion-width` | int | `512` | Rollout output size; SFT target media must already be this size. |
| `--diffusion-h3-aspect-ratio` | str | `16:9` | MiniMax H3 only: `21:9` / `16:9` / `4:3` / `1:1` / `3:4` / `9:16`. |
| `--diffusion-h3-duration-seconds` | float | `5.0` | MiniMax H3 only: rollout duration, 4.0–15.0 s. |
| `--diffusion-audio-flow-shift` | float | `3.0` | MiniMax H3 only: audio flow shift for rollout. |
| `--diffusion-sde-type` | str | `sde` | `sde` / `cps` / `ode`, checked at startup unless `--sde-step-backend-path` is set. Selects the train-side SDE backend too. |
| `--sde-step-backend-path` | str | – | Custom dynamics. See [SDE backends](#sde-step-backends). |
| `--diffusion-num-sde-steps` | int | `0` | Read by the step strategy. |
| `--diffusion-sde-window-range` | `"lo,hi"` | – | For `sde_window`. Defaults to `[0, num_inference_steps)`. |
| `--diffusion-sde-candidate-steps` | `"1,2,3"` | – | Required by `epoch_global_random_choice`. |
| `--diffusion-step-strategy-path` | str | – | `(args, sample, num_steps, seed) -> (sde, ret)` choosing the SDE and returned steps, e.g. `miles.rollout.step_strategy_hub.sde_window`. Flow-GRPO needs one. |
| `--rollout-return-full-trajectory` | flag | off | Ask the engine for the whole denoising trajectory instead of only the steps the step strategy requests. The train tensors are identical; kept for debugging and A/B runs. |
| `--diffusion-log-prob-no-const` | flag | off | Drop log-prob constants on the engine (pairs with the CPS backend). Forced on for `cps` / `ode`. |
| `--diffusion-generator-device` | str | `cuda` | |
| `--rollout-patch-group` | str | – | Comma-separated numeric-parity patch groups: `qwen_image`, `wan`, `ltx`. |
| `--update-weight-target-module` | str | `transformer` | Modules to train and sync. Wan2.2: `transformer,transformer_2`. |
| `--update-weight-buffer-size` | int | 512 MiB | Weight-sync chunk size in bytes. |
| `--rollout-seed` | int | `42` | |
| `--rollout-shuffle` | flag | on | Shuffle the prompt order once per epoch (seeded by `--rollout-seed`). `--no-rollout-shuffle` reads the jsonl in file order; the SFT rollout (`miles.rollout.sft_rollout`) shuffles every epoch regardless. |
| `--over-sampling-batch-size` | int | – | Must equal `--rollout-batch-size` today. |
| `--sglang-server-concurrency` | int | `512` | In-flight generate requests are capped at this × the number of engines — one global limit, with each request routed to the least-loaded engine. |
| `--use-distributed-post` | flag | off | Rollout HTTP POSTs go through per-node Ray actors instead of the local client (`--num-gpus-per-node` actors per node). |
| `--use-miles-router` | flag | off | **Required** — the SGLang router is not supported here. |
| `--miles-router-timeout` | float | – | |
| `--miles-router-max-connections` | int | – | |
| `--miles-router-health-check-failure-threshold` | int | `3` | |

### Data and batching

| Flag | Type | Default | Notes |
|---|---|---|---|
| `--prompt-data` | str | – | jsonl, one row per prompt. |
| `--input-key` | str | `input` | |
| `--metadata-key` | str | `metadata` | |
| `--data-source-path` | str | – | Defaults to `RolloutDataSourceWithBuffer`. |
| `--disable-rollout-global-dataset` | flag | off | Manage data yourself. The shipped diffusion rollout asserts the global dataset, so this needs a custom `--rollout-function-path`. |
| `--rollout-batch-size` | int | – | **Required.** |
| `--n-samples-per-prompt` | int | `1` | |
| `--global-batch-size` | int | derived | |
| `--num-steps-per-rollout` | int | derived | |
| `--micro-batch-size` | int | `1` | Flat: train-pair dicts per DiT forward, contiguous within an optimizer window. |
| `--micro-batch-size-sample` | int | – | 2D: samples per DiT-forward tile. |
| `--micro-batch-size-tstep` | int | – | 2D: SDE timesteps per tile. Set with the above. |
| `--diffusion-train-iter-order` | enum | `sample_major` | `sample_major` / `timestep_major`. Tile visit order; only meaningful with the 2D pair. |
| `--train-dp-split-mode` | enum | `contiguous` | `contiguous` lets a micro-batch reproduce a rollout microgroup exactly; `stride` deals round-robin. |
| `--num-rollout` | int | – | Required unless `--num-epoch` is set. |
| `--num-epoch` | int | – | |
| `--start-rollout-id` | int | – | Resumed from `--load` when unset. |
| `--sft-encoder-checkpoint` | str | – | SFT only: tokenizer/text_encoder/vae source. |
| `--sft-frame-stride` | int | `1` | SFT encode temporal stride; target videos hold exactly `(num_frames - 1) * stride + 1` frames. |
| `--sft-offload-encoder` | flag | off | SFT only: keep the frozen encoder in host RAM, on the GPU only during encode bursts. |

### Evaluation

| Flag | Type | Default | Notes |
|---|---|---|---|
| `--eval-interval` | int | – | Requires configured eval datasets. |
| `--eval-prompt-data` | str+ | – | `<name> <path>` pairs, all after one flag (repeating the flag keeps only the last). |
| `--eval-config` | str | – | OmegaConf YAML/JSON; overrides `--eval-prompt-data`. |
| `--eval-function-path` | str | – | Defaults to `--rollout-function-path`. |
| `--eval-input-key` | str | – | |
| `--n-samples-per-eval-prompt` | int | `1` | |
| `--diffusion-eval-num-steps` | int | – | Defaults to `--diffusion-num-steps`. |
| `--skip-eval-before-train` | flag | off | |
| `--eval-reward-key` | str | – | Defaults to `--reward-key`. |

### Algorithm and loss

| Flag | Type | Default | Notes |
|---|---|---|---|
| `--loss-type` | enum | `policy_loss` | `policy_loss` / `nft` / `sft_loss` / `custom_loss`. |
| `--advantage-estimator` | enum | `grpo` | |
| `--disable-grpo-std-normalization` | flag | off | Subtract the group mean only; skip dividing by the group std. Std normalization is also skipped when `n_samples_per_prompt == 1`, since a one-sample group has no std. With `--globalize-reward-mean` the batch mean is subtracted instead, and the batch-wide std from `--globalize-reward-std` is skipped too. |
| `--globalize-reward-mean` | flag | off | Leave **off** for flow_grpo parity. |
| `--globalize-reward-std` | flag | off | **On** for flow_grpo's PickScore recipe. |
| `--diffusion-clip-range` | float | `1e-4` | |
| `--diffusion-adv-clip-max` | float | `5.0` | Under `nft` this also sets the advantage-to-`r` slope. |
| `--diffusion-recompute-old-log-prob` | flag | off | Recompute old log-probs with the trainer forward instead of trusting the rollout's. `policy_loss` only. |
| `--diffusion-kl-beta` | float | `0.0` | KL to the `--ref-mode` model; under `nft` adds `kl_beta * mean((v_θ − v_ref)²)` per pair. |
| `--ref-mode` | enum | – | KL reference: `none` / `lora_base` (actor with adapters disabled) / `ref` (the model from `--ref-load`). Auto: `lora_base` when KL > 0. |
| `--ref-load` | str | – | HF pipeline checkpoint of an independent frozen reference model. Requires `--ref-mode ref`. |
| `--ref-lora-adapter-path` | str+ | – | PEFT adapter directories applied to `--ref-load`, one per `--update-weight-target-module` entry, in order. |
| `--ref-cpu-offload` | flag | off | Keep the reference shards in pinned CPU memory between forwards (FSDP `CPUOffloadPolicy`). |
| `--teacher-load` | str | – | HF pipeline checkpoint of a frozen teacher model, read only by custom prepare/loss hooks as `ctx.teacher_models`, so it requires `--custom-prepare-train-batch-path` or `--custom-loss-function-path`. |
| `--teacher-lora-adapter-path` | str+ | – | PEFT adapter directories applied to `--teacher-load`, one per `--update-weight-target-module` entry, in order. |
| `--teacher-cpu-offload` | flag | off | Keep the teacher shards in pinned CPU memory between forwards. |
| `--custom-prepare-train-batch-path` | str | – | Builds DiT inputs. |
| `--custom-loss-function-path` | str | – | Loss **formula** only — the DiT forward stays in the actor. |
| `--diffusion-nft-beta` | float | `1.0` | |
| `--diffusion-nft-timestep-strategy-path` | str | `…nft.drop_random_steps` | `(args, num_steps, generator) -> list[int]` returning the denoising steps each sample trains on, in training order; `generator` is seeded per sample. Built in: `drop_random_steps` (random order), `drop_final_steps` (schedule order, noisiest first). The final clean output (σ = 0) is never a training step. |
| `--diffusion-nft-num-dropped-timesteps` | int | `0` | Steps the built-in strategies drop: random ones per sample for `drop_random_steps`, the final (lowest-noise) ones for `drop_final_steps`. |
| `--no-diffusion-nft-adaptive-weight` | flag | off | |

### Reward

| Flag | Type | Default | Notes |
|---|---|---|---|
| `--rm-type` | str | – | `pickscore` / `hps` / `ocr` / `openai_api` / `custom_api`. Overridable per sample via `metadata.rm_type`. Ignored when `--custom-rm-path` is set. |
| `--custom-api-rm-config` | str | – | Required for `--rm-type custom_api`: a YAML file or inline `base64:<payload>` with `actor_class`, `actor_kwargs`, `max_concurrency`. A flat mapping configures the default OpenAI-compatible image actor. |
| `--openai-api-rm-config` | str | – | Required for `--rm-type openai_api`; same format as `--custom-api-rm-config`. |
| `--group-rm` | flag | off | Score a whole prompt group at once. |
| `--custom-rm-path` | str | – | `async def rm(args, samples) -> list[float]`. Batched only; replaces the `--rm-type` dispatch entirely. Shipped: `miles.rollout.rm_hub.weighted_mixture_rm.weighted_mixture_rm` (weighted sum of built-in rewards). |
| `--custom-rm-args` | str | – | Opaque config string for the custom RM, read as `args.custom_rm_args`; e.g. `"hps=0.7,pickscore=0.3"` for `rm_hub.weighted_mixture_rm`. |
| `--rm-url` | str | – | URL of a remote reward service, e.g. `http://localhost:8000`; mirrors miles core. No built-in `--rm-type` reads it yet, so read it from a `--custom-rm-path` function. |
| `--reward-key` | str | – | For dict-valued rewards: the entry GRPO trains on. Every entry is also logged as `rollout/reward/<key>_mean` and `eval/<dataset>/<key>`. |
| `--custom-reward-post-process-path` | str | – | Replace advantage normalisation. |
| `--pickscore-model-path` | str | – | Required for `--rm-type pickscore`. |
| `--pickscore-processor-path` | str | – | Required for `--rm-type pickscore`. |
| `--pickscore-num-workers` | int | `1` | |
| `--pickscore-num-gpus-per-worker` | float | `1.0` | Standalone workers only; fractional values allowed. |
| `--pickscore-reward-colocate` | flag | off | Seats `--pickscore-num-workers` actors on the rollout GPUs, one per placement-group bundle, sharing them with the train actor and rollout engine. Requires `--colocate`. |
| `--pickscore-batch-size` | int | `8` | |
| `--pickscore-num-frames` | int | – | Frames scored per video; unset = all. |
| `--hps-num-workers` | int | `1` | |
| `--hps-num-gpus-per-worker` | float | `1.0` | Standalone workers only; fractional values allowed. |
| `--hps-reward-colocate` | flag | off | Seats `--hps-num-workers` actors on the rollout GPUs, one per placement-group bundle, sharing them with the train actor and rollout engine. Requires `--colocate`. |
| `--hps-batch-size` | int | `8` | |
| `--hps-version` | enum | `v2.1` | `v2.0` / `v2.1`. |
| `--hps-checkpoint-path` | str | – | Local checkpoint; unset downloads from Hugging Face. |
| `--ocr-num-workers` | int | `4` | |
| `--rollout-parser-num-workers` | int | `1` | Ray actors deserializing rollout responses. Raise when trajectory tensors are large. |
| `--rollout-fetch-in-parser` | flag | off | Parser actors pick the least-loaded engine through the miles router, then fetch and parse its response in place, skipping the router's data plane and the manager's event loop. Requires `--use-miles-router` and at least as many parser workers as concurrency slots. |

### Rollout customization hooks

Every one takes a dotted path.

| Flag | Replaces |
|---|---|
| `--custom-generate-function-path` | The inner `generate` call in the diffusion rollout. |
| `--dynamic-sampling-filter-path` | Per-group keep/drop (DAPO-style). |
| `--buffer-filter-path` | Buffer selection. |
| `--custom-convert-samples-to-train-data-path` | The whole samples → train-data conversion. |
| `--custom-expand-samples-to-train-pairs-path` | Just the sample → train-pair expansion. |
| `--custom-rollout-log-function-path` | Train rollout logging. |
| `--custom-eval-rollout-log-function-path` | Eval rollout logging. |

### LoRA

| Flag | Type | Default | Notes |
|---|---|---|---|
| `--use-lora` | flag | off | |
| `--lora-rank` / `--lora-alpha` | int | `64` / `64` | |
| `--lora-adapter-path` | str+ | – | PEFT adapter directories the actor continues from, one per `--update-weight-target-module` entry, in order; `--lora-rank` / `--lora-alpha` must match their configs. |
| `--lora-target-modules` | str+ | – | Defaults per model family. |
| `--lora-init-weights` | str | `gaussian` | `kaiming-uniform` maps to PEFT's default; other PEFT schemes pass through. |
| `--lora-ipc-weight-sync` | flag | off | Push only `lora_A`/`lora_B` over CUDA IPC and let the rollout merge them (`weight_update_mode=lora_merge`); without it, the trainer merges LoRA into the base weights and pushes full weights. Requires `--use-lora`, and `--colocate` whenever weights are synced (not under `--train-only` / `--debug-rollout-only`). Without `--colocate`, LoRA always pushes `lora_A`/`lora_B` over NCCL for the rollout to merge. |

### EMA

| Flag | Type | Default | Notes |
|---|---|---|---|
| `--use-ema` | flag | off | Maintains an EMA copy as `π_old`. Needs a consumer: `--rollout-weights ema`, or `--loss-type nft`, which trains against it and requires both this flag and `--rollout-weights ema`. |
| `--rollout-weights` | enum | `actor` | `actor` / `ema`: which weights get pushed to rollout. `ema` requires `--use-ema`. |
| `--ema-decay-init` | float | `0.001` | Decay during the flat period, counted in actor optimizer steps. |
| `--ema-decay-ramp` | float | `0.001` | Increase per actor optimizer step after the flat period; the ramp restarts from zero. |
| `--ema-decay-max` | float | `0.5` | Ceiling. |
| `--ema-decay-flat-steps` | int | `0` | |

### Checkpointing

| Flag | Type | Default | Notes |
|---|---|---|---|
| `--save` | str | – | |
| `--save-interval` | int | – | Requires `--save`. |
| `--no-save-optim` | flag | off | Skips AdamW and LR-scheduler state; `--load` still resumes the model, EMA, and RNG with a fresh optimizer. |
| `--load` | str | – | |
| `--ckpt-step` | int | – | Defaults to `latest_checkpointed_iteration.txt`. |
| `--no-load-optim` / `--no-load-rng` | flag | off | |
| `--async-exact-resume` | flag | off | Async training only. Also checkpoints the EMA from before the latest step and resamples the first resumed batch with it, so a resume matches an uninterrupted run. Off: that batch samples the loaded EMA. |

### Logging

| Flag | Type | Default | Notes |
|---|---|---|---|
| `--use-wandb` | flag | off | |
| `--wandb-project` | str | – | |
| `--wandb-group` / `--wandb-run-id` / `--wandb-team` / `--wandb-host` / `--wandb-key` | str | – | |
| `--wandb-mode` | enum | – | `online` / `offline` / `disabled`. Overrides `WANDB_MODE`. |
| `--wandb-dir` | str | – | Defaults to `./wandb`. |
| `--disable-wandb-random-suffix` | flag | off | Run names include a random suffix by default; pass this flag to disable it. |
| `--wandb-log-num-images` | int | `0` | Images/videos per rollout; `0` disables. |
| `--wandb-log-image-interval` | int | `1` | Send media every N rollouts. |
| `--log-loss-sigma-bucket` | int | `10` | Sigma buckets for the per-bucket loss curves (emitted by the SFT loss): `loss_<stream>_sigma_*` for each supervised stream, bucketed by its own sigma; the total `loss` is never bucketed, since each stream reads the shared grid index through its own shift. `0` disables. |
| `--use-miles-dashboard` | flag | off | Async phase/trajectory telemetry. |
| `--miles-dashboard-workspace` | str | `./miles_dashboard` | |

### Fault tolerance

| Flag | Type | Default | Notes |
|---|---|---|---|
| `--use-fault-tolerance` | flag | off | Health-check rollout engines and kill one that fails a check; restarting it is not wired into the diffusion training loop. |
| `--rollout-health-check-interval` | float | `30.0` | Seconds between engine probes (`GET /health_generate`, a readiness check that runs no generation); the miles router polls `/health` on the same interval. |
| `--rollout-health-check-timeout` | float | `30.0` | |
| `--rollout-health-check-first-wait` | float | `0` | Grace period for compilation/init. |

### Debugging

| Flag | Type | Default | Notes |
|---|---|---|---|
| `--debug-rollout-only` | flag | off | Rollout, no training. |
| `--train-only` | flag | off | No engines, no weight sync, no eval. (`--debug-train-only` is a legacy alias.) |
| `--debug-skip-optimizer-step` | flag | off | No backward/step — weights never drift. Use it to measure pure forward divergence from the engine. |
| `--save-debug-rollout-data` | path template | – | Fills `{rollout_id}`. |
| `--load-debug-rollout-data` | path template | – | Replays a rollout with **no engines started**. |
| `--load-debug-rollout-data-subsample` | float | – | |
| `--save-debug-train-data` | path template | – | Fills `{rollout_id}` and `{rank}`. |
| `--dump-details` | dir | – | Sets the debug dump paths for post-hoc analysis. |
| `--diffusion-debug-mode` | flag | off | Engine returns per-step debug tensors; enables the `model_output_*_abs_diff` metrics. |

`--debug-skip-optimizer-step` + `--diffusion-debug-mode` is the standard train/rollout alignment
probe: frozen weights, and the metrics report exactly how far the two forwards drift.

### CI

`--ci-test`, `--ci-metric-checker-key`, `--ci-metric-checker-threshold`. Used by `tests/e2e`;
not meant for manual runs.

---

## SDE step backends

`--diffusion-sde-type` picks the train-side backend automatically:

| `--diffusion-sde-type` | Backend | Dynamics |
|---|---|---|
| `sde` | `FlowSdeStepBackend` | Flow-matching SDE over the rollout's recorded sigmas. |
| `ode` | `FlowSdeStepBackend` | Same class; deterministic rollout (used by NFT). |
| `cps` | `CpsSdeStepBackend` | CPS kernel over the rollout's recorded sigmas, log-prob without constants. |

`--sde-step-backend-path` overrides the mapping with your own `SdeStepBackend` subclass.
