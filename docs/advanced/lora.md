---
title: LoRA Training and Serving
description: Train LoRA adapters with miles, synchronize them to SGLang, and run dense, MoE, quantized-rollout, multi-LoRA, and agentic recipes.
---

miles trains the adapter matrices in Megatron and serves the same live adapter
from SGLang. The base model stays frozen; at each configured weight-update
boundary, miles exports and synchronizes the updated LoRA weights. The default
boundary is once per rollout/train iteration. No merge-to-base or checkpoint
conversion is required inside the training-rollout loop.

## Training and rollout lifecycle

For a single adapter, miles exports the updated LoRA tensors at each configured
publish boundary. Colocated jobs load them through the local tensor/IPC path;
disaggregated Bridge jobs broadcast them to remote SGLang engines over NCCL.
Rollout requests then select the live named adapter with `lora_path`.

Multi-LoRA currently supports disaggregated rollout only and rejects
`--colocate` at launch. Each SGLang engine keeps the base checkpoint resident;
miles selectively exports and NCCL-broadcasts newly loaded or optimizer-stepped
adapters into their corresponding SGLang slots. New or restarted engines receive
every loaded adapter, while unchanged adapters are not resent.

Model support is therefore a three-way contract rather than a hard-coded
allowlist:

1. Megatron-Bridge (or a model-specific provider) must build and wrap the
   training model.
2. miles must map and export the selected module names correctly.
3. SGLang must be able to allocate and apply the same adapter modules.

A model can support attention-only LoRA while still having caveats for expert,
linear-attention, or model-specific projections. Start from a validated recipe
instead of assuming that any checkpoint with familiar module names will work.

## LoRA training implementations

Both implementations use `--train-backend megatron`. The difference is how
LoRA is attached to the Megatron model and converted for serving; this is
separate from SGLang's serving-side `--sglang-lora-backend` choice.

| Implementation | How the adapter is built | Model coverage on current `main` | Status |
|---|---|---|---|
| **Megatron-Bridge PEFT** | `AutoBridge` builds the provider, applies Bridge LoRA before DDP, and exports HF-named adapter tensors. Select it with `--megatron-to-hf-mode bridge`. | Qwen2.5, Qwen3, GPT-OSS, Kimi K2.5, GLM-5/5.1/5.2, and Qwen3.5/3.6, subject to the evidence and module caveats below. | General production path. Current multi-LoRA also requires this path. |
| **Native / raw-mode LoRA** | Under `--megatron-to-hf-mode raw`, miles builds the model with its own Megatron provider and attaches model-aware adapter modules directly before DDP. | Inkling and Inkling-Small. The current implementation is an Inkling-specific integration, including custom attention, MLP, routed/shared experts, LM head, adapter import, and export. | Specialized path on current `main`; use the Inkling launcher rather than assuming `raw` works for another model. |

<Note>
The planned maintenance direction is native-first: after the generalized native plugin
lands, new model enablement and ongoing LoRA maintenance will primarily target
the native path. Bridge remains the compatibility path for existing validated
recipes during the transition; this is not an immediate Bridge deprecation.
</Note>

### Native LoRA

Native LoRA attaches adapter modules directly to raw-mode Megatron models
instead of relying on Bridge PEFT conversion. The implementation on `main` is
model-specific to Inkling and Inkling-Small. Generalized model-provider support
is under development in open [PR #1792](https://github.com/radixark/miles/pull/1792)
and is not released on `main`.

## Validated models and recipes

The table distinguishes CI-tested configurations from maintained recipes or
full-scale experiment evidence. It is not an exhaustive model whitelist.

| Implementation | Model family | Architecture exercised | Evidence | Notes |
|---|---|---|---|---|
| Bridge | Qwen2.5 0.5B / 3B | Dense | [0.5B CUDA and ROCm E2E](https://github.com/radixark/miles/blob/main/tests/e2e/lora/test_lora_qwen2.5_0.5B.py), [3B disaggregated recipe](https://github.com/radixark/miles/blob/main/examples/lora/run-qwen2.5-3B-megatron-lora-disaggregated.sh) | The simplest starting point; `all-linear` works. |
| Bridge | Qwen3 4B | Dense | [Single-LoRA recipe](https://github.com/radixark/miles/blob/main/examples/lora/run-qwen3-4B-megatron-lora.sh), [multi-LoRA recipe](https://github.com/radixark/miles/tree/main/examples/multi_lora) | Used by the current multi-adapter example. |
| Bridge | GPT-OSS 20B | MoE | [Recipe](https://github.com/radixark/miles/blob/main/examples/lora/run-gpt-oss-20B-megatron-moe-lora.sh), [MoE LoRA E2E](https://github.com/radixark/miles/blob/main/tests/e2e/megatron/model_scripts/test_gpt_oss_20b_moe_lora_ci.py) | Uses the SGLang `triton` LoRA backend. |
| Bridge | Kimi K2.5 | Multimodal MoE + MLA | [16-node recipe](https://github.com/radixark/miles/blob/main/examples/lora/run-kimi-k25-megatron-lora.sh) | Demonstrates shared-outer expert LoRA and an INT4 rollout / fake-QAT setup. |
| Bridge | GLM-5 / 5.1 / 5.2 744B-A40B | MoE + MLA + DSA | [GLM-5.1 launcher](https://github.com/radixark/miles/blob/main/scripts/run_glm5_1_744b_a40b_lora.py), [GLM-5.2 launcher](https://github.com/radixark/miles/blob/main/scripts/run_glm5_2_744b_a40b_lora.py) | CI covers reduced 6-layer / 5-layer checkpoints; historical full-744B results are described below. |
| Bridge | Qwen3.5 / Qwen3.6 35B-A3B | Hybrid GDN + MoE | [Launcher](https://github.com/radixark/miles/blob/main/scripts/run_qwen3_5_35b_a3b_lora.py), [Qwen3.5 E2E](https://github.com/radixark/miles/blob/main/tests/e2e/megatron/test_qwen3_5_35b_a3b_lora_ci.py) | Uses explicit wildcard targets to exclude MTP and vision modules. |
| Native / raw | Inkling / Inkling-Small | Native multimodal MoE | [Launcher](https://github.com/radixark/miles/blob/main/scripts/run_inkling.py), [Inkling-Small 4-layer E2E](https://github.com/radixark/miles/blob/main/tests/e2e/megatron/model_scripts/test_inkling_small_4layer_lora_ci.py) | Current `main` model-specific native path; larger profiles are launcher/experiment evidence rather than LoRA CI. |

## Quick start

Append the following to a Megatron dense-model recipe:

```bash
LORA_ARGS=(
  --lora-rank 32
  --lora-alpha 32
  --lora-dropout 0.0
  --target-modules all-linear
  --megatron-to-hf-mode bridge
)
```

`--lora-rank 0` is the default and disables LoRA when no adapter path is
provided. For RL, dropout is normally set to zero. Alpha is recipe-dependent:
maintained recipes use both `alpha = rank` and `alpha = 2 * rank`.

<Warning>
This quick start is the current Bridge path. On `main`, non-Inkling models must
use `--train-backend megatron --megatron-to-hf-mode bridge`; the Inkling launcher
uses its model-specific native/raw path. The generalized native flags described
in PR #1792 are not released on `main` yet. FSDP does not currently implement
LoRA training.
</Warning>

`all-linear` expands to Q/K/V/O and gate/up/down projections, and conditionally
adds MLA Q/KV projections based on the HF config. It does not literally wrap
every linear layer. GDN and other model-specific projections require an explicit
target list. Current GLM recipes validate models that contain DSA while leaving
the DSA indexer unadapted; current hybrid-model recipes also leave MTP blocks and
vision towers unadapted. Use the model launcher as the source of truth.

### Core arguments

| Flag | Default | Purpose |
|---|---:|---|
| `--lora-rank` | `0` | Adapter rank; a positive value enables LoRA. Supplying an adapter path also marks LoRA enabled, but still requires matching positive-rank configuration. |
| `--lora-alpha` | `16` | Adapter scaling factor. |
| `--lora-dropout` | `0.0` | Dropout on the adapter path. |
| `--lora-type` | `lora` | `lora` uses fused Megatron projections; `canonical_lora` uses split Q/K/V and gate/up projections. The canonical path is implemented and covered by fast name-mapping tests, but has no maintained recipe or E2E validation. |
| `--target-modules` | none | Required with a positive rank. Accepts `all-linear`, HF leaf names, Megatron names, or model-specific wildcard paths. |
| `--exclude-modules` | none | Comma-separated exact entries removed from the resolved targets. |
| `--lora-adapter-path` | none | Warm-start/resume path. Also provide the matching positive rank, alpha, and target modules. Bridge progress resume requires a standard `iter_*/adapter` directory with miles' native shards and the same topology; other shard directories are weight-only warm starts. HF PEFT-only adapters cannot yet load directly into Bridge. Inkling native has its own HF adapter loader. |
| `--lora-base-cpu-backup` | off | Colocated mode only: keep a CPU mirror of the frozen SGLang base and avoid re-sending base weights. This trades host RAM for faster and more reliable pause/resume. |
| `--lora-train-only` | off | Train the adapter while keeping ordinary rollout engines on the frozen base policy. |
| `--experts-shared-outer-loras` | off | Use shared outer factors for grouped MoE experts. This layout is not checkpoint-compatible with per-expert LoRA. |
| `--check-lora-weight-equal` | off | On the colocated path, verify each synchronized adapter tensor with SHA-256. |
| `--update-weights-interval` | `1` | Publish new weights every N rollout/train iterations. This is not LoRA-specific, but it controls when the live adapter is synchronized. |

This argument table describes the general Bridge surface. Current native Inkling
uses a fixed model-specific adapter schema: `--target-modules` does not select
individual training modules, `--exclude-modules` is not applied, and
`canonical_lora` is not implemented. Use the Inkling launcher defaults.

### Rollout topology

| Topology | Adapter transport | Requirements |
|---|---|---|
| Colocated | Local tensor serialization / CUDA IPC | Add `--colocate`; large-model recipes generally also use `--lora-base-cpu-backup`. Pipeline parallel adapter shards are assembled before the load. |
| Disaggregated, including multi-node | NCCL broadcast to remote SGLang engines | Use `--update-weight-transfer-mode broadcast`, Bridge mode, and `--pipeline-model-parallel-size 1`. |

The current native Inkling path is colocated only. Remote/disaggregated LoRA
sync on `main` requires Bridge.

LoRA is not supported by the P2P/RDMA or disk-delta weight-transfer modes. A
hybrid job with both colocated and additional remote rollout engines also cannot
send LoRA weights to the remote engines today.

The legacy `--lora-sync-from-tensor` flag is still accepted by the parser but is
not consumed by the implementation. The topology selects the synchronization
path automatically; do not rely on this flag.

## MoE adapters

This section describes the Bridge path; native Inkling uses its fixed custom
expert layout. For grouped MoE experts, the training and serving layouts must
agree. A typical Bridge configuration is:

```bash
LORA_ARGS=(
  --lora-rank 32
  --lora-alpha 32
  --lora-dropout 0.0
  --target-modules "gate_proj,up_proj,down_proj"
  --sglang-lora-backend triton
  --megatron-to-hf-mode bridge
)
```

The SGLang `triton` backend is required by the maintained MoE recipes; the
default backend can skip MoE layers. Per-expert adapters are the default.
`--experts-shared-outer-loras` selects the shared-outer layout, and miles enables
SGLang virtual-expert serving for expert adapters by default. Use
`--no-sglang-lora-use-virtual-experts` only when intentionally selecting the
alternative aligned-expert path.

## Supported features

- **Losses and algorithms.** LoRA wraps the actor model independently of the
  policy loss. The maintained RL recipes use critic-free estimators such as
  GRPO. The SFT loss path can use the same adapter hooks, although the repository
  does not yet have a dedicated LoRA-SFT E2E test. Shared actor/critic PPO with
  Bridge LoRA is untested: the critic is always built as a full model without
  adapters.
- **Checkpoints.** miles saves native per-rank adapter shards and
  optimizer/scheduler state. Exact resume expects the same TP/PP topology. It
  also attempts a best-effort HF PEFT `adapter_model.bin` plus
  `adapter_config.json` export for external serving and warns if that export
  fails. Direct HF PEFT-to-Bridge resume is not implemented yet; native Inkling
  supplies a model-specific HF adapter importer. A native `iter_*/adapter` resume
  also restores the next rollout ID, the LR schedule position and the
  global-dataset cursor; weight-only adapters start a new run. The schedule
  length is derived from `--num-rollout`, so resuming with a different one needs
  `--override-opt-param-scheduler`.
- **Weight synchronization.** Colocated IPC and remote NCCL broadcast both ship
  adapter tensors at each configured update boundary without merging them into
  the base. A checksum checker is available for the colocated path.
- **Model structures.** Dense attention/MLP, MLA, GDN, multimodal models, models
  containing DSA, and both per-expert and shared-outer MoE adapters have
  validated recipes or tests. DSA-indexer adapters themselves are not validated;
  see the model-specific caveats in the table above.
- **Agentic sessions.** Session-server requests carry the active single-LoRA
  name, so multi-turn trajectories are generated by the updated policy rather
  than the frozen base. See the experimental example below.

### Quantized rollout

Precision support is asymmetric: the validated FP8 configuration trains the
LoRA actor in BF16 while SGLang serves a separately quantized FP8 base
checkpoint. Adapter tensors remain unquantized and are synchronized at each
configured weight-update boundary. The GLM-5.2 launcher exposes this as its
`--fp8-rollout` option and writes an SGLang config with `update_weights: true`.
See [Low Precision RL](/advanced/low-precision) and
[INT4 QAT](/advanced/int4-qat) for the underlying precision features.

Historical GLM-5.2 validation measured:

| Configuration | Train/rollout abs diff | KL |
|---|---:|---:|
| 5-layer GLM-5.2, BF16 rollout | `0.0104` | `1.2e-4` |
| 5-layer GLM-5.2, FP8 rollout | `0.0347` | `9.8e-4` |
| Full GLM-5.2 744B, FP8 rollout, two 16-GPU engines | `0.0196` | `2.7e-3` |

These values are configuration-specific rather than universal FP8 thresholds.
The full-744B row is historical multi-node evidence and cannot be reproduced
directly by the current single-node `main` launcher; multi-node launcher work is
tracked in [PR #2033](https://github.com/radixark/miles/pull/2033).

The same work also produced the following curves from a different, longer
validation run. The exact configuration for these curves was not published, so
their plotted levels should not be mapped to any one of the final point
measurements in the table:

![GLM-5.2 LoRA FP8 rollout train-rollout log-prob difference](/assets/images/lora-glm5-2-fp8-logprob.png)

*Train/rollout log-prob difference in the longer FP8 rollout validation.*

![GLM-5.2 LoRA FP8 rollout reward](/assets/images/lora-glm5-2-fp8-reward.png)

*Raw reward in the longer FP8 rollout validation.*

Kimi K2.5 provides a separate model-specific INT4 example: SGLang serves the
INT4 checkpoint while the trainer uses BF16 fake-QAT and TIS correction. Do not
generalize these two recipes into blanket support for FP8, MXFP8, or INT4 LoRA
training on every model. In particular, multi-LoRA on MoE expert leaves rejects
FP8/FP4 experts.

## GLM-5.2 744B BF16 validation

Historical full-scale runs validated the GLM-5/5.1/5.2 Bridge LoRA path across
MoE and MLA on models containing DSA; the DSA indexer itself was excluded from
the adapter targets. The full GLM-5.2 744B run used 64 GPUs and completed more
than 50 rollout -> train -> save steps. The validation also compared expert
target selections, including a run that excluded `down_proj`.

Treat this as historical implementation and scale evidence. The reported
train/rollout log-prob gaps are configuration-specific and are not established
as a portable acceptance threshold, so the corresponding curves are not
reproduced here.

<Warning>
This historical validation is not a claim that the current `main` launcher
reproduces the 744B run on one node. The checked-in launcher is single-node; the
repository's copy-paste validation path uses a reduced checkpoint. Multi-node
full-744B launcher work is tracked in
[PR #2033](https://github.com/radixark/miles/pull/2033).
</Warning>

## Agentic RL with LoRA (experimental)

Agentic rollout has one extra correctness requirement: every turn sent through
the session server must select the newly synchronized adapter. The session
server therefore attaches `lora_path=miles_lora` to its requests; otherwise the
trainer could update LoRA while the agent continues collecting trajectories
from the frozen base policy. The current session integration selects the fixed
single-adapter name; it is not multi-LoRA slot routing, and it should not be
combined with `--lora-train-only`.

[Draft PR #2280](https://github.com/radixark/miles/pull/2280) adds the first
LoRA-specific agentic recipe: GLM-5.2 744B-A40B with synchronous GRPO on
Terminal-Bench-2-style tasks, a Harbor agent server, and Daytona sandboxes. Its
reference configuration uses:

- 4 nodes x 8 H200, TP8 / EP32 / PP1 / CP1;
- BF16 training with an FP8 rollout checkpoint;
- rank 16, alpha 32, attention/MLA-only targets;
- 64K sessions, TileLang DSA, full recompute, and disk actor offload; and
- two 16-GPU rollout engines with `--lora-base-cpu-backup`.

The author reports 30+ 64K-context GRPO rollouts with stable rewards and no
NaN/OOM, but the PR does not publish a reward value, solve rate, or learning
curve. It remains a draft and depends on an open memory-headroom companion
change ([PR #2199](https://github.com/radixark/miles/pull/2199)), so treat it as
stability evidence rather than a released benchmark.

## Multi-LoRA training

### Current dataset-driven backend

The implementation on `main` trains multiple adapters against one shared base
model through the [fully async example](https://github.com/radixark/miles/tree/main/examples/multi_lora).
Each registered adapter supplies its own dataset, reward, rollout batch shape,
rank/alpha, and checkpoint directory, with most fields inheriting process-wide
defaults. LR/WD hyperparameters come from the global CLI; each fixed slot has
its own Adam state and independently clocked scheduler. The trainer coalesces
ready prompt-group slices or partial adapter batches and selectively upserts
only changed adapters into SGLang.

Set the slot capacity with `--multi-lora-n-adapters N`. A bounded run registers
repeatable `--multi-lora-adapter NAME PATH` entries at startup; service mode can
start with empty slots and register adapters through the controller HTTP API.
This path currently forces Megatron-Bridge LoRA and requires disaggregated NCCL
broadcast, PP1, THD, Adam, and no train offload. Shared-outer expert adapters are
unsupported, and MoE expert adapters cannot use FP8/FP4 experts.

Native multi-LoRA is not implied by the native single-adapter work: both current
`main` and the Tinker-oriented branch below still build multi-LoRA through
Megatron-Bridge. Native multi-LoRA is tracked separately in
[issue #2141](https://github.com/radixark/miles/issues/2141).

### Future Tinker-compatible operation backend

[PR #2273](https://github.com/radixark/miles/pull/2273) is the active
Tinker-oriented backend proposal. It changes ownership of the training loop:
instead of the server owning a dataset, reward function, and one-step schedule,
clients submit explicit operations against a registered adapter. Its primary
intended consumer is a Tinker-compatible training service rather than a generic
server-owned dataset scheduler.

```text
Tinker-style client
  | register + ordered operations
  v
controller / operation ledger
  | bind one fixed LoRA slot
  v
Megatron-Bridge multi-LoRA trainer
  | forward, backward, optimizer, checkpoint
  | save_weights_for_sampler publish barrier
  v
SGLang router + registration-scoped adapter identity
```

The operation surface separates compute, optimization, and publication:

| Operation | Contract |
|---|---|
| `forward` / `forward_backward` | Return per-datum log probabilities; backward calls accumulate client-scaled gradient sums. |
| `optim_step` | Apply client-supplied Adam parameters and clipping to one slot, with an all-rank non-finite veto. |
| `save_weights_for_sampler` | Publish the latest adapter and complete only after the new serving version is live. |
| `save_state` / `load_state` | Save or restore immutable per-adapter weights and optimizer state behind shape, world-size, and ownership fences. |

The design uses fixed residency rather than transparent LRU eviction. Operations
are strictly serialized per registration, while idempotent retries, gap-buffered
arrival, acknowledgements, and backpressure make execution retry-safe and
order-safe. A registration-scoped serving identity prevents an old request from
using a slot after that slot has been reassigned. Authenticated remote access is
the responsibility of the future frontend, not the Ray operation API in #2273.

The v1 scope in the PR is deliberately narrow: text-only synchronous training,
one shared base model, shifted 1-D targets, `cross_entropy`, importance-sampling,
and PPO losses, per-call Adam, and latest-only sampler weights. Multimodal,
top-K/SDFT targets, CISPO/DRO, asynchronous or pinned-snapshot off-policy
training, and cross-world-size restore are outside v1.

<Warning>
This backend is implemented in an open PR, not released on `main`; the PR
reports H200 validation. PR #2273 provides the operation backend, but its v1
training operations are still exposed through the controller's Ray API. The
stacked [PR #2346](https://github.com/radixark/miles/pull/2346) adds a REST
frontend compatible with the official `tinker==0.24.1` client; its GPU frontend
E2E is still pending. If #2273 lands as proposed, it replaces the current
dataset-driven driver.
</Warning>

## Compatibility and limitations

- **Training backend:** Megatron only; FSDP has no LoRA training path.
- **Implementation path on `main`:** Bridge is the general path; native/raw LoRA
  is model-specific to Inkling. General native coverage is pending PR #1792.
- **Remote transport:** NCCL broadcast only, with PP1. P2P/RDMA and disk-delta
  reject LoRA.
- **PPO:** shared actor/critic PPO with Bridge LoRA is untested; the critic never
  gets adapters.
- **Resume:** miles adapter shards are resumable with the matching parallel
  topology. Direct HF PEFT import into the Bridge model is not yet implemented;
  native Inkling has a custom importer.
- **Memory optimizations:** `--rematerialize-param-from-master-weight` and
  streamed optimizer state on NVMe reject LoRA. Ordinary actor disk offload is a
  different feature and is used by the draft agentic recipe.
- **Multi-LoRA:** the current and proposed Tinker operation paths both require
  Bridge; native multi-LoRA remains roadmap work. Evaluation, colocate, PP,
  train offload, shared-outer experts, and FP8/FP4 MoE expert adapters are not
  supported by the current multi-adapter path.
- **Agentic sessions:** the current session integration selects the fixed
  `miles_lora` adapter, not a multi-LoRA slot; `--lora-train-only` is also not a
  supported combination for this path.

## Internals

- `miles/backends/megatron_utils/bridge_lora_helpers.py` builds and wraps the
  general Bridge LoRA model.
- `miles/backends/megatron_utils/lora_utils.py` resolves module names, creates
  standard/canonical adapters, and implements adapter checkpoint helpers.
- `miles_plugins/models/inkling/lora.py` implements the native/raw LoRA path
  available on current `main`.
- `miles/backends/megatron_utils/update_weight/update_weight_from_tensor.py`
  handles colocated adapter export and IPC loading.
- `miles/backends/megatron_utils/update_weight/update_weight_from_distributed/`
  gathers and broadcasts adapters to remote SGLang engines.
- `miles/rollout/session/core.py` attaches the single adapter to agentic session
  requests.
- `miles/ray/multi_lora/`, `miles/rollout/multi_lora/`, and
  `miles/backends/megatron_utils/multi_lora_*.py` implement the multi-adapter
  controller, routing, scheduling, optimization, and checkpoint path.
