# torchtitan_utils — torchtitan as a full miles training backend

torchtitan (pip-installed, pinned) supplies the entire training side: native models and
kernels (flex attention, GDN via fla, grouped MoE), `parallelize_fn` composition
(FSDP2/TP/EP over `ParallelDims`), streaming HF-safetensors load, `OptimizersContainer`
(+ MoE load-balancing hook), `LRSchedulersContainer`, DTensor-aware grad clipping.
miles keeps the RL orchestration: rollout via colocated SGLang, data packing,
advantages/loss, the weight-sync wire protocol, checkpoint/resume conventions.
Peer to `megatron_utils`; seam-identical to the #1469-endstate `fsdp_utils` so the two
torch-native backends converge into a shared core after the chain merges.

Design validated by a 3-lens adversarial review (miles contract, torchtitan API
feasibility on torch 2.11, weight-sync/numerics); all blocking findings are folded in
below and marked ⚠ where they changed the original plan.

## Environment

- `pip install --no-deps "git+https://github.com/pytorch/torchtitan@<PIN>"` (currently
  `7e3f2ebc`). PyPI/tag releases predate `models/common`, `qwen3_5`, `distributed/fsdp.py`
  — the pin is main. torchtitan declares no torch dependency, so sglang's torch 2.11 pin
  is untouched.
- torch-2.11 shim (`compat.py`): inject a placeholder `DataParallelMeshDims` dataclass
  into `torch.distributed.fsdp` before any torchtitan import. Empirically verified to
  unblock `distributed.fsdp` and the full qwen3/qwen3_5 model+parallelize+adapter chain.
  Known genuinely-broken-on-2.11 (kept off): EP>1 `apply_fsdp_to_decoder` branch,
  `full_dtensor` mode, varlen attn kwargs (default `flex`).
- Deps: `spmd_types==0.2.1` (hard), `torchdata>=0.8.0` (install for import-gate
  completeness; the runtime path below avoids the `torchtitan.trainer` import chain),
  `fla-core` (optional, gates qwen3_5; NOT the `flash-linear-attention` metapackage
  which drags a conflicting transformers pin).
- Never import `torchtitan.experiments.rl`: hard vllm import + sets
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments` at import time, which breaks CUDA-IPC
  weight transfer. `compat.py` also guards that env var. Anything we want from there
  (batch-invariance converter, loss references) is copied, never imported.

## Package layout

| file | role | reuses |
|---|---|---|
| `__init__.py` | `from . import compat` first, then guarded 2-symbol export (`TorchTitanTrainRayActor`, `load_torchtitan_args`) | fsdp_utils pattern |
| `compat.py` | shim + pin assert (direct_url.json, fingerprint fallback) + `HAS_FLA` probe + alloc-conf guard | — |
| `arguments.py` | `TorchTitanArgs` dataclass → argparse + YAML; `load_torchtitan_args(extra_args_provider=add_miles_arguments)` | fsdp_utils load pattern |
| `models.py` | per-arch registry `model_type → spec_from_hf(config.json) → titan ModelSpec`; v1: `qwen3`; then `qwen3_moe`, `qwen3_5` (gated on `HAS_FLA`) | titan `_build_qwen3_layers`, `ModelSpec`, adapters |
| `model.py` | build+load: sharding-config → meta-build → `parallelize_fn` → `to_empty` → `init_weights` → DCP HF load; `OptimizersContainer` + `LRSchedulersContainer` | titan wholesale |
| `parallel.py` | `ParallelDims(world_size=…)` → meshes → populate miles `ParallelState` (real tp GroupInfo — `get_batch` pads by it) | training_utils ParallelState |
| `actor.py` | `TorchTitanTrainRayActor(TrainRayActor)`; loop bodies from the fsdp shape over shared training_utils; forward seam is titan-native | training_utils data/loss/logging |
| `update_weight_utils.py` | endstate protocol shell + the `named_hf_tensors()` seam (below) | sglang RPC protocol |
| `dtensor.py` | `gather_full_param` (async redistribute→Replicate; generic over FSDP/TP placements) | endstate copy |
| `checkpoint.py` | miles DCP conventions (`iter_%07d/{model,optimizer,lr_scheduler}` + `rng.pt` + `meta.json` + tracker); titan containers are `Stateful`, they slot in directly | endstate copy |
| `models/qwen3_5_packing.py` | GDN packed-doc patch: `cu_seqlens` into fla `chunk_gated_delta_rule` + conv boundary masking | boundaries math (copied) |
| `tests/` | fast: import gate (incl. the 3 near-nightly torch surfaces), mapper asserts, args parse; GPU: keymap gate, adapter round-trip, packed-forward parity vs HF (incl. padded batch), qwen3_5 packed-vs-separate parity | — |

No `loss/`: miles `loss_hub` is the loss (it already emits
`train/train_rollout_logprob_abs_diff` when `rollout_log_probs` is in the batch — the
project's success metric — and keeps curves apples-to-apples with megatron/fsdp).
titan's GRPO/DAPO files are numerics reference only.

## Wiring (3 edits, none touched by the #1469 chain)

1. `--train-backend` choices += `"torchtitan"`.
2. `parse_args`: explicit `elif backend == "torchtitan"` → `load_torchtitan_args(...)`,
   `args.rank=0`, `args.world_size=actor_num_nodes*actor_num_gpus_per_node`,
   `assert context_parallel_size == 1`. No ci-test hard block (warning only).
3. `actor_factory`: explicit `elif backend == "torchtitan"` branch (today non-megatron
   silently falls through to FSDP). LD_PRELOAD memory-saver stays megatron-only.

⚠ `TorchTitanArgs` derives from FSDPArgs' full field list, minus fsdp-only knobs
(`attn_implementation`, `fsdp_*`, `deterministic_mode`), plus titan knobs
(`tt_tensor_parallel_size`, `tt_expert_parallel_size` (=1 v1), `tt_dp_replicate`,
`tt_attn_backend=flex`, `tt_ac_mode`, `tt_compile=False`). It must NOT redefine
main-parser args (`update_weight_buffer_size` collides) and MUST keep the fields shared
code dereferences bare: `context_parallel_size`, `fp16`, the profiler fields.
`log_probs_chunk_size` gets an explicit default sized for 248k vocab (qwen3_5).

## Model build + load

```
spec = models.spec_from_hf(hf_config)         # programmatic; flavors are hardcoded upstream
set_<arch>_sharding_config(...)               # ⚠ direct call — bypasses update_from_config,
                                              #   whose call-time `from torchtitan.trainer import
                                              #   Trainer` drags checkpoint/dataloader/datasets
rope max_seq_len := max(needed, hf)           # dataclasses.replace per layer (cache size)
meta-build under set_default_dtype            # fp32 master weights
spec.parallelize_fn(model, parallel_dims=…, training=…, parallelism=…,
                    compile_config=…, ac_config=…, dump_folder=…)   # bf16 param / fp32 reduce
model.to_empty(device); model.init_weights(buffer_device=None)
adapter = spec.state_dict_adapter(spec.model, hf_dir)
hf_sd = adapter.to_hf(model.state_dict())
dcp.load(hf_sd, storage_reader=adapter.get_hf_storage_reader(hf_dir))   # streaming, sharded
model.load_state_dict(adapter.from_hf(hf_sd))
```

⚠ `fuse_qkv=False` is unreachable via `model_registry` (flavor factories hardcode True);
`spec_from_hf` builds `config.layers` directly with `_build_qwen3_layers(fuse_qkv=False, …)`
— required so `state_dict()` never triggers the per-call wqkv allgather hook — plus a
runtime assert that no module is a `FusedQKVLinear`. (The keymap gate cannot catch this:
fused checkpoints emit the same FQNs via save-hooks.)
⚠ `ParallelDims` requires explicit `world_size=`; `__post_init__` validates the product.

fp32 master + bf16 `mixed_precision_param` forward = the generator-parity recipe from
torchtitan's own RL work: the trainer's forward is the same bf16 the rollout engine runs.

Optimizer: titan `OptimizersContainer` (AdamW config from miles args) +
`spec.post_optimizer_build_fn` (MoE `register_moe_load_balancing_hook`) +
titan `LRSchedulersContainer`. Grad clip: titan `dist_utils.clip_grad_norm_`
(DTensor/TP/EP-correct), `.full_tensor().item()` for logging/ci.

## RL loop integration

Rollout data arrives unpacked; shared `training_utils` does everything
(`get_rollout_data → get_data_iterator → get_batch`): `tokens [1,T_pad]`, per-doc
restarting `position_ids`, `cu_seqlens`, `full_loss_masks`. The single forward-shaped
seam:

```
model(tokens=batch["tokens"], positions=batch["position_ids"],
      attention_masks=model.get_attention_masks(batch["position_ids"]))
```

flex block-causal document masks from position resets replace HF-FA2's implicit varlen
handling. Titan returns raw logits (no `.logits`). Pad tails appear as runs of
position-0 one-token docs — harmless (loss-masked, sliced out), but the M1 parity test
includes a padded batch to prove mask construction tolerates it. Logprobs v1 via miles
`get_log_probs_and_entropy` (fp32 cast, chunked); TP>1 later uses the vocab-sharded
layout miles already supports for megatron. Grad-accum keeps miles' loss scaling
(don't double-apply titan's `global_valid_tokens` division). sleep/wake = `.cpu()/.cuda()`
+ optimizer-state move iterating the container's optimizers.

## Weight sync

Wire protocol byte-identical to the proven shell: `pause_generation(mode=args.
pause_generation_mode)` → `flush_cache` → `begin_weight_update` → dtype-grouped
`FlattenedTensorBucket` buckets (≤ `update_weight_buffer_size`) via CUDA-IPC + per-engine
gloo gather → `update_weights_from_tensor(load_format="flattened_bucket")` →
`end_weight_update` → `continue_generation`.

One day-one seam: `UpdateWeight.update_weights()` consumes an injected
`named_hf_tensors() -> Iterable[(hf_name, tensor)]`. Titan implementation iterates
**layer-grouped chunks** (mandatory for qwen3_5: `to_hf`'s qkv/conv/gate_up fusions need
the whole group in one call): collect layer's titan tensors → `gather_full_param` each
DTensor → `adapter.to_hf(chunk)` → yield. Peak = one gathered layer + one bucket
(embedding/lm_head chunk is the real peak — measured in M3).

⚠ Per-tensor wire dtype = the checkpoint/serving dtype (bf16 matmul weights; **fp32
preserved** for GDN `A_log`/`dt_bias` — a blanket bf16 cast injects compounding
train-vs-rollout drift). The endstate shell's per-tensor `target_dtype` machinery does
this; `UpdateWeightFromDistributed` metadata reflects the same.
⚠ Completeness check is **set equality** per sync: union of pushed HF names ==
`fqn_to_index_mapping` keyset (captured AFTER the first `to_hf` — the adapter pops tied
`lm_head` on first call) minus documented exclusions (qwen3_5 `visual.*`/`mtp.*`).
Membership-only asserts miss silent drops (qwen3_5 fusion guards emit nothing on an
incomplete group). Handle single-file checkpoints where the mapping is None.

Tied embeddings: flavor/mapper sets `enable_weight_tying` from `tie_word_embeddings`
(assert against `index.json` presence of `lm_head.weight`); `to_hf` drops `lm_head` and
sglang re-ties — correct. Buffers with no HF mapping (`expert_bias_E`) drop by
construction. qwen3_5 text-only: sglang keeps its disk-loaded visual/mtp weights across
syncs (visual tower frozen-by-omission) — with `--ci-test`, add `visual.*`/`mtp.*` to
`check_weight_update_skip_list` or the engine equality check fails on the reset vision
tower.

## qwen3_5 (Qwen3.5-4B text-only)

- titan `models/qwen3_5` 4B flavor matches the HF text_config (dim 2560, 32 layers,
  GDN + every-4th full attention — pattern via the `full_attention_interval` kwarg,
  MRoPE, vocab 248320). Mapper validates every GDN field against the checkpoint's
  `config.json` on the pod before first build.
- `fla-core` gates the registry entry (`compat.HAS_FLA`); smoke-test the chunked kernel
  fwd/bwd on the pod's triton before relying on it.
- Vision prune: `model.vision_encoder = None` after meta-build (the PP-prune pattern;
  all forward/parallelize guards handle None). `special_tokens` dict is passed even
  text-only (unconditionally indexed); 2D positions → plain RoPE, correct for text.
- Tie: HF says tied; flavor says untied — set from `index.json` ground truth.
- **The correctness cliff**: GDN layers ignore positions/masks → packed docs leak
  recurrent state and depthwise-conv context across boundaries. `qwen3_5_packing.py`
  threads `cu_seqlens` into `chunk_gated_delta_rule` (native kwarg, `[1,total,…]`
  layout) and masks conv boundary leakage. Hard gate before any RL step: packed-2-docs
  vs separately-forwarded parity (~0 logit diff).

## Unification with #1469

Sibling package, copy-don't-import, converge post-merge. Main's `fsdp_utils` is
pre-refactor and the 32-PR chain rewrites it, so imports are impossible today; a shared
`torch_native_utils/` now would force rebasing 32 open PRs. The chain touches neither
`arguments.py` nor `actor_factory.py`, so the conflict surface is one choices line + an
elif. Copies come from the endstate with seam names/shapes kept byte-compatible; every
copied file carries a twin-file note (bug fixes mirror same-day). Intentional
divergences: titan grad clip / OptimizersContainer / meta+parallelize+DCP-HF build, and
no `adaptations/` layer (titan-native models + `state_dict_adapter` subsume it; the
per-arch seam is `models.py`'s registry). Post-merge: retrofit `named_hf_tensors()` into
fsdp_utils, extract `torch_native_utils/` (sync shell, dtensor, checkpoint, offload),
optionally a shared train-loop mixin.

⚠ Signatures come from **current main's** FSDP actor (`train(rollout_id,
rollout_data_ref, witness_info=None, attempt=0)`, `init(..., recv_ckpt_src_rank=None,
indep_dp_info=...)`) — the endstate predates those kwargs; only loop bodies are copied
from it.

## Milestones

1. **M1** skeleton + build/load/forward parity (qwen3-4b): import gate, keymap gate
   (already passed 398/398 on the mapper prototype), packed+padded forward logprobs vs
   HF transformers (<1e-2 bf16 / <1e-4 fp32).
2. **M2** data-path integration (no sglang): ParallelState, synthetic rollout through
   `get_batch` → forward → logprob order restored; one optimizer step; checkpoint
   save→resume.
3. **M3** weight-sync unit vs 1 colocated engine: set-equality coverage, no-op sync
   bit-identical `/generate` logprobs, weight_version, wall time + peak GPU.
4. **M4** full RL qwen3-4b (gsm8k, colocate, offload): `abs_diff` ~0.01 stable, reward
   tracks FSDP baseline, sleep/wake zero residual, `--ci-test` green (its
   `check_weight_update_equal` reset+compare is a free end-to-end conversion gate),
   save→resume mid-run. → wandb `miles_titan_eval`.
5. **M5** hardening: sync on CPU-offloaded DTensors (train.py offloads before
   update_weights — fallback wake-before-sync), ac modes, step-time vs FSDP.
6. **M6** qwen3_5 offline: fla smoke, mapper+prune+tie, keymap gate, single-doc parity,
   then the GDN packing patch gated by packed-vs-separate parity. Blocks M7.
7. **M7** qwen3_5 full RL run alongside qwen3-4b → both curves in `miles_titan_eval`.
8. **M8** stretch: TP>1, qwen3-30B MoE EP=1, EP>1 port, post-merge convergence PRs.
