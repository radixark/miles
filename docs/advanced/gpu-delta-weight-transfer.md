---
title: GPU-Delta Weight Transfer
description: Lossless XOR weight updates compressed on trainer GPUs and applied in place on SGLang rollout GPUs, with recovery, eval snapshots and metrics.
---

`--update-weight-transfer-mode gpu-delta` is an experimental weight synchronization mode
for non-colocated Megatron training with SGLang rollout. Each update publishes a lossless
XOR delta from the previous version. Trainer owners compress the changed bytes on their
GPUs and write them to shared storage; every SGLang engine prepares the publication while
it keeps serving, then XORs it into its existing execution-layout buffers during a short
pause. Updates neither reload a checkpoint nor requantize weights. See
[GPU-accelerated weight teleportation](https://humansand.ai/blog/gpu-accelerated-weight-teleportation)
for the design and motivation.

This page covers the trainer side, recovery, dedicated eval and metrics. The
[receiver contract](https://github.com/radixark/miles/blob/main/miles/backends/sglang_utils/gpu_delta/README.md)
owns layout admission, control APIs, cache lifecycle, decoder scheduling and receipt
fields; the
[development guide](https://github.com/radixark/miles/blob/main/tests/manual/gpu_delta/README.md)
owns development variables, the five-layer E2E test and the benchmark.

## Usage

```bash
--update-weight-transfer-mode gpu-delta \
--update-weight-delta-encoding xor \
--update-weight-disk-dir /shared/miles/weight-updates \
--hf-checkpoint /models/immutable-hf-base
```

- `--hf-checkpoint` must be the immutable local safetensors checkpoint that the rollout
  engines load with a standard load format; dummy loading is rejected. Both sides trust
  this prior, so no model-sized baseline fingerprint is computed.
- `--update-weight-disk-dir` must be visible to every trainer and serving rank.
- Install the prebuilt nvCOMP library without changing the image's dependency closure:
  `python -m pip install --no-deps nvidia-libnvcomp-cu13==6.0.0.12`. No custom C++/CUDA
  extension is built.
- Launch validation (`miles/utils/gpu_delta/admission.py`) requires separate training
  and rollout GPUs, the direct Megatron exporter, retract pauses and a static MTP draft,
  and rejects LoRA, PD, `--check-weight-update-equal` and trainer fault tolerance.
  Rollout weights must stay resident (no `--offload-rollout`). Supported serving layouts
  and parallelism are part of the receiver contract.

The codecs are `snappy-zstd`, `lz4-zstd`, `lz4`, `bitcomp-zstd` (default) and `bitcomp`.
`*-zstd` wraps the inner codec with GPU Zstd on the trainer and CPU Zstd on the receiver;
`lz4` and `bitcomp` skip both outer stages. Bitcomp always uses the sparse algorithm with
lossless U8 bytes and no quantization; LZ4 uses byte input with bitshuffle disabled.
Neither adds a weight-layout transform. Scalars and vectors carry uncompressed XOR masks;
the `overwrite` encoding belongs to disk-delta. The trainer reads `GPU_DELTA_CODEC` and
`GPU_DELTA_INITIAL_SYNC_CODEC` once at launch (see the development guide); receivers take
the codec from each authenticated manifest. GPU-delta does not require Blackwell: only
Snappy and LZ4 decoding use the SM100/SM103 decompression engine, and model or backend
kernels keep their own hardware requirements.

`miles.utils.gpu_delta` owns canonical publication, encoding and the reusable CUDA apply,
codec, memory and host-staging primitives. The training protocol, engine activation, eval
snapshots and metrics live in `miles.backends.training_utils.weight_update.protocols.gpu_delta`,
and Megatron export ownership in `miles.backends.megatron_utils.update_weight.gpu_delta`.
SGLang imports the receiver hooks of `miles.backends.sglang_utils.gpu_delta` directly, so
every serving process needs the matching Miles package.

## Startup delta

GPU-delta always captures HF baseline version 0, then publishes and applies the loaded
trainer's version 1 before the first rollout. This includes a trainer resumed from a
checkpoint that differs from the HF base. Later updates start at version 2; transfer
versions are independent of restored optimizer and rollout steps.

Initial and learned publications both default to Bitcomp-Zstd. Phase-specific codec
overrides follow the initial-sync call, not transfer-version arithmetic. Encoder setup is
lazy and cached before publication or export; receivers cache decoders by the manifest
codec during preparation. The canonical plan and version-stream identity do not depend on
the codec. The startup delta may be large: it reuses the ordinary transport and
publication directory and is not a full-checkpoint bootstrap.

## Trainer pipeline

The trainer keeps old canonical weights in pinned CPU RAM and stages each new export
there asynchronously. At expert TP 1, routed experts stay on their exporter EP/EDP owner;
expert TP above 1 keeps the existing gather-before-convert path. Non-routed tensors use
cached PP-local layer ownership over TP × CP × DP ranks and gather only the required TP
shards to each owner. Contiguous owner slices order decoder and MTP layers before the
embedding, LM head and remaining tensors, so the tail pairs fewer layers with those
higher-precision tensors.

Immutable owner geometry partitions scalar/vector masks and matrix batches once, before
learned updates. One CPU worker writes the raw scalar/vector XOR masks while the GPU
compresses matrices. Matrix batches follow baseline callback order: when a batch's
export D2H event completes, the encoder uploads its old bytes and computes the XOR and
change counts. An immediately ready batch reuses the current GPU tensors of the active
export bucket; queued batches upload their pinned current snapshots, so retained
exporter allocations stay bounded to one encoding batch.

The encoder compresses all independent inner-codec frames of a batch in one call while
later exports continue. Unchanged frames are omitted; incompressible changed frames keep
the selected codec. Compact, 16-byte-aligned inner-codec arenas stay in HBM across
batches. Wrapped codecs then compress all of an owner's inner arenas together with GPU
Zstd, in independent outer chunks of at most 1 MiB that preserve tensor boundaries. Both
paths pack once and return one aligned slab to pinned CPU RAM for hashing and file
writes. There is no intermediate inner-codec host slab, CPU compression or raw matrix
fallback.

`--update-weight-buffer-size` bounds each canonical input batch; a larger single tensor
stands alone. An owner's complete compact inner-codec payload must fit in HBM through
finalization, and host snapshots must fit in RAM; there is no out-of-memory fallback.
Inner frames are 1 MiB (the benchmark's `--frame-bytes` accepts positive sizes up to
4 MiB, subject to nvCOMP alignment); the receiver checks actual encoded and decoded
lengths against its device limit.

## Publication format

`manifest.json` records the codec, explicit `frame_bytes`, publication identity, payload
files and authenticated sidecars. `catalog.json` holds immutable tensor names, dtypes,
shapes and canonical encodings; `metadata.safetensors` holds the named numeric tensor,
frame, outer-chunk and zero-gap columns that receivers use directly. Only changed tensors
contribute work, and the Triton apply kernels stay independent of model shapes. The
receiver contract defines the outer descriptor of unwrapped codecs.

SHA-256 authenticates final owner files by default. Updates and checkpoint or recovery
artifacts trust the immutable HF prior; they do not hash old or new weights or
intermediate inner-codec bytes. The manifest uses sorted `orjson` serialization, while
the canonical plan digest keeps its compact sorted JSON encoding.

Miles derives the canonical plan from the immutable checkpoint inventory and export
format and never requests layout plans or tensor views from rollout; connection queries
establish only participant identity and version state. The global canonical inventory
must have complete, unique owner coverage: duplicate exports, including overlapping
PP/MTP names, are rejected rather than deduplicated.

## Activation and failures

One Miles coordinator exclusively owns the engine endpoints during an update and submits
one `update_weights_from_gpu_delta` request per engine. Engines prepare and activate
independently; each waits for its own original ranks before resuming, so independent
replicas may briefly serve different versions. Ordinary updates, recovery and eval all
use this path with the same explicit payload-hash policy. The receiver's phases are
defined in its
[lifecycle contract](https://github.com/radixark/miles/blob/main/miles/backends/sglang_utils/gpu_delta/README.md#ownership-and-lifecycle).

The trainer drains every engine request before handling failures. A failed request has
an unknown phase, so it never authorizes a client abort or retry; an uncertain receiver
is replaced, never reconciled or given an XOR replay. Successful peers keep their target
version. Successful activation alone does not prove full weight-content equality.

For an ordinary delta of N matrix bytes, the trainer transfers N export bytes D2H and N
previous-snapshot bytes H2D, plus current-snapshot H2D for batches that cannot reuse the
active export bucket; final encoded bytes return D2H. Small masks avoid matrix H2D
uploads. Recovery, when requested, additionally uploads the retained HF/current pair
without a second export D2H. Export, compression, publication and activation enclose
trainer blocking. Receiver preparation runs before the pause; inner decompression and
in-place mutation run within it. Do not sum nested phases or add trainer and receiver
times from different workloads.

Publication diagnostics keep `metadata_gather_s` in each rank's producer metrics and
`manifest_seal_s` on the returned descriptor. The first is that rank's owner-metadata
gather wall time, not payload transfer; the second is root's manifest validation,
serialization, hashing and exclusive write/link span. Neither adds a collective or a
training-metric reduction.

## Recovery and checkpoint companions

Each trainer owner keeps an immutable copy of its canonical HF-base bytes besides the
rolling committed snapshot and the pending target. Owners compress HF-base to current
bytes only when a replacement engine, checkpoint or eval snapshot needs them, using the
initial-sync codec and the already captured target. Healthy updates perform no recovery
compression, and the result is cached for reuse at the same target step.

A fresh engine starts from the HF checkpoint; its next update supplies a base-relative
payload while existing engines receive the rolling delta, through the same endpoint and
captured participant identities. Recovery keeps the receiver cache for later updates.
The trainer settles every engine request before committing its rolling baseline, which
requires live trainer snapshots and the same canonical plan. With `--use-fault-tolerance`
and the `weight_update` component, an in-flight receiver failure or lost reply marks only
its rollout cell failed, and the trainer commits the target for healthy receivers and
future replacements; without it, a receiver error fails the update. Existing Miles fault
tolerance replaces the failed cell while healthy cells keep their progress, and the next
update initializes the replacement from the HF baseline without replaying the uncertain
XOR. One trainer-side sender serves the whole receiver pool; P2P's partial-target sender
dispatch is not used. Losing the trainer's canonical baseline requires a job or
checkpoint restart, and live trainer failover is not supported. Existing all-target
failure and dedicated-eval fault-tolerance restrictions are unchanged.

On an ordinary Megatron save, the stable target is captured and compressed ahead of the
next weight sync, which reuses that capture; a final save also produces a companion when
no later sync runs. Owner payloads, `catalog.json`, `metadata.safetensors` and
`manifest.json` live in the checkpoint iteration's `gpu_delta/` directory, and
`READY.json` appears only after the Megatron writer finishes. The manifest records the
HF-base path, training step and separate transfer version. Ship the completed directory
together with the unchanged HF base that both sides trust.

## Dedicated eval snapshots

Dedicated GPU-delta eval uses the ordinary asynchronous snapshot queue, including its
in-flight limit, overflow policy and snapshot retention. Instead of a full HF export, an
eval snapshot holds a `gpu_delta/READY.json` descriptor that references the target's
delta from the immutable HF checkpoint in the shared publication directory. The trainer
reuses its HF-relative recovery payload; eval has no separate rolling baseline or
transfer counter. Full HF exports remain available for custom checkpoint evaluators.

The eval executor holds its existing eval lock across pinning and generation. To pin a
new snapshot, the controller calls the ordinary update API twice, each time with a fresh
session ID: it applies the previous snapshot again to return to the HF base, then applies
the new one. No eval generation runs between those calls. Scalars, vectors and compact
UE8M0 scales also carry XOR masks, so cancellation is the same operation, and derived
scalar values are prepared before each pause. Skipped eval points need no intermediate
updates.

Queue cleanup removes staging descriptors; the publications stay in shared storage for
later cancellation, with the same lifetime as other GPU-delta publications. A failed or
uncertain activation is never retried on that receiver incarnation and requires a fresh
engine. Dedicated eval gains no automatic fault tolerance.

## Training metrics

Successful updates publish `perf/gpu_delta/*` through the existing tracking backend,
including W&B. Receiver summaries reuse the activation broadcast and producer summaries
reuse the existing owner gather, so no engine RPC, device synchronization or collective
is added. Distributions have `/min`, `/p50` and `/max`.

- `receiver_<field>` reduces receipt field `<field>` over every original rank's RESUMED
  receipt, and appears only when every rank reports it. Field meanings are in the
  receiver's [receipt tables](https://github.com/radixark/miles/blob/main/miles/backends/sglang_utils/gpu_delta/README.md#configuration-and-observations).
  The rank work totals `host_rank_outer_zstd_{encoded_bytes,decoded_bytes,tensors,frames}`,
  `host_rank_allocation_{calls,bytes}`, `host_rank_{arena,capacity,private_capacity}_bytes`
  and `host_rank_staging_copy_bytes` also add `/sum`.
- `creator_<field>` reduces `<field>` over the ranks that created an engine-host cache.
- Unprefixed cache capacities count once per engine-host cache, including reuse.

| Metric | Meaning |
|---|---|
| `receiver_ranks`, `receiver_engines`, `base_version`, `target_version` | Coverage (original ranks with RESUMED receipts, engines with server timings) and publication versions, independent of the rollout step |
| `receiver_scheduler_pause_s` | Each original process's completed pause from its RESUMED scheduler timestamps: reader fence, retraction/cache flush, application, its own engine's APPLIED wait and resume. Failed or open intervals are never reported |
| `receiver_reader_fence_s` | Pause start through reader-fence completion |
| `receiver_host_prepare_s`, `receiver_host_metadata_prepare_s`, `receiver_host_metadata_wait_s` | Background preparation before the pause, including small metadata preparation and its own stream waits; the nested metadata span; those waits alone |
| `receiver_host_manifest_read_parse_s`, `receiver_host_plan_validate_s`, `receiver_host_tensor_prepare_s`, `receiver_host_raw_pack_s` | Manifest, canonical-plan qualification, local tensor planning and CPU mask packing spans |
| `receiver_host_rank_prepare_s` | Publication access and CPU-child work or follower wait, rank-local canonical admission, mapping and local input preparation |
| `receiver_host_apply_kernel_prepare_s`, `receiver_apply_kernel_cache_misses`, `receiver_apply_kernel_cache_reused` | Fixed-family kernel cache lookup or first compilation/loading before the pause; misses count module warmup requests, not compiler runs (a request can load a disk binary); reused marks a cache hit |
| `receiver_paused_setup_host_s`, `receiver_paused_apply_host_wall_s` | Paused slot allocation, pointer uploads and binding; complete paused apply through GPU completion and scratch release |
| `receiver_host_matrix_enqueue_s`, `receiver_host_apply_completion_wait_s` | Host enqueue of the paused matrix batches; host wait for paused GPU completion |
| `receiver_host_plan_cache_reused`, `receiver_host_batch_plan_reused` | Per-rank reuse of the admitted static plan and of the active-geometry cache |
| `receiver_compressed_batches`, `receiver_compressed_tensors`, `receiver_layers_per_batch`, `receiver_decoded_zero_ranges`, `receiver_decoded_zero_bytes` | Actual batches and tensors, configured layers per batch, and omitted canonical ranges zeroed before decode |
| `receiver_h2d_bytes`, `receiver_raw_h2d_bytes`, `receiver_decoder_metadata_h2d_bytes`, `receiver_apply_metadata_h2d_bytes`, `receiver_apply_descriptor_h2d_bytes`, `receiver_decoder_metadata_uploads` | Explicit uploads: the total (small masks, metadata, CUDA compressed input and CPU-resident weight mirrors) and its raw and metadata components; descriptor bytes are part of apply metadata. Decoders upload descriptors once in preparation and output pointers once in the pause; CUDA also uploads input pointers |
| `receiver_de_host_input_bytes` | Compressed bytes DE reads directly from host memory; a different traffic category from `receiver_h2d_bytes`, not a bandwidth estimate |
| `receiver_decoded_buffers`, `receiver_decoded_scratch_bytes`, `receiver_decoder_workspace_bytes` | Per-rank prepared capacities, not peak HBM; decoded scratch totals both paused slots |
| `receiver_host_encoded_cache_{created,reused,wait_s}`, `receiver_host_prepared_cache_{created,reused,wait_s}` | Engine-host creator and follower flags and the owner wait; the prepared image is built with the encoded cache, so its values match |
| `receiver_host_encoded_caches`, `host_encoded_cache_creators` | Engine-host caches in this update (not physical nodes) and the ranks that built one |
| `creator_host_encoded_cache_{read_hash_s,read_worker_sum_s,sha256_worker_sum_s,build_s,allocation_s}`, `creator_host_encoded_cache_{hash_bytes,hash_files,allocation_calls,allocation_bytes}/sum` | Creator-only cache work, excluding reusers' zero counters. `read_hash_s` is the submission/join wall time of parallel owner-file verification; the worker sums add overlapping intervals and must not be added to it. All file tasks drain before READY or failure |
| `host_encoded_cache_capacity_{bytes,generation}`, `host_prepared_cache_{capacity_bytes,capacity_generation,arena_bytes}` | Retained tmpfs capacity, growth generation and prepared extent per engine-host cache; byte fields add `/sum`. Independent engines may retain duplicate encoded bytes |
| `creator_host_prepared_cache_{layout_s,build_s,allocation_s}`, `creator_host_prepared_cache_allocation_{calls,bytes}/sum` | Prepared-image creator spans and allocation work, once per engine-host |
| `receiver_host_rank_outer_zstd_{decode_s,worker_decode_sum_s,encoded_bytes,decoded_bytes,tensors,frames}` | Actual outer-Zstd work: one owner per engine-host decodes the full image; followers and unwrapped codecs report zero. `decode_s` is the wrapped path's wall time including submission, raw copies and joins; the worker sum adds each worker's native decode and scatter |
| `receiver_host_rank_cpu_workers` | Each rank's CPU worker count; zero on followers |
| `receiver_host_rank_{arena_bytes,capacity_bytes,allocation_s,allocation_calls,allocation_bytes,capacity_generation,mapping_reused}` | Each rank's selected input extent, mapped capacity and growth work. CUDA ranks map the same shared pages, so their sums count aliases and their allocation calls count registrations; DE inputs are separate allocations. Fitting warm updates allocate or register nothing new |
| `receiver_host_rank_private_capacity_bytes`, `receiver_host_rank_staging_copy_{s,bytes}` | DE-private input storage and the selected-span copy into it; zero for CUDA decode |
| `host_input_capacity_bytes/sum` | Shared prepared pages once per engine-host plus DE rank-private capacity; excludes temporary native Zstd output slabs and is not total host RAM |
| `engine_coordinator_{prepare,apply,resume,activation}_s` | Server-side tokenizer phase spans of each engine's single update request, excluding optional resource clearing |
| `coordinator_activation_s` | Trainer-side await of all engine requests; no global phases are inferred from overlapping engine lifetimes |
| `activation_join_s` | This trainer's wait, after publication and any recovery compression, for the settled activation summary |
| `recovery_owner_encode_s`, `recovery_load_s` | This owner's HF-relative recovery compression during the update (zero when none); activation of replacement engines, present only when replacements exist |
| `producer_ranks`, `producer_prefix_{producer_wall_s,export_staging_wait_s,encoding_tail_wait_s,owner_seal_s}` | Owner count and per-owner prefixes that end at the existing pre-publication owner gather, before receiver activation: update start to sealed shard, the final export D2H fence, the encoding, finalization and write tail after export, and shard sealing |
| `payload_{tensor_count,canonical_bytes,changed_bytes,raw_tensor_count,raw_changed_tensors,raw_bytes,matrix_inner_arena_bytes,matrix_payload_bytes}` | Owner payload totals; see below |
| `trainer_logging_rank`, `trainer_logging_rank_blocked_s` | The logging trainer's rank and its interval from `begin_sync` through the existing final trainer barrier. It excludes actor reconnect/cleanup and controller orchestration and is not an all-trainer maximum |

Payload totals count each owner file once, not receiver fanout, and reuse the existing
owner gather without another payload scan, collective or CUDA synchronization.
`perf/update_weights_wire_bytes` sums payload files across owners: compressed matrix
data, uncompressed scalar/vector XOR masks and alignment padding, excluding the JSON
manifest. `payload_matrix_inner_arena_bytes` is the aligned inner-codec size before
optional Zstd and is not additive with wire bytes. `perf/update_weights_density` is
changed over canonical bytes, and `perf/update_weights_gpu_delta_s` is the local
interval from `begin_sync` to the activation commit, before the final barrier.

All seconds use local monotonic clocks. Nested phases are not additive, and no metric
implies GPU-idle time or an RL throughput improvement. After the final trainer barrier,
the protocol logs the summary on the usual metrics rank (TP0, last PP stage, effective
DP-CP rank 0) on the existing `rollout/step` axis of the last trained rollout, including
a final update that no train call follows. Initial sync has no trained rollout step: the
metrics rank logs only its payload breakdown (masks, matrix payload, file alignment,
canonical and changed bytes, inner arena) to the process log and never invents a step. A
tracking failure is logged with version context and never retries an applied publication.
The next train call cannot emit the same summary again.

## Limits

- Receiver admission, supported layouts and excluded serving features (pipeline
  parallelism, CP decode-attention-TP caches, quantized replicated Q projections, EPLB,
  redundant experts and elastic joins, among others) are listed in the receiver contract.
- Shared publications last for the run; no automatic stale-file garbage collection is
  added.

## Related guides

- [Disaggregated RL Rollout](/advanced/disaggregated-rollout)
- [P2P Weight Transfer](/advanced/p2p-weight-transfer)
- [Fault Tolerance](/advanced/fault-tolerance)
