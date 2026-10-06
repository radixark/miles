# GPU-delta development benchmark

The experimental `--update-weight-transfer-mode gpu-delta` uses the paired
SGLang `update_weights_from_delta` API. `disk-delta` remains a separate checkpoint
handoff path. GPU-delta supports `snappy-zstd` (default) and `lz4-zstd`, with only
`--update-weight-delta-encoding xor`; `overwrite` is supported by disk-delta only.

The backend-neutral codec and publication modules live in `miles.utils.gpu_delta`.
Training protocol, session, and metrics live in
`miles.backends.training_utils.weight_update.protocols.gpu_delta`; Megatron export
ownership stays in `miles.backends.megatron_utils.update_weight.gpu_delta`.
Feature-only fast tests use matching `gpu_delta/` subdirectories, while the manual
benchmarks remain in this directory.

## Environment

Use a CUDA 13 Miles development image, the paired SGLang branch on `PYTHONPATH`,
and eight Blackwell GPUs. The receiver benchmark starts one TP8/DP8/EP8 engine
with GLM5.2 NVFP4 W4A16, CuTe DSL MoE, no MoE A2A and static bundled MTP.
The original checkpoint must already be available and is never modified.

Install the prebuilt encoder/decoder without changing the image dependency closure:

```bash
python -m pip install --no-deps nvidia-libnvcomp-cu13==5.3.0.16
```

The optional CPU correctness oracles require `python-snappy`, `lz4` and
`zstandard`; production inner compression/decompression uses nvCOMP. LZ4
oracles use raw blocks without a prepended uncompressed-size header.

No custom C++/CUDA extension is built. Both sender compression stages use CUDA
SMs. Receiver Zstd decompression runs on CPU; either inner codec explicitly
requests the Blackwell hardware decompression engine and rejects unsupported
hardware or allocation modes. DE reads directly from each rank's original host arena.
LZ4 uses byte input with bitshuffle disabled; there is no extra layout transform.

These `GPU_DELTA_*` variables are development/debug controls, not a stable
user-facing configuration API. Runtime defaults are sufficient for normal use.

| Development variable | Meaning |
| --- | --- |
| `GPU_DELTA_CODEC=snappy-zstd` | Default; `lz4-zstd` selects LZ4 instead. Frozen at launch on both sides and matched against the receiver plan and immutable publication. |
| `GPU_DELTA_SORT_BEFORE_HW_DECOMPRESS=0` | Receiver only. Set `1` to let nvCOMP sort chunks during paused DE submission; default off for both codecs. It does not change publication bytes or move sorting into preparation. |
| `GPU_DELTA_TIMING=1` | Optional per-phase CUDA events. Default off; instrumentation can perturb timing. |
| `GPU_DELTA_CPU_WORKERS=32` | CPU workers per rank. Each engine-host cache creator also uses its pool for parallel owner-file read/hash before local outer Zstd. Two EP4 engines have eight pools (up to 256 workers at the default). |
| `GPU_DELTA_HOST_CACHE_DIR` | Tmpfs base for engine-local host arenas; defaults to `/dev/shm/sglang-gpu-delta-<uid>`. Engines use separate subdirectories, identities and locks; only ranks of the same engine share an arena. |
| `GPU_DELTA_SOURCE_DIGEST` | Optional producer-benchmark provenance annotation; unset by default. Does not configure the transport. |

For Ray launches, set the job `runtime_env` environment or use the provided
`execute_train(extra_env_vars=...)` path. The submitting shell alone does not
forward arbitrary variables into existing workers. Trainer-only settings can
use `--train-env-vars`; receiver profiling needs the timing setting on rollout
actors too. Set the DE sorting control on rollout actors. The five-layer GPU-delta
E2E forwards both codec and sorting controls and checks
every learned publication's protocol and codec. No old codec/encoder setting is migrated.

The five-layer test is dedicated to GPU delta: it prepares checkpoints and data,
enables the initial sync, then runs four rollouts to exercise three learned
publications with deterministic rewards. The change gate excludes the startup
publication, so startup-only changes cannot satisfy it. It takes no CLI options:

```bash
python tests/e2e/megatron/test_glm5_2_744b_a40b_5layer_nvfp4_w4a16.py
```

Publications use `/root/shared_data/<run_id>/gpu_delta`. Engine replacement and
attention FP8 conversion are disabled for this test.

## Startup delta

`--update-weight-delta-initial-sync` is an opt-in flag for both `disk-delta` and
`gpu-delta`. By default, the initial call captures HF baseline version 0 without
replacing rollout weights. Enable the flag when the loaded trainer
weights differ from that HF checkpoint: the initial call then publishes and
applies version 1 before the first rollout. Subsequent updates start at version 2;
these transfer versions are independent of restored optimizer/rollout steps.
This requires real common HF weights in SGLang, not dummy loading. The startup
delta may be large. It reuses the ordinary transport, publication directory and
(disk-delta only) host-local checkpoint; it is not a full-checkpoint bootstrap.
For a new disk-delta stream, use fresh host-local checkpoint state: an already
advanced checkpoint is not reset by the version-0 pull.

Completed training updates report the final delta payload size through the existing
`perf/update_weights_wire_bytes` metric to W&B when enabled, with the normal trainer
namespace. This sums payload files across every owner, including compressed matrix
data, raw scalar/vector replacements and alignment padding, excluding the JSON
manifest. It reuses the existing owner gather; no codec-specific metric, additional
payload scan, collective or CUDA synchronization is required.

## Fixed producer and receiver pipeline

The sender retains old canonical weights in pinned CPU RAM and stages each new
export there asynchronously. At expert TP=1, routed experts stay on their exporter
EP/EDP owner; expert TP>1 keeps the existing gather-before-convert sender path.
Non-routed tensors use cached PP-local layer ownership over TP × CP × DP ranks
and gather only the required TP shards to each owner.
Immutable owner geometry partitions scalar/vector bypass and matrix batches once
before learned updates.
Raw target writes run on one CPU worker concurrently with matrix GPU compression;
there is no scalar/vector branch inside the matrix compression loop.

After export D2H completes, each bounded name-sorted matrix batch uploads old/new
bytes, computes XOR/change counts, and compresses all independent Snappy or LZ4 frames
in one call. Unchanged frames are omitted; incompressible changed frames retain
the selected inner codec. Compact, 16-byte-aligned inner-codec tensor arenas remain
in HBM across batches.
The sender then compresses **all** owner inner-codec arenas together with GPU Zstd,
using independent outer chunks of at most 1 MiB while preserving tensor boundaries.
Only the final encoded slab returns to pinned CPU RAM for file hashing/writing.
There is no intermediate inner-codec host slab, CPU compression, or raw matrix fallback.

The existing `update_weight_buffer_size` bounds each canonical input batch; a
larger single tensor stands alone. The full compact owner inner-codec payload must fit
HBM for the outer call. Host snapshots are assumed to fit RAM; no OOM fallback is
implemented. Production and fixture creation default to 1 MiB inner frames. The
producer benchmark can select 4 MiB with `--frame-bytes 4194304`; the receiver
benchmark accepts prepared fixtures declaring 64 KiB, 1 MiB or 4 MiB inner frames.
The receiver checks actual encoded and decoded frame sizes against its hardware
limit. Outer Zstd chunks remain fixed at 1 MiB. The low-level encoder's 2 MiB
control is producer-only.

Protocol 4 records the selected `codec` (`snappy-zstd` or `lz4-zstd`),
explicit `frame_bytes`, natural tensor
identity and outer chunk offsets/lengths. SHA-256 authenticates final owner files;
old/new weights and intermediate inner-codec bytes are not hashed. The receiver reads
and verifies immutable encoded files once per engine-host, then every rank
CPU-decompresses its local tensors directly into its own original HOST_NUMA
allocation during background preparation. Each allocation requests the hardware-
decompression flag and checks actual pointer capability; no CUDA handles are
exported or imported. The initial host capacity fits the required extent rounded to allocation
granularity; fitting updates reuse it, and growth reserves twice the new requirement.
Wrapped-file staging remains a separate retained tmpfs mapping.

Preparation builds CPU plans and small GPU metadata/workspace, and uploads raw
scalar/vector targets without writing model weights. After the serving pause and
reader fence, the receiver allocates two decoded HBM slots sized for the largest
batch. Hardware Snappy or LZ4 reads compressed bytes directly from the rank-owned host arena;
there is no encoded HBM ring or explicit compressed H2D copy. A DE stream decodes
the next batch while the apply stream checks status and applies the current masks.
Events protect decoded-slot and status-row reuse. Slots are released after GPU
completion before resume; static plans and host arena capacity remain cached.
`GPU_DELTA_LAYERS_PER_BATCH` defaults to 1 and groups adjacent active model layers;
embeddings, the language-model head and other standalone tensors form three
separate groups, omitting empty groups. GPU Zstd decompression is not part of this path.

Miles negotiates the immutable plan and original participant cohort once when
connecting; learned updates reuse that plan rather than sort and hash it again.
The receiver advertises an opaque engine-host encoded-cache identity. Miles supplies
that engine's union of canonical tensor names; each rank decodes only its local
bindings. Ranks share encoded-file verification, while independent engines have
separate caches. No host-wide cache lock or release barrier couples those engines.
Each rank bounds queued decode futures to `4 * GPU_DELTA_CPU_WORKERS`; it does
not enqueue one unbounded future for every tensor/frame.
The global canonical inventory must have complete, unique owner coverage. Duplicate
exports, including overlapping PP/MTP names, are rejected rather than deduplicated.

One Miles coordinator exclusively owns the original engine endpoints during an
update; concurrent administration, other mutations and external pause/resume are
unsupported. Each engine independently prepares while its old version serves,
waits for its own original ranks to be PREPARED, then calls
`update_weights_from_delta(session_id)`. That engine closes admission, pauses,
fences readers, retracts requests, flushes caches and applies. A failed reader
fence never reclaims KV. Miles awaits the successful all-rank apply reply before
calling `resume_weights_from_delta(session_id)`; the engine records the new
version and resumes without waiting for other engines. Ordinary pause/continue
APIs remain unchanged. Each engine's local TP/EP participants still synchronize
for safe activation; independent replicas may temporarily serve different versions.

The trainer awaits every engine coroutine and advances its pinned baseline only
after all original RESUMED receipts. On failure it still settles the other engine
tasks before raising. A preparation failure aborts only that engine's preparation;
an uncertain reply after apply dispatch is terminal for that engine: do not abort,
replay XOR, automatically resume or recover. Other engines may already have resumed
successfully; failure does not roll them back or report overall success. A failed
update never advances the common sender baseline. Miles is the sole ordered caller;
these APIs do not support replay, reordered calls or concurrent administration.
Successful activation does not prove full weight-content equality.
External cancellation can leave outstanding remote work; it is incomplete and
does not authorize automatic retry, cleanup or recovery.

The sender's N matrix bytes still incur new export D2H plus old/new H2D (3N total),
followed by final compressed D2H. Raw bypass avoids both matrix H2D uploads. Export,
bulk compression, publication and activation enclose trainer blocking. Ready
owner batches compress on a private worker/stream during later exports; outer
Zstd runs after all inner batches finish. Receiver preparation runs before pause;
direct-host DE and in-place mutation run within the serving pause.
Do not sum nested phases or add sender/receiver times from different workloads.

Publication diagnostics retain `metadata_gather_s` on each sender rank and
`manifest_seal_s` on the returned descriptor. The first is that rank's existing
owner-metadata gather wall time; it does not measure payload transfer. The second
is root's manifest validation, serialization, hashing and exclusive write/link
span. Diagnostic benchmarks may collect the local values after their timed span;
these fields add no collective or training-metric reduction. The manifest uses
sorted orjson serialization, while canonical plan-digest JSON remains unchanged.

See [bench_gpu_delta_producer.md](bench_gpu_delta_producer.md) for the producer
benchmark. Compare new evidence with the saved matched-workload baseline, keeping source,
workload, raw timing rows and transfer/residency metrics separate.

## Training metrics

Successful actor weight updates publish `perf/gpu_delta/*` through the existing
tracking backend, including W&B. No extra engine RPC, device synchronization,
or collective is added: receiver summaries reuse the activation broadcast and
producer prefixes reuse the existing owner gather.

- `receiver_scheduler_pause_s/{min,p50,max}` joins each original process's
  completed RESUMED receipt's scheduler timestamps. It includes its reader fence, retraction/cache
  flush, application, its own engine's APPLIED wait and resume. Failed/open intervals are
  never reported as completed pauses.
- `receiver_reader_fence_s` and `receiver_paused_apply_host_wall_s` have separate
  rank distributions. `receiver_host_prepare_s` covers background preparation
  before pause, including small metadata preparation and its own stream waits.
  `receiver_host_metadata_prepare_s`/`receiver_host_metadata_wait_s` expose those
  nested spans. `receiver_paused_setup_host_s` includes decoded allocation,
  pointer uploads and cold kernel work; `receiver_paused_apply_tune_s` isolates
  tuning. Other `receiver_host_*` summaries retain their source span names.
- `creator_host_encoded_cache_*` samples the one encoded-cache creator per
  engine-host, excluding reusers' zero counters. `host_encoded_cache_creators`
  records coverage. `read_hash_s` is elapsed submission/join wall time for parallel
  owner-file verification; `read_worker_sum_s` and `sha256_worker_sum_s` sum
  overlapping worker intervals and must not be added to that wall span. These
  timings, metadata-validation and cache-build spans have `{min,p50,max}`; hash
  bytes/files, frame validations and allocation calls/bytes have `/sum`. All
  file tasks drain before READY or failure; rank-local decode uses verified bytes.
- `host_encoded_cache_capacity_bytes/{min,p50,max,sum}` counts retained tmpfs
  capacity once per engine-host cache, including when there is no new creator.
  `receiver_host_encoded_caches` counts those caches, not physical nodes.
  Independent engines may retain duplicate encoded bytes.
- `receiver_host_rank_outer_zstd_*` includes every rank's own canonical tensors.
  Validation and worker-decode sums are summed worker elapsed intervals;
  `decode_s` is that rank's wall time including submission, raw copies and joins.
  These nested spans must not be added. Encoded/decoded bytes, tensors and frames
  have rank distributions and `/sum`; `receiver_host_rank_cpu_workers` records
  each rank's worker count. `receiver_host_rank_prepare_s` encloses cache access,
  rank allocation and local outer decode.
- `receiver_host_rank_{arena,capacity}_bytes/{min,p50,max,sum}` counts distinct
  original DE host storage on every rank. Used bytes and retained capacity stay
  separate. Rank allocation duration has `{min,p50,max}`; calls/bytes also have
  `/sum`. `receiver_host_rank_mapping_reused` reports local arena reuse. Fitting
  warm updates allocate no new arena while retaining physical capacity.
- `receiver_host_plan_cache_reused/{min,p50,max}` reports per-rank static-plan
  reuse. `creator_host_encoded_cache_frames_validate_s` is nested inside cache
  build; `creator_host_encoded_cache_frames_validations/sum` counts dynamic
  metadata validation once per newly cached publication.
- `receiver_de_host_input_bytes` counts compressed bytes read directly by DE;
  `receiver_h2d_bytes` counts explicit raw-target and metadata uploads. They are
  different traffic categories, not a throughput estimate. `receiver_raw_h2d_bytes`,
  `receiver_decoder_metadata_h2d_bytes` and `receiver_apply_metadata_h2d_bytes`
  expose those upload components; decoder descriptors use one prepare upload and
  one paused output-pointer upload.
- `receiver_decoded_buffers`, `receiver_decoded_scratch_bytes` and
  `receiver_decoder_workspace_bytes` report per-rank prepared capacities, not peak
  HBM. Decoded scratch is the total of both paused slots. Batch, layer-count,
  gap-zeroing and tuning counters retain their source names. These optional rank
  summaries use `{min,p50,max}` only when all ranks provide the field; GPU capacities
  are not summed as shared host storage.
- `engine_coordinator_{prepare,apply,resume,activation}_s/{min,p50,max}` reports
  independent per-engine RPC lifetimes. `coordinator_activation_s` is the enclosing
  trainer-side await of all engines; no global preparation/apply/resume phases are
  inferred from overlapping engine lifetimes. `receiver_engines` counts coverage.
  `producer_prefix_*` distributions end at the existing pre-publication owner
  gather, before receiver activation.
- `trainer_logging_rank_blocked_s` measures this update's `begin_sync` through
  the existing final trainer barrier on the rank selected by the training logger;
  `trainer_logging_rank` identifies it. It excludes actor reconnect/cleanup and
  controller orchestration, and is not an all-trainer maximum. The older
  `perf/update_weights_gpu_delta_s` remains the local pre-final-barrier interval.

All seconds use local monotonic clocks. Nested phases are not additive, and no
metric implies GPU-idle time or an RL throughput improvement. After the update
completes, the updater returns and drains its GPU-delta summary. The actor logs it
on the usual TP0/last-PP/effective-DP-CP0 rank at the last trained rollout's
existing step axis.
This includes the final evaluation update even when no subsequent train call
occurs. `base_version` and `target_version` identify the publication independently
of that rollout step, including asynchronous training and update intervals.
Startup baseline capture emits no completed-update metrics; without a trained
rollout, the logger never invents a completed step. A tracking submission failure
is logged with version context and does not retry an already-applied publication.
The normal training timer and other protocols' next-train metric drains are
unchanged; the next train call cannot emit the same GPU-delta summary again.

## Persistent fixture

Run from the Miles checkout with matching SGLang on `PYTHONPATH`. Use a new
output directory and persistent storage with space for an independent altered
checkpoint and three cumulative publications. Fixture creation uses one GPU.

```bash
export PYTHONPATH=/workspace/sglang/python:/workspace/miles
export GPU_DELTA_CODEC=snappy-zstd
python tests/manual/gpu_delta/bench_gpu_delta.py inventory \
  --model /models/GLM5.2-NVFP4 --output /data/gpu-delta/inventory
python tests/manual/gpu_delta/bench_gpu_delta.py fixture \
  --model /models/GLM5.2-NVFP4 \
  --inventory /data/gpu-delta/inventory/inventory.json \
  --output /data/gpu-delta/fixture --ratio 0.002 --versions 3
```

The fixture calibrates sparse finite mantissa/packed-FP4 perturbations against
plain CPU Zstd level-1 sample frames. It measures the actual selected-codec publication
ratio separately; it does not force that ratio to 0.2%. Static draft and calibration
scales stay unchanged in the proxy; native tests cover scalar/scale updates.
The altered checkpoint contains the final cumulative version and does not alias
original checkpoint files.

The builder uses the production GPU encoder: pinned old/new CPU snapshots,
bounded GPU XOR/inner-codec batches, then one owner-wide GPU Zstd pass per version.
Only compact inner-codec output survives between batches. Raw scalars/vectors bypass both
codecs. It records inner/outer bytes, alignment and final manifest/file sizes.
Fixture creation is setup, excluded from receiver timing and not a distributed
producer measurement. Native fixture tests independently replay CPU Zstd plus Snappy or LZ4
and verify exact altered targets and source/draft immutability.

Only protocol 4 publications matching the configured codec are admitted. Saved
Snappy fixtures remain usable with `GPU_DELTA_CODEC=snappy-zstd`. Generate a new
LZ4 fixture from the same model, seed and mutation settings for codec comparisons;
never relabel a Snappy payload. Confirm the altered targets match across codecs.
Obsolete codec profiles
are not accepted by production; historical artifact migration is external setup,
not a fallback in this harness.

To compare codecs, repeat inventory/fixture creation with
`GPU_DELTA_CODEC=lz4-zstd` and separate output directories. Keep the same seed,
ratio, original checkpoint and three versions. For each codec, reuse its same
fixture across receiver sorting 0/1 runs. Keep 1 MiB frames and report inner bytes,
final wire bytes, preparation and full scheduler pause separately. LZ4 and sorting
are implemented but have no native correctness or performance evidence yet.

## Full-model receiver benchmark

By default, each invocation starts one fresh TP8/DP8/EP8 GLM5.2 W4A16 engine
with static MTP, CuTe DSL MoE and no MoE A2A. It applies three cumulative immutable
publications: one first-use update and two warm updates. It saves original-rank
receipts, server logs and generation through DP routes0–7.

```bash
python tests/manual/gpu_delta/bench_gpu_delta.py run --model /models/GLM5.2-NVFP4 \
  --fixture /data/gpu-delta/fixture --output /data/gpu-delta/snappy-zstd-new
python tests/manual/gpu_delta/bench_gpu_delta.py oracle --model /models/GLM5.2-NVFP4 \
  --fixture /data/gpu-delta/fixture --output /data/gpu-delta/target-oracle
```

For two TP4/DP4/EP4 engines on one eight-GPU host, provide two ports. The first
engine uses GPUs0–3; the second uses GPUs4–7. Both use the same tmpfs base
`GPU_DELTA_HOST_CACHE_DIR`, but each engine owns a distinct subdirectory/arena.
The harness checks one arena per engine, all eight original scheduler identities, each engine's
TP/DP ranks0–3, and captures compute-process PID→GPU UUID observations. If NVML
uses host PIDs unavailable in the container's `NSpid` mapping, that join remains
explicitly unqualified; requested GPU masks are not presented as native proof. Generation
records retain engine IDs explicitly; use `(engine_id, dp_rank)` as the route key.

An EP8 fixture's view-bound plan digest does not describe EP4. First inventory
the new topology, then rebind its views into a **new** immutable fixture directory:

```bash
export GPU_DELTA_HOST_CACHE_DIR=/dev/shm/gpu-delta-benchmark
python tests/manual/gpu_delta/bench_gpu_delta.py inventory --model /models/GLM5.2-NVFP4 \
  --ports 31135 31235 --output /data/gpu-delta/ep4-inventory
python tests/manual/gpu_delta/bench_gpu_delta.py rebind --model /models/GLM5.2-NVFP4 \
  --inventory /data/gpu-delta/ep4-inventory/inventory.json \
  --fixture /data/gpu-delta/fixture --output /data/gpu-delta/ep4-fixture
python tests/manual/gpu_delta/bench_gpu_delta.py run --model /models/GLM5.2-NVFP4 \
  --fixture /data/gpu-delta/ep4-fixture --ports 31135 31235 \
  --output /data/gpu-delta/ep4-snappy-zstd-new
python tests/manual/gpu_delta/bench_gpu_delta.py oracle --model /models/GLM5.2-NVFP4 \
  --fixture /data/gpu-delta/ep4-fixture --ports 31135 31235 \
  --output /data/gpu-delta/ep4-target-oracle
```

Rebinding is CPU-only setup, excluded from update timing. It verifies source
manifest/payload hashes, exact canonical names/dtypes/shapes/encodings/byte counts
and byte order, and the fresh checkpoint-header/receiver-plan agreement. Only
view definitions and plan/stream/publication identities change; matrix frames,
raw targets and the final altered checkpoint stay unchanged. Payloads are
hardlinked on the same filesystem, with no symlink/copy fallback. `rebind.json`
records source/new hashes, unchanged non-view tensor metadata and verified payload
inode identities. Preserve both fixtures as immutable. Inventory and run use
separate fresh engine pairs; their startup/teardown is outside update timing.

Both engines negotiate one canonical publication and update independently. Each engine
resumes after its own original participants have returned APPLIED; the trainer joins
both completions before advancing its baseline. Untimed generation
visits all four DP routes in both engines after each update. EP4/EP8 comparisons
report observed differences; topology changes alone do not establish numerical
equivalence. The same-topology altered-checkpoint oracle supplies a separate
functional reference.

The oracle loads the final altered checkpoint with the original static draft;
its generation/logprobs are untimed functional evidence, not every-weight-byte
proof. Native tests establish exact payload/application behavior separately.
The five-layer W4A16 E2E exercises actual training-driven publications and
requires a changed learned update; it is not full-model RL validation.

Separate coordinator wall time, background read/hash/CPU-Zstd and small GPU input
preparation, explicit scheduler pause, hardware inner-codec decode, layout/application and
derived refresh. Pause measures the original scheduler flag-to-resume interval;
it excludes earlier prepare/status handler service and does not quantify serving
interference. Do not sum nested events or concurrent rank durations.

Each engine-host reads/hashes immutable owner files once into its encoded cache.
Every scheduler independently CPU-unwraps its local tensors into its original
DE-capable host arena and allocates two decoded HBM slots only during paused
application. Small decoder metadata/workspace is prepared earlier. Rank arenas
stay alive until their GPU readers finish; the encoded cache is released only
after the original engine cohort completes. The benchmark
generates before/after updates, not during preparation; realized serving overlap,
request latency and production throughput need separate study.
The harness terminates only its own engine processes, retains partial evidence
on failure and never releases a devbox allocation.
