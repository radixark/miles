# GPU-delta development benchmark

The experimental `--update-weight-transfer-mode gpu-delta` uses SGLang's paired
prepare/apply/resume APIs and blocking `update_weights_from_gpu_delta` endpoint. `disk-delta` remains a separate checkpoint
handoff path. GPU-delta supports `snappy-zstd` (default), `lz4-zstd` and plain
`lz4`, with `--update-weight-delta-encoding xor`. Changed scalar/vector tensors
carry complete target bytes; the `overwrite` transport option belongs to disk-delta.

The backend-neutral codec and publication modules live in `miles.utils.gpu_delta`.
Training protocol, session, and metrics live in
`miles.backends.training_utils.weight_update.protocols.gpu_delta`; Megatron export
ownership stays in `miles.backends.megatron_utils.update_weight.gpu_delta`.
Feature-only fast tests use matching `gpu_delta/` subdirectories, while the manual
benchmarks remain in this directory.

## Environment

Use a CUDA 13 Miles development image, the paired SGLang branch on `PYTHONPATH`,
and eight Blackwell GPUs. This manual workload uses GLM5.2 NVFP4 W4A16, CuTe DSL
MoE, no MoE A2A and static bundled MTP. The benchmark defaults to two
TP4/DP4/EP4 engines; one `--ports` value selects one TP8/DP8/EP8 engine.
FlashInfer autotuning and prefill CUDA graphs are disabled; BS1 decode graphs
are enabled to initialize the CuTe decode path.
The original checkpoint must already be available and is never modified.

Install the prebuilt encoder/decoder without changing the image dependency closure:

```bash
python -m pip install --no-deps nvidia-libnvcomp-cu13==5.3.0.16
```

The optional CPU correctness oracles require `python-snappy`, `lz4` and
`zstandard`; production inner compression/decompression uses nvCOMP. LZ4
oracles use raw blocks without a prepended uncompressed-size header.

No custom C++/CUDA extension is built. Sender compression uses CUDA SMs.
Wrapped codecs use receiver CPU Zstd; plain LZ4 skips both outer stages and
copies its packed inner arenas into private DE input buffers. Both inner codecs
request the Blackwell hardware decompression engine and reject unsupported
hardware or allocation modes. DE reads each rank's original host allocation.
LZ4 uses byte input with bitshuffle disabled; there is no extra layout transform.

These `GPU_DELTA_*` variables are development/debug controls, not a stable
user-facing configuration API. Runtime defaults are sufficient for normal use.

| Development variable | Meaning |
| --- | --- |
| `GPU_DELTA_CODEC=snappy-zstd` | Trainer codec for ordinary learned updates; `lz4-zstd` selects LZ4 with outer Zstd; `lz4` skips outer Zstd. Read once at launch. The receiver selects the codec from each authenticated publication. |
| `GPU_DELTA_INITIAL_SYNC_CODEC=lz4-zstd` | Trainer codec for initial sync, engine recovery and checkpoint companions; `snappy-zstd` or plain `lz4` overrides it. Recovery uses this setting even when startup sync is disabled. |
| `GPU_DELTA_SORT_BEFORE_HW_DECOMPRESS=0` | Receiver only. Set `1` to let nvCOMP sort chunks during paused DE submission; default off for all codecs. It does not change publication bytes or move sorting into preparation. |
| `GPU_DELTA_TIMING=1` | Optional per-phase CUDA events. Default off; instrumentation can perturb timing. |
| `GPU_DELTA_CPU_WORKERS=32` | CPU workers per rank. Each engine-host cache creator also uses its pool for parallel owner-file read/hash before local payload preparation. Two EP4 engines have eight pools (up to 256 workers at the default). |
| `GPU_DELTA_SKIP_PAYLOAD_HASH=1` | Shared sender/receiver opt-in: omit owner-file SHA256 generation and verification. Defaults off. The manifest declares `payload_checksum_format="none"` and null file hashes; receivers accept this only when their same flag is enabled. Manifest SHA256 and native decoder checks remain mandatory. |
| `GPU_DELTA_HOST_CACHE_DIR` | Tmpfs base for engine-host encoded caches; defaults to `/dev/shm/sglang-gpu-delta-<uid>`. Each engine has separate identities and locks. Its ranks share encoded files, while DE input arenas remain private to each rank. |

For Ray launches, set job-level variables through `runtime_env` or
`execute_train(extra_env_vars=...)`; the submitting shell does not configure
existing workers. Both trainer and rollout must inherit the shared hash flag.
Trainer-only settings can use `--train-env-vars`; sorting and receiver timing
belong on rollout actors. The five-layer E2E forwards these controls and checks
each publication's manifest and phase-specific codec.

The five-layer test is dedicated to GPU delta: it prepares checkpoints and data,
enables the initial sync, then runs four rollouts to exercise three learned
publications with deterministic rewards. The change gate excludes the startup
publication, so startup-only changes cannot satisfy it. It takes no CLI options:

```bash
python tests/e2e/megatron/test_glm5_2_744b_a40b_5layer_nvfp4_w4a16.py
```

Publications use `/root/shared_data/<run_id>/gpu_delta`. The test enables rollout
fault tolerance, restarts one engine before publication 3, and saves Megatron
checkpoints with delta companions. Attention FP8 conversion remains disabled.

## Startup delta

`--update-weight-delta-initial-sync` is an opt-in flag for both `disk-delta` and
`gpu-delta`. By default, the initial call captures HF baseline version 0 without
replacing rollout weights. Enable the flag when the loaded trainer
weights differ from that HF checkpoint: the initial call then publishes and
applies version 1 before the first rollout. Subsequent updates start at version 2;
these transfer versions are independent of restored optimizer/rollout steps.
GPU delta defaults to LZ4-Zstd for this initial publication, then Snappy-Zstd for
learned updates. Selection follows the initial-sync call, not transfer-version
arithmetic. Encoder setup is lazy and cached before publication/export; receivers
cache decoders during preparation from the manifest codec. Canonical plan and
version-stream identity do not depend on the codec.
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

## Recovery and checkpoint companions

Each trainer owner keeps an immutable copy of its canonical HF-base bytes in
addition to the rolling committed snapshot and pending target. After publishing
an ordinary delta, receiver activation starts asynchronously while each owner
compresses HF-base → current bytes with the initial-sync codec. This reuses the
captured target and compression batches; it performs no second model export or
payload gather. Compressed recovery bytes stay on their owners until needed.
Training resumes after activation and this background work have both drained.

Existing Miles rollout fault tolerance recreates failed engines from the HF
checkpoint between weight updates. On the next update, healthy engines receive
the rolling delta while fresh engines use the blocking one-shot API with the
base-relative payload. Recovery keeps the receiver cache for later updates. The
normal update RPC returns only after all engines resume, then commits the rolling
baseline. This requires live trainer snapshots and the same canonical plan.
An in-flight activation error follows the existing Miles failure path; GPU delta
does not retry an uncertain XOR or add a separate trainer recovery mechanism.

On an ordinary Megatron save, the stable target is captured and compressed ahead
of the next weight sync, which reuses that capture. A final save also produces a
companion when no later sync runs. Owner files and `manifest.json` live under the
checkpoint iteration's `gpu_delta/` directory; `READY.json` appears only after
the Megatron writer finishes. The manifest records the HF-base path, owner byte
fingerprints, training step and separate transfer version. Ship the completed
directory together with that unchanged HF base.

For inference deployment, start SGLang from the base HF checkpoint and call
`POST /update_weights_from_gpu_delta` with `{"manifest_path": "/bundle/manifest.json"}`.
It prepares, applies and releases delta resources by default, preserving an
existing caller pause. `flush_cache` defaults to `true` for KV and multimodal
caches; set it to `false` only when the caller already owns cache invalidation.
`abort_all_requests` defaults to `false`: in-flight requests are retracted and
requeued at apply. Set it to `true` to abort them after preparation succeeds.
The staged `apply_gpu_delta` endpoint accepts the same `flush_cache` and
`abort_all_requests` controls; `prepare_gpu_delta` leaves serving state unchanged.
Set `"release_state": false` to retain delta buffers and codec setup for later
updates, as Miles recovery does. The committed version and model weights remain
resident. `POST /clear_gpu_delta_state` with `{}` also releases idle delta
resources after the staged API. Keep a new deployment out of routing until its
one-shot update succeeds; an uncertain apply requires restarting from the base,
not replaying XOR on that process.

## Producer and receiver pipeline

The sender retains old canonical weights in pinned CPU RAM and stages each new
export there asynchronously. At expert TP=1, routed experts stay on their exporter
EP/EDP owner; expert TP>1 keeps the existing gather-before-convert sender path.
Non-routed tensors use cached PP-local layer ownership over TP × CP × DP ranks
and gather only the required TP shards to each owner. Contiguous owner slices
order decoder/MTP layers before embedding, LM head and remaining tensors, so
the tail can pair fewer layers with those higher-precision tensors.
Immutable owner geometry partitions scalar/vector bypass and matrix batches once
before learned updates.
Raw target writes run on one CPU worker concurrently with matrix GPU compression;
there is no scalar/vector branch inside the matrix compression loop.

Matrix batches follow baseline callback order. When a batch's export D2H event
completes, the encoder uploads its old/new bytes, computes XOR/change counts,
and compresses all independent Snappy or LZ4 frames in one call. Encoding can
overlap later exports. Unchanged frames are omitted; incompressible changed
frames retain the selected codec. Compact, 16-byte-aligned inner-codec tensor
arenas remain in HBM across batches.
For wrapped codecs, the sender then compresses **all** owner inner-codec arenas
together with GPU Zstd, using independent outer chunks of at most 1 MiB while
preserving tensor boundaries. Plain `lz4` bypasses Zstd and its metadata fence.
Both paths pack once and return one final aligned slab to pinned CPU RAM for
file hashing/writing.
There is no intermediate inner-codec host slab, CPU compression, or raw matrix fallback.

`update_weight_buffer_size` bounds each canonical input batch; a
larger single tensor stands alone. The full compact owner inner-codec payload must fit
HBM through finalization. Host snapshots must fit RAM; there is no OOM fallback.
Inner frames default to 1 MiB. The benchmark `--frame-bytes` option accepts
positive integer sizes at most 4 MiB, subject to nvCOMP alignment requirements.
The receiver also checks actual encoded and decoded lengths against its device
limit. Outer Zstd chunks remain at most 1 MiB.

The manifest records the selected `codec` (`snappy-zstd`, `lz4-zstd` or `lz4`),
explicit `frame_bytes`, natural tensor identity and outer chunk offsets/lengths.
For plain `lz4`, the outer descriptor
has `frames=[]` and equal encoded/decoded byte lengths: its payload is the
aligned inner arena directly, without trailing padding. Raw tensors and omitted
zero-XOR frames keep the same representation. SHA-256 authenticates final owner files by default.
Ordinary updates do not hash old/new weights or intermediate inner-codec bytes;
checkpoint/recovery artifacts additionally fingerprint the immutable HF base. The receiver reads
and verifies immutable encoded files once per engine-host, then every rank
CPU-decompresses (or copies plain LZ4) its local tensors into its own original HOST_NUMA
allocation during background preparation. Each allocation requests the hardware-
decompression flag and checks actual pointer capability; no CUDA handles are
exported or imported. The initial host capacity fits the required extent rounded to allocation
granularity; fitting updates reuse it, and growth reserves twice the new requirement.
Encoded-file staging uses a separate retained tmpfs mapping for every codec.

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
Participant identities include an opaque engine-host encoded-cache identity.
Ranks share encoded-file verification and prepare only their local bindings;
independent engines have separate caches and no shared release barrier.
Each rank bounds queued decode futures to `4 * GPU_DELTA_CPU_WORKERS`; it does
not enqueue one unbounded future for every tensor/frame.
The global canonical inventory must have complete, unique owner coverage. Duplicate
exports, including overlapping PP/MTP names, are rejected rather than deduplicated.

One Miles coordinator exclusively owns the original engine endpoints during an
update; concurrent administration, other mutations and external pause/resume are
unsupported. Each engine independently prepares while its old version serves,
waits for its own original ranks to be PREPARED, then calls
`apply_gpu_delta(session_id)`. That engine closes admission, pauses,
fences readers, retracts requests, flushes caches and applies. A failed reader
fence never reclaims KV. Miles awaits the successful all-rank apply reply before
calling `resume_gpu_delta(session_id)`; the engine records the new
version and resumes without waiting for other engines. Ordinary pause/continue
APIs remain unchanged. Each engine's local TP/EP participants still synchronize
for safe activation; independent replicas may temporarily serve different versions.

The trainer settles every engine coroutine before handling failures. Preparation
failure aborts only that engine's preparation; an uncertain reply after apply
requires a fresh engine before any base-relative recovery. Successful peers keep
their target version. The common sender baseline advances only after the whole
intended cohort resumes. Miles is the sole ordered caller; these APIs do not
support replay, reordered calls or concurrent administration.
Successful activation does not prove full weight-content equality.
External cancellation can leave outstanding remote work; it is incomplete and
does not authorize automatic retry, cleanup or recovery.

For the ordinary delta of N matrix bytes, the sender transfers N new export bytes D2H plus old/new
snapshots H2D (3N total), followed by final encoded D2H. Raw targets avoid both
matrix H2D uploads. Background recovery compression additionally uploads the
retained HF/current pair and returns its compressed bytes, without a second
export D2H. Export, bulk compression, publication and activation enclose
trainer blocking. Receiver preparation runs before pause;
direct-host DE and in-place mutation run within the serving pause.
Do not sum nested phases or add sender/receiver times from different workloads.

Publication diagnostics retain `metadata_gather_s` on each sender rank and
`manifest_seal_s` on the returned descriptor. The first is that rank's existing
owner-metadata gather wall time; it does not measure payload transfer. The second
is root's manifest validation, serialization, hashing and exclusive write/link
span. Diagnostic benchmarks may collect the local values after their timed span;
these fields add no collective or training-metric reduction. The manifest uses
sorted orjson serialization, while canonical plan-digest JSON remains unchanged.

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
  timings and cache-build spans have `{min,p50,max}`; hash bytes/files and
  allocation calls/bytes have `/sum`. All
  file tasks drain before READY or failure; rank-local decode uses verified bytes.
- `host_encoded_cache_capacity_bytes/{min,p50,max,sum}` counts retained tmpfs
  capacity once per engine-host cache, including when there is no new creator.
  `receiver_host_encoded_caches` counts those caches, not physical nodes.
  Independent engines may retain duplicate encoded bytes.
- `receiver_host_rank_outer_zstd_*` covers each rank's wrapped matrix tensors
  and is zero for plain LZ4. Worker-decode sums accumulate worker intervals;
  `decode_s` is the wrapped path's wall time including submission, raw copies and joins.
  These nested spans must not be added. Encoded/decoded bytes, tensors and frames
  have rank distributions and `/sum`; `receiver_host_rank_cpu_workers` records
  each rank's worker count. `receiver_host_rank_prepare_s` encloses cache access,
  metadata release, rank allocation and local payload preparation.
- `receiver_host_rank_{arena,capacity}_bytes/{min,p50,max,sum}` counts distinct
  original DE host storage on every rank. Used bytes and retained capacity stay
  separate. Rank allocation duration has `{min,p50,max}`; calls/bytes also have
  `/sum`. `receiver_host_rank_mapping_reused` reports local arena reuse. Fitting
  warm updates allocate no new arena while retaining physical capacity.
- `receiver_host_plan_cache_reused/{min,p50,max}` reports per-rank static-plan
  reuse.
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
  controller orchestration, and is not an all-trainer maximum.
  `perf/update_weights_gpu_delta_s` is the local pre-final-barrier interval.

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

## End-to-end benchmark from an HF base

The only input artifact is an existing NVFP4 serving checkpoint with bundled MTP.
Run from the Miles checkout with paired SGLang on `PYTHONPATH`:

```bash
export PYTHONPATH=/workspace/sglang/python:/workspace/miles
python tests/manual/gpu_delta/bench_gpu_delta.py \
  --model /models/GLM5.2-NVFP4 --output /data/gpu-delta/new-run
```

`--output` must be new; omit it to create a timestamped directory. Allow disk
space for an independent copy of the base checkpoint and three publications.
The source checkpoint is immutable. No Megatron checkpoint, trainer initialization
or distributed export setup is required.

One invocation performs four serial stages, releasing each stage's CUDA contexts
before the next starts:

1. Start the receiver pair, record its canonical tensor plan, then stop it.
2. Copy the checkpoint, construct three cumulative perturbations and encode
   their actual payloads with Miles' production `GpuBatchEncoder` and
   `PublicationWriter`.
3. Start fresh receivers from the original base, prepare/apply/resume each
   publication, and record all rank receipts and selected generation routes.
4. Load the final altered checkpoint into separate receivers with the original
   static draft, then compare final text, tokens and prompt/output logprobs.

Sender compression uses all visible GPUs by default; `--sender-gpus N` limits
it to the first N visible devices. Model layers, embedding, LM head and remaining tensors form ordered whole
groups, split into balanced contiguous slices using the Miles ordinary-owner
rule. Putting embedding/head at the tail leaves that slice fewer model layers
to offset their higher-precision storage. Ownership and canonical bytes per GPU are recorded
in `fixture/owner-plan.json`. Every worker reads and mutates only its assigned
ranges in the copied checkpoint and writes its own publication shard. The parent
seals the shards after all workers finish. This uses the production codec and
publication path with simple HF ownership; it does not measure Megatron export,
TP reconstruction, expert ownership or distributed gather.

Pinned old/new input batches target 512 MiB; a larger tensor stays whole. Each
GPU retains only its owner's compact inner-codec payload through finalization.
Group-count balancing reduces per-GPU residency but does not guarantee equal
byte counts or make an oversized individual tensor fit. Wrapped codecs perform
one owner-wide GPU Zstd finalization per version. Scalar/vector target bytes
bypass compression.

Defaults are Snappy-Zstd, 1 MiB inner/outer frames, three cumulative versions,
two EP4 receivers and receiver payload hashing skipped. `--codec`,
`--frame-bytes`, `--versions`, `--ports` and `--verify-payload-hash` select a run's
configuration. Fixtures always retain payload SHA256; the flag controls receiver
verification. CUDA event instrumentation is enabled for both sides so the report
contains actual phase measurements. Instrumented timings are distinct from
historical `GPU_DELTA_TIMING=0` results.

Mutations are deterministic finite mantissa/packed-FP4 bit changes seeded by
name and version. `--ratio` defaults to 0.002 and calibrates mutation density
against CPU Zstd sample frames; the actual selected-codec ratio is measured,
not forced to that value. Static draft and calibration scales stay unchanged.
The versions are constructed cumulative targets, not learned gradients or three
statistical repeats of one delta.

`REPORT.md` summarizes each version; `summary.json` retains every sender owner
and receiver rank metric, and `comparison.json` records the final selected-output
oracle. Phase logs, launch configuration, owner assignments, publications,
original-rank receipts and generation responses remain in the output directory.
A failure stops advancement and preserves partial output; it is never retried.

Report timing scopes separately:

- **Sender:** summed inner-compression CUDA events per owner, outer-compression
  CUDA events, packing/final D2H and CPU publication work. Summary columns take
  independent owner maxima, not a sum or synchronized global critical path.
  Checkpoint copy, perturbation, H2D, XOR and hashing are excluded from codec-only
  event columns. Manifest sealing is a separate parent span.
- **Receiver:** background preparation, CPU outer-decode wall time, DE-stream
  CUDA interval, matrix application and actual scheduler pause. CPU decode wall
  time includes submission/raw copies/drain; DE intervals include zero-fill,
  enqueue gaps and decode. They are not pure hardware busy-time measurements.
- **Coordinator:** complete prepare/apply/resume RPC lifetime. It overlaps rank
  work and cannot be added to preparation, decode or pause columns.

Receiver state is released after successful updates; only benchmark-owned engine
process groups are stopped. The devbox is retained. Generation is outside update
timing and checks selected outputs rather than every weight byte. This workload
has no concurrent serving traffic or real training, so it does not measure
serving interference or RL throughput. Production codec/recovery tests and the
five-layer learned-update E2E remain separate from this benchmark.
