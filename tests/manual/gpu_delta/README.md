# GPU-delta development benchmark

The experimental `--update-weight-transfer-mode gpu-delta` uses SGLang's
`update_weights_from_gpu_delta` endpoint for both rolling updates and recovery.
Miles' SGLang adapter owns preparation, application and resume within each engine.
`disk-delta`
remains a separate checkpoint handoff path. GPU-delta supports `snappy-zstd`,
`lz4-zstd`, `lz4`, `bitcomp-zstd` (default) and `bitcomp`, with
`--update-weight-delta-encoding xor`. Bitcomp always uses the sparse algorithm,
lossless U8 byte interpretation and no quantization. Changed scalar/vector tensors
carry complete target bytes; the `overwrite` transport option belongs to disk-delta.

See [GPU-accelerated weight teleportation](https://humansand.ai/blog/gpu-accelerated-weight-teleportation)
for the design and motivation.

`miles.utils.gpu_delta` owns canonical publication/encoding and reusable CUDA
apply, codec, memory and host-staging primitives. The serving adapter lives in
`miles.backends.sglang_utils.gpu_delta`; see its
[receiver contract](../../../miles/backends/sglang_utils/gpu_delta/README.md) for
layout admission, control APIs, cache lifecycle and decoder scheduling. SGLang
retains optional registration hooks, so GPU-delta engines need the matching Miles
package available in every serving process.

Training protocol, session and metrics live in
`miles.backends.training_utils.weight_update.protocols.gpu_delta`; Megatron export
ownership stays in `miles.backends.megatron_utils.update_weight.gpu_delta`.
Fast tests follow these ownership boundaries. Manual encoder, codec and receiver
CUDA tests remain separate from the benchmark.

## Environment

GPU-delta's CUDA decoder path, including the default Bitcomp-Zstd codec, does
not require Blackwell. This GLM5.2 NVFP4 W4A16 benchmark requires eight Blackwell
GPUs, a CUDA 13 Miles development image and the paired SGLang branch on
`PYTHONPATH`. It uses CuTe DSL MoE, no MoE A2A and static bundled MTP.
The benchmark defaults to two
TP4/attention-DP4/EP4 engines; one `--ports` value selects one TP8/attention-DP8/EP8
engine. On the current SGLang base, these are one replica per engine with
`attn_dp_size` equal to its TP size. The existing `dp_size` plus
`enable_dp_attention` launch settings normalize to this topology.
FlashInfer autotuning and prefill CUDA graphs are disabled; BS1 decode graphs
are enabled to initialize the CuTe decode path.
The original checkpoint must already be available and is never modified.

Install the prebuilt encoder/decoder without changing the image dependency closure:

```bash
python -m pip install --no-deps nvidia-libnvcomp-cu13==6.0.0.12
```

The optional CPU correctness oracles require `python-snappy`, `lz4` and
`zstandard`; production inner compression/decompression uses nvCOMP. LZ4
oracles use raw blocks without a prepended uncompressed-size header.

No custom C++/CUDA extension is built. Sender compression uses CUDA SMs.
Wrapped codecs use GPU Zstd compression on the sender and CPU Zstd decompression
during receiver preparation. Plain LZ4 and Bitcomp skip both outer stages.
Snappy/LZ4 automatically use the hardware decompression engine on SM100/SM103,
reading each rank's original host allocation. Other CUDA architectures and
Bitcomp use GPU compute after double-buffered H2D transfers. GPU-delta itself
does not require SM100/SM103; model/backend kernels retain their hardware requirements.
Receiver host sharing, CPU-process ownership, capacity reuse and the fixed
DE/CUDA pipelines are described in the
[receiver contract](../../../miles/backends/sglang_utils/gpu_delta/README.md#payload-and-cache-contracts).
LZ4 uses byte input with bitshuffle disabled; Bitcomp uses sparse lossless U8.
Neither adds a weight layout transform.

These `GPU_DELTA_*` variables are development/debug controls, not a stable
user-facing configuration API. Runtime defaults are sufficient for normal use.

| Development variable | Meaning |
| --- | --- |
| `GPU_DELTA_CODEC=bitcomp-zstd` | Trainer codec for ordinary learned updates; accepts the codecs listed above. `*-zstd` adds GPU Zstd wrapping; `lz4` and `bitcomp` skip it. Read once at launch. The receiver selects the codec from each authenticated publication. |
| `GPU_DELTA_INITIAL_SYNC_CODEC=bitcomp-zstd` | Trainer codec for initial sync, engine recovery and checkpoint companions; accepts the same codecs. Initial synchronization is mandatory. |
| `GPU_DELTA_SKIP_PAYLOAD_HASH=1` | Miles-only opt-in: omit owner-file SHA256 generation and send `validate_payload_hash=false` with each receiver update. Defaults off. Such publications declare `payload_checksum_format="none"` and null file hashes; SGLang accepts them only with this explicit request option and does not read the environment variable. Manifest SHA256 and native decoder checks remain mandatory. |

Receiver development settings are documented in the
[receiver configuration table](../../../miles/backends/sglang_utils/gpu_delta/README.md#configuration-and-observations).

For Ray launches, set job-level variables through `runtime_env` or
`execute_train(extra_env_vars=...)`; the submitting shell does not configure
existing workers. Miles reads the hash flag and sends the resulting
`validate_payload_hash` option to SGLang; SGLang workers do not read that flag.
Trainer-only settings can use `--train-env-vars`; receiver sorting belongs
on rollout actors. The five-layer E2E leaves both codec settings unset, using the
default Bitcomp-Zstd for initial sync and learned updates. It forwards the hash
control and checks each publication's manifest and phase-specific codec.

The five-layer test is dedicated to GPU delta: it prepares checkpoints and data,
performs the required initial sync, then runs five rollouts with deterministic
rewards. Four GPUs train and two TP2/DP1/EP1 engines use the remaining four GPUs.
Attention, routed experts and BF16 shared experts use TP2. DCP2 uses TRT-LLM DSA
prefill/decode with replicated Q projection, covering its derived Q/K weights.
GPU-delta supports DCP groups contained within attention TP and plain prefill CP.
Pipeline parallelism and CP's optional decode-attention-TP weight-slice cache remain
unsupported. This test keeps TP2/DCP2; prefill CP has unit coverage only.
The test checks every TP rank's
warm apply, including zero new Triton compilations. Its 300 ms gate covers both
rank scheduler pause and each engine's combined apply/resume interval, including
all-TP acknowledgement before admission reopens. The change gate excludes the
startup publication, so startup-only changes cannot satisfy it. It takes no CLI options:

```bash
python tests/e2e/megatron/test_glm5_2_744b_a40b_5layer_nvfp4_w4a16.py
```

Publications use `/root/delta-updates/<run_id>/gpu_delta`. The test saves Megatron
checkpoints with delta companions. It does not enable rollout fault tolerance or
exercise engine restart. GPU-delta receiver recovery uses the existing Miles
rollout/weight-update fault-tolerance lifecycle; the ordinary five-layer test does
not cover it. Attention FP8 conversion remains disabled.

## Startup delta

GPU delta always captures HF baseline version 0, then publishes and applies the
loaded trainer's version 1 before the first rollout. This includes a trainer
resumed from a checkpoint that differs from the HF base.
Disk-delta also always performs initial synchronization. Subsequent GPU-delta
updates start at version 2;
these transfer versions are independent of restored optimizer/rollout steps.
GPU delta defaults to Bitcomp-Zstd for both initial and learned publications.
Phase-specific overrides follow the initial-sync call, not transfer-version
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
manifest. Initial sync has no trained rollout step, so it logs a payload-only
summary once on the logging rank, separately for rollout and eval. The breakdown
includes raw replacements, final matrix payload, file alignment, canonical and
changed bytes, and the aligned inner-codec arena size before optional Zstd. The
inner size is not additive with wire bytes. These owner totals also accompany
trained W&B metrics under `perf/gpu_delta/payload_*`; receiver fanout is not counted.
All counters reuse the existing owner gather, without another payload scan,
collective or CUDA synchronization.

## Dedicated eval synchronization

With `--update-weight-transfer-mode gpu-delta` and dedicated eval GPUs, each
receiver pool keeps its own protocol, baseline, connection and transfer version.
Eval synchronization blocks until its receivers have applied and resumed, then
the eval executor checks that pool's version before generation. It uses live
trainer weights without exporting an HF checkpoint or requiring snapshot-save
cadence. Rollout and eval updates remain serialized; existing engine launch
settings are unchanged. Eval completion metrics use an `eval/` prefix in the
trainer namespace.

The shared eval and receiver-pool interfaces admit protocols through one capability
query. GPU-delta currently provides that capability; disk-delta keeps its existing
snapshot-based eval path.

## Recovery and checkpoint companions

Each trainer owner keeps an immutable copy of its canonical HF-base bytes in
addition to the rolling committed snapshot and pending target. Owners compress
HF-base → current bytes only when a replacement or checkpoint needs them, using
the initial-sync codec and the already captured target. Healthy updates perform
no recovery compression. The result is cached for reuse at the same target step.

For a fresh engine initialized from the HF checkpoint,
the next update supplies a base-relative payload; existing engines receive
the rolling delta. Both use the
same update endpoint and captured participant identities. Recovery keeps the
receiver cache for later updates. The sender settles every engine request before
committing its rolling baseline. This requires live trainer snapshots and the
same canonical plan. With `--use-fault-tolerance`, an in-flight receiver failure
or lost reply marks only its Miles rollout cell failed. Existing Miles FT replaces
that cell; healthy cells keep their progress. The next update initializes the
replacement from its immutable HF baseline, without replaying the uncertain XOR.
GPU-delta keeps one sender for the complete receiver pool; it does not use P2P's
partial-target sender dispatch. Losing the sender's canonical baseline requires
job/checkpoint restart; live trainer failover remains unsupported. Existing
all-target failure and dedicated-eval FT restrictions are unchanged.

On an ordinary Megatron save, the stable target is captured and compressed ahead
of the next weight sync, which reuses that capture. A final save also produces a
companion when no later sync runs. Owner payloads, `catalog.json`, named numeric
columns in `metadata.safetensors`, and `manifest.json` live under the
checkpoint iteration's `gpu_delta/` directory; `READY.json` appears only after
the Megatron writer finishes. The manifest records the HF-base path, owner byte
fingerprints, training step and separate transfer version. Ship the completed
directory together with that unchanged HF base.

Receiver update fields, pause behavior and resource release are defined in the
[control and lifecycle contract](../../../miles/backends/sglang_utils/gpu_delta/README.md#ownership-and-lifecycle).
Ordinary updates, recovery and eval all use that same activation path and explicit
payload-hash policy. An uncertain update requires replacing the failed receiver;
it never triggers reconciliation or XOR replay on that process.

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
completes, the encoder uploads its old bytes and computes XOR/change counts.
An immediately ready batch reuses current GPU tensors from the active export
bucket; queued batches upload their pinned current snapshots. This bounds retained
exporter allocations to one encoding batch. The encoder compresses all independent inner-codec frames in one call. Encoding can
overlap later exports. Unchanged frames are omitted; incompressible changed
frames retain the selected codec. Compact, 16-byte-aligned inner-codec tensor
arenas remain in HBM across batches.
For wrapped codecs, the sender then compresses **all** owner inner-codec arenas
together with GPU Zstd, using independent outer chunks of at most 1 MiB while
preserving tensor boundaries. Plain `lz4` and `bitcomp` bypass Zstd and its metadata fence.
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

The manifest records the selected `codec`, explicit `frame_bytes`, publication
identity, payload files, and authenticated catalog/metadata sidecars. `catalog.json`
holds immutable tensor names, dtypes, shapes and canonical encodings;
`metadata.safetensors`
holds named numeric tensor, frame, outer-chunk and zero-gap columns. The receiver
uses those columns directly and caches its rank-local binding table. Only changed
tensors contribute work tiles; the Triton kernels remain independent of model shapes.
For unwrapped codecs, the outer descriptor
has `frames=[]` and equal encoded/decoded byte lengths: its payload is the
aligned inner arena directly, without trailing padding. Raw tensors and omitted
zero-XOR frames keep the same representation. SHA-256 authenticates final owner files by default.
Ordinary updates do not hash old/new weights or intermediate inner-codec bytes;
checkpoint/recovery artifacts additionally fingerprint the immutable HF base.
Receiver preparation and the fixed DE/CUDA streaming pipeline are described in the
[receiver contract](../../../miles/backends/sglang_utils/gpu_delta/README.md#fixed-decodeapply-pipeline).

The agreed prior is the immutable checkpoint inventory and canonical export format.
Miles derives names, shapes, dtypes and encoding from that prior; it never requests
layout plans or tensor views from rollout. SGLang independently binds canonical
tensors to its local TP/EP slices, backend layouts and live buffers. Connection
queries establish only participant identity and version state. Learned updates
reuse the canonical contract and local bindings.
The global canonical inventory must have complete, unique owner coverage. Duplicate
exports, including overlapping PP/MTP names, are rejected rather than deduplicated.

One Miles coordinator exclusively owns the engine endpoints during an update and
submits one update request per engine. Engines prepare and activate independently;
each waits for its own original ranks before resuming. Independent replicas may
temporarily serve different versions. The receiver's internal phases are defined
in the [lifecycle contract](../../../miles/backends/sglang_utils/gpu_delta/README.md#ownership-and-lifecycle).

The trainer drains every engine coroutine before handling failures. The server
aborts preparation only before apply dispatch. An update HTTP failure has an
unknown phase and does not authorize a client abort or retry; the server may have
already completed and resumed. Successful peers keep their target version. With
weight-update FT enabled, the sender records failed cells for the existing Miles
controller and commits the target for healthy receivers and future replacements.
Without it, a receiver error fails the update. Successful activation alone does
not prove full weight-content equality.

For the ordinary delta of N matrix bytes, the sender transfers N export bytes D2H
and N previous-snapshot bytes H2D, plus current-snapshot H2D for batches that
cannot reuse the active export bucket. Final encoded bytes return D2H. Raw targets
avoid matrix H2D uploads. Recovery, when requested, additionally uploads the
retained HF/current pair without a second export D2H. Export, bulk compression, publication and activation enclose
trainer blocking. Receiver preparation runs before pause; inner decompression
(direct-host DE or H2D plus CUDA decode) and in-place mutation run within the
serving pause.
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
  pointer uploads and binding; kernel preparation happens before the pause.
  `receiver_host_apply_kernel_prepare_s` includes the fixed-family cache lookup or
  first compilation/loading. `receiver_apply_kernel_cache_misses` counts module
  warmup requests, not guaranteed compiler invocations; a request can load a disk
  binary. `receiver_apply_kernel_cache_reused` identifies a cache hit.
  `receiver_apply_descriptor_h2d_bytes` is included in total apply metadata bytes.
  Other `receiver_host_*` summaries retain their source span names.
- `creator_host_encoded_cache_*` samples the one encoded-cache creator per
  engine-host, excluding reusers' zero counters. `host_encoded_cache_creators`
  records coverage. `read_hash_s` is elapsed submission/join wall time for parallel
  owner-file verification; `read_worker_sum_s` and `sha256_worker_sum_s` sum
  overlapping worker intervals and must not be added to that wall span. These
  timings and cache-build spans have `{min,p50,max}`; hash bytes/files and
  allocation calls/bytes have `/sum`. All
  file tasks drain before READY or failure; payload preparation uses verified bytes.
- `host_encoded_cache_capacity_bytes/{min,p50,max,sum}` counts retained tmpfs
  capacity once per engine-host cache, including when there is no new creator.
  `receiver_host_encoded_caches` counts those caches, not physical nodes.
  Independent engines may retain duplicate encoded bytes.
- `receiver_host_rank_outer_zstd_*` reports actual work: one owner per engine-host
  decodes the full image and followers report zero on both decoder paths. These
  fields are zero for unwrapped codecs. Worker-decode sums include each worker's
  native batch decode and scatter into the shared image;
  `decode_s` is the wrapped path's wall time including submission, raw copies and joins.
  These nested spans must not be added. Encoded/decoded bytes, tensors and frames
  have rank distributions and `/sum`; `receiver_host_rank_cpu_workers` records
  each rank's worker count (zero on followers).
  `receiver_host_rank_prepare_s` encloses publication access and child work or
  follower wait, rank-local canonical admission, mapping and local input preparation.
- `host_prepared_cache_*` describes the shared image. Creator build/layout
  timings and allocation work count once per engine-host. Shared capacities have
  `{min,p50,max,sum}`; `host_input_capacity_bytes/sum` combines these physical pages
  with DE rank-private input capacity without counting shared mappings repeatedly.
  These capacity counters exclude temporary native Zstd output slabs and do not
  measure total host RAM usage.
- `receiver_host_rank_{arena,capacity}_bytes/{min,p50,max,sum}` describes each rank's
  selected input extent and mapped capacity. CUDA ranks map the same shared pages,
  so summing their capacities counts aliases. DE inputs are separate physical
  allocations. Rank allocation duration has `{min,p50,max}`; calls/bytes also have
  `/sum` and count CUDA registration work. `receiver_host_rank_mapping_reused`
  reports retained mapping reuse. Fitting warm updates allocate or register no
  new arena. `receiver_host_rank_private_capacity_bytes` counts only DE-private
  storage. `receiver_host_rank_staging_copy_{s,bytes}` records the selected-span
  copy into that storage; CUDA reports zero.
- `receiver_host_plan_cache_reused/{min,p50,max}` reports per-rank static-plan
  reuse.
- `receiver_de_host_input_bytes` counts compressed bytes read directly by DE;
  `receiver_h2d_bytes` counts explicit raw-target, metadata and CUDA compressed-input
  uploads. They are different traffic categories, not a throughput estimate.
  `receiver_raw_h2d_bytes`, `receiver_decoder_metadata_h2d_bytes` and
  `receiver_apply_metadata_h2d_bytes` expose raw/metadata components; the CUDA
  compressed-input component is retained as `compressed_input_h2d_bytes` in raw
  receiver receipts. Decoder descriptors use one prepare upload and one paused
  output-pointer upload; CUDA adds a paused input-pointer upload.
- `receiver_decoded_buffers`, `receiver_decoded_scratch_bytes` and
  `receiver_decoder_workspace_bytes` report per-rank prepared capacities, not peak
  HBM. Decoded scratch is the total of both paused slots. Batch, layer-count,
  gap-zeroing and kernel-cache counters retain their source names. These optional rank
  summaries use `{min,p50,max}` only when all ranks provide the field; GPU capacities
  are not summed as shared host storage.
- `engine_coordinator_{prepare,apply,resume,activation}_s/{min,p50,max}` reports
  server-side tokenizer phase spans, excluding optional resource clearing. These
  retain their metric names but no longer include a separate HTTP round trip per
  phase. `coordinator_activation_s` remains the enclosing trainer-side await of
  all engine HTTP requests; no global preparation/apply/resume phases are inferred
  from overlapping engine lifetimes. Scheduler pause retains its rank-local
  timestamps and definition. `receiver_engines` counts coverage.
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
  --model /models/GLM5.2-NVFP4 --output /data/gpu-delta/new-run \
  --prepared-inputs /data/gpu-delta/prepared-inputs
```

`--output` must be new; omit it to create a timestamped directory.
`--prepared-inputs` creates a reusable cache when absent, or reads a completed
one; it defaults to `OUTPUT/inputs`. Repeat the command with the same cache and
a new output directory to skip inventory, calibration, perturbation and target
checkpoint construction. Codec, hash policy, frame size and sender GPU count
can change without regenerating these inputs.

The cache stores intermediate canonical versions as packed raw files; version 0
references the immutable base and the final version uses the shared oracle
checkpoint. With the default three versions, allow disk space for two canonical
copies plus one full checkpoint, in addition to the arms’ encoded publications.
The source checkpoint is immutable. No Megatron checkpoint, trainer initialization
or distributed export setup is required.

One invocation uses shared prepared inputs and an optional oracle, with two serial
Bitcomp-Zstd arms by default: payload hashing on and off. Each stage releases its
CUDA contexts before the next starts:

1. For a new cache, read the immutable checkpoint headers and config directly.
   CPU workers construct each cumulative target once and write disjoint cached
   ranges and the final oracle checkpoint; no receiver is needed for encoding.
2. For each arm, read the prepared versions into bounded pinned buffers and
   encode them with Miles’ production `GpuBatchEncoder` and `PublicationWriter`,
   using that arm’s hash policy. Construction is shared; pinned input loading
   remains per arm and outside compression timing.
3. After each arm’s compression, start fresh receivers from the original base,
   prepare/apply/resume its publications, and record all rank receipts and
   selected generation routes.
4. With `--validate-final-target`, load the shared final checkpoint once with
   the original static draft and compare every arm’s final text, tokens and
   prompt/output logprobs. Without this flag, target equivalence remains
   `NOT_VALIDATED` and the oracle phase is `NOT_RUN`.

`READY.json` is published only after all prepared files are complete. Reuse
requires the same base path/file sizes/modification times, canonical plan,
seed, mutation algorithm, target ratio and version count, plus matching cached
file sizes. This assumes immutable local artifacts; it does not rehash an entire
checkpoint or every raw snapshot. Incomplete or mismatched caches fail without
being overwritten. Fresh receivers independently check canonical metadata and local bindings for
every arm.

Sender compression uses all visible GPUs by default; `--sender-gpus N` limits
it to the first N visible devices. Model layers, embedding, LM head and remaining tensors form ordered whole
groups, split into balanced contiguous slices using the Miles ordinary-owner
rule. Putting embedding/head at the tail leaves that slice fewer model layers
to offset their higher-precision storage. Ownership and canonical bytes per GPU are recorded
in each arm’s `fixture/owner-plan.json`. Every worker reads only its assigned
prepared tensor ranges and writes its own publication shard. The parent seals
the shards after all workers finish. This uses the production codec and
publication path with simple HF ownership; it does not measure Megatron export,
TP reconstruction, expert ownership or distributed gather.

Pinned old/new input batches target 512 MiB; a larger tensor stays whole. Each
GPU retains only its owner's compact inner-codec payload through finalization.
Group-count balancing reduces per-GPU residency but does not guarantee equal
byte counts or make an oversized individual tensor fit. Wrapped codecs perform
one owner-wide GPU Zstd finalization per version. Scalar/vector target bytes
bypass compression.

The default performance proxy uses Bitcomp-Zstd with payload hashing on and off.
Inner and outer frames are 1 MiB. Explicit `--codecs` selections retain all five
codecs; unwrapped codecs have no outer stage. Every arm uses three
cumulative versions and two EP4 receivers. `--codecs`, `--hashing`,
`--frame-bytes`, `--versions` and `--ports` select the configuration. For a quick
single arm, use `--hashing on`. Hash-on arms generate and
verify payload SHA256; hash-off arms skip it on both sides. Host wall clocks and
scheduler pause timestamps remain available without per-phase CUDA events.
`--nsys` launches Nsight Systems and captures one warm receiver update, always
including CUDA, NVTX, OS-runtime and Python GIL tracing, CPU
sampling/context switches, and Python stack sampling. It enables all receiver
NVTX ranges through `SGLANG_ENABLE_NVTX_OPERATIONS=1`; there is no separate
GPU-delta NVTX mode or configurable decode-stage count.
Historical event measurements in saved fixtures remain identifiable as old
measurements; missing CUDA intervals are reported as unmeasured, not zero.

All arms share one calibrated mutation rate and the same prepared bytes.
Mutations are deterministic finite mantissa/packed-FP4 bit changes seeded by
name and version. `--ratio` defaults to 0.002 and calibrates mutation density
against CPU Zstd sample frames; the actual selected-codec ratio is measured,
not forced to that value. Static draft and calibration scales stay unchanged.
The versions are constructed cumulative targets, not learned gradients or three
statistical repeats of one delta.

Stdout and `REPORT.md` show a summary row per arm/version: codec, inner frame size (KiB), payload
SHA policy, compression/publication wall time (seconds), receiver preparation
(seconds), scheduler pause (milliseconds), and inner, outer and total payload
sizes (GiB). Unwrapped codecs have no outer stage. Full-precision timing components,
exact byte counts and every sender owner/receiver rank metric remain in
`summary.json`; `comparison.json` records
each arm’s final selected-output comparison, or `NOT_RUN` when disabled. `inputs.json` records the cache
identity, whether it was reused, and its original preparation measurements.
Arm artifacts live under `arms/`; the inventory and final checkpoint stay in
the input cache with their preparation logs. The fresh oracle, arm logs, launch
configuration, owner assignments, publications, original-rank receipts and
generation responses remain in the output directory.
A failure stops advancement and preserves partial output; it is never retried.

### Saved-delta apply mode for receiver iterations

`--reuse-fixtures OLD_RUN_ROOT` skips sender encoding. It uses the existing sealed
publications in place, starts fresh receivers from the original HF base, and runs
fresh prepare/apply/resume, generation and clear-state checks for every version.
Add `--validate-final-target` for the expensive independent final target oracle.
Default fresh encoding remains the mode for measuring sender performance.

```bash
python tests/manual/gpu_delta/bench_gpu_delta.py \
  --model /models/GLM5.2-NVFP4 --prepared-inputs /data/gpu-delta/prepared-inputs \
  --output /data/gpu-delta/new-saved-apply-run \
  --reuse-fixtures /data/gpu-delta/original-encoder-run \
  --reuse-producer-spec /data/gpu-delta/original-encoder-run/producer-spec.json \
  --reuse-producer-spec-sha256 ORIGINAL_REVIEWED_SPEC_SHA256 \
  --codecs bitcomp-zstd --hashing on --frame-bytes 1048576 \
  --versions 3 --sender-gpus 8
```

For a receiver profile, add `--nsys` to this saved-delta command and choose a new
output directory. It captures version 2 by default; `--profile-version N`
selects another warm update and does not enable profiling by itself.
Invoke the script directly: it owns the `nsys`
launch, so do not wrap it in another `nsys profile` command. `nsys` must be on
`PATH`, with CPU profiling permitted in the container. The output contains
`receiver-profile.nsys-rep` and the exact arguments/environment overrides in
`nsys-command.json`.

The cold first update runs without capture. The selected warm update emits one
`gpu_delta_benchmark_capture` range around preparation through both engine resume
replies, including a 0.2-second capture-arming lead-in. Capture stops when that
range ends; the remaining generation/update checks continue. Profiled timings
are diagnostic and must be kept separate from unprofiled performance results.

Fresh encoding runs write `producer-spec.json` with explicit encoder settings,
prepared-cache metadata hashes, the settings hash and producer source-file hashes.
Its reviewed SHA is required for reuse; today's receiver source does not establish
old producer provenance. Package versions remain unknown unless separately recorded.
For older runs, a reviewed provenance adapter can supply the same `argv`,
`cache_files`, `prepared_settings_sha256` and `guard_files` fields, while linking
the exact original producer evidence. Do not fabricate a historical launch record
or relabel an old failed oracle as a successful run. The selected fixture phase
must have completed successfully; a later oracle failure does not invalidate it.
It requires SHA-on payloads, the same input cache/settings/plan, codec/frame,
version chain, mutation settings and sender owner partition. Each selected arm must
have a completed original fixture phase and all owner/version records. Merely finding
`fixture.json` is insufficient because older encoders rewrite it after each round.

Admission streams every saved payload's SHA and verifies stable size/file identity,
sealed manifest SHA, canonical plan and original owner records before any receiver
starts. It rejects symlinks, path escapes, partial seals, missing owners, changed
settings and incomplete producer workflow. Missing prepared inputs fail without
starting preparation or encoding. The existing production receiver rechecks the
publication against its checkpoint inventory and local bindings when consuming it. Prepared checkpoint bytes
retain the existing metadata/settings/size boundary; this does not claim every
checkpoint byte was rehashed or that selected generation proves all weights equal.

No bulk files are copied, relocated or symlinked. The new output contains a small
`arms/ARM/fixture-reuse.json` reference, fresh receiver artifacts and an optional
fresh oracle. The original fixture/publication paths and bytes remain unchanged.
Workflow says `REUSED` / `SKIPPED_ENCODING`, with no fresh encoder wall time;
`fixture-verification.json` records current verification I/O separately.
All sender/read/encode/seal/compression measurements in reports are labeled
`historical_saved_fixture`; they preserve the original values and provenance.
They must not be counted as current sender performance or current end-to-end time.

External capture/reduction scripts must resolve this explicit reference and handle
`NOT_RUN` oracle status. Scripts assuming output-local fixtures or a mandatory
oracle are not compatible without an explicit update. Do not relabel a quick run
as a formal validated benchmark.

Report timing scopes separately:

- **Sender:** the compression-and-publication wall column includes payload
  hashing when enabled, raw/matrix writes and shard flush/fsync, alongside
  encoding and finalization. It sums sequential owner call spans, excluding
  shared input construction and pinned input reads. Publication wall is not pure
  SHA time. Saved fixtures may retain historical codec-only CUDA intervals,
  excluding hashing, transfers and XOR; fresh encoding leaves those unmeasured.
  Summary columns take independent owner maxima, not a synchronized global
  critical path. Parent manifest sealing is separate.
- **Receiver:** background preparation, CPU outer-decode and matrix-application
  host wall times, and actual scheduler pause. CPU decode wall time includes
  submission/raw copies/drain. Historical DE-stream CUDA intervals include
  zero-fill, enqueue gaps and decode; they are not pure hardware busy time.
  Fresh runs use Nsight for GPU profiling and leave these intervals unmeasured.
- **Coordinator:** complete prepare/apply/resume RPC lifetime. It overlaps rank
  work and cannot be added to preparation, decode or pause columns.

Receiver state is released after successful updates; only benchmark-owned engine
process groups are stopped. The devbox is retained. Generation is outside update
timing and checks selected outputs rather than every weight byte. This workload
has no concurrent serving traffic or real training, so it does not measure
serving interference or RL throughput. Production codec/recovery tests and the
five-layer learned-update E2E remain separate from this benchmark.
