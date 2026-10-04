# GPU-delta development benchmark

The experimental `--update-weight-transfer-mode gpu-delta` uses the paired
SGLang `update_weights_from_delta` API. `disk-delta` remains a separate checkpoint
handoff path. GPU-delta has one codec: `snappy-zstd`.

## Environment

Use a CUDA 13 Miles development image, the paired SGLang branch on `PYTHONPATH`,
and eight Blackwell GPUs. The receiver benchmark starts one TP8/DP8/EP8 engine
with GLM5.2 NVFP4 W4A16, CuTe DSL MoE, no MoE A2A and static bundled MTP.
The original checkpoint must already be available and is never modified.

Install the prebuilt encoder/decoder without changing the image dependency closure:

```bash
python -m pip install --no-deps nvidia-libnvcomp-cu13==5.3.0.16
```

The optional CPU correctness oracles also require `python-snappy` (0.7.3 in the
measured image); production compression does not use that package.

No custom C++/CUDA extension is built. Both sender compression stages use CUDA
SMs. Receiver Zstd decompression runs on CPU; Snappy explicitly requests the
Blackwell hardware decompression engine and rejects unsupported hardware or
allocation modes. Pinned host buffers supply streamed Snappy H2D copies.

| Environment variable | Meaning |
| --- | --- |
| `WEIGHT_DELTA_CODEC=snappy-zstd` | The sole supported value and default. Frozen at launch and matched against the receiver plan and immutable publication. |
| `WEIGHT_DELTA_TIMING=1` | Optional per-phase CUDA events. Default off; instrumentation can perturb timing. |
| `WEIGHT_DELTA_CPU_WORKERS=32` | CPU outer-Zstd workers per engine-host arena creator, not per rank, plus one independent SHA worker. Two colocated engines have separate pools (64 decoder workers at the default). |
| `WEIGHT_DELTA_HOST_CACHE_DIR` | Tmpfs base for engine-local host arenas; defaults to `/dev/shm/sglang-gpu-delta-<uid>`. Engines use separate subdirectories, identities and locks; only ranks of the same engine share an arena. |

For Ray launches, set the job `runtime_env` environment or use the provided
`execute_train(extra_env_vars=...)` path. The submitting shell alone does not
forward arbitrary variables into existing workers. Trainer-only settings can
use `--train-env-vars`; receiver profiling needs the timing setting on rollout
actors too. The explicit five-layer GPU-delta E2E arm forwards the codec and checks
every learned publication's protocol and codec. No old codec/encoder setting is migrated.

The existing five-layer test keeps its ordinary two-rollout `broadcast_packed`
default. Select the GPU-delta arm explicitly to exercise at least three learned
updates with a deterministic nonzero reward and the publication-change gate:

```bash
python tests/e2e/megatron/test_glm5_2_744b_a40b_5layer_nvfp4_w4a16.py \
  --gpu-delta --skip-prepare --num-rollout 4 \
  --update-weight-disk-dir /data/gpu-delta/e2e-new
```

`--skip-prepare` requires the existing checkpoints and dataset. GPU-delta disables
engine replacement and attention FP8 conversion only in its selected test arm.

## Fixed producer and receiver pipeline

The sender retains old canonical weights in pinned CPU RAM and stages each new
export there asynchronously. At expert TP=1, routed experts stay on their exporter
EP/EDP owner; expert TP>1 keeps the existing gather-before-convert sender path.
Non-routed tensors use the existing data-replica sender within each PP stage.
Immutable owner geometry partitions scalar/vector bypass and matrix batches once
before learned updates.
Raw target writes run on one CPU worker concurrently with matrix GPU compression;
there is no scalar/vector branch inside the matrix compression loop.

After export D2H completes, each bounded name-sorted matrix batch uploads old/new
bytes, computes XOR/change counts, and compresses all independent Snappy frames
in one call. Unchanged frames are omitted; incompressible changed frames remain
Snappy. Compact, 16-byte-aligned Snappy tensor arenas remain in HBM across batches.
The sender then compresses **all** owner Snappy arenas together with GPU Zstd,
using independent outer chunks of at most 1 MiB while preserving tensor boundaries.
Only the final encoded slab returns to pinned CPU RAM for file hashing/writing.
There is no intermediate Snappy host slab, CPU compression, or raw matrix fallback.

The existing `update_weight_buffer_size` bounds each canonical input batch; a
larger single tensor stands alone. The full compact owner Snappy payload must fit
HBM for the outer call. Host snapshots are assumed to fit RAM; no OOM fallback is
implemented. Production uses 1 MiB inner frames. The low-level encoder's 64 KiB
and 2 MiB controls are for isolated tests; the receiver accepts at most 1 MiB.

Protocol 4 records `codec: snappy-zstd`, explicit `frame_bytes`, natural tensor
identity and outer chunk offsets/lengths. SHA-256 authenticates final owner files;
old/new weights and intermediate Snappy bytes are not hashed. The receiver reads
and verifies immutable files, then CPU-decompresses locally needed outer chunks
once per engine-host arena into shared Snappy storage during background preparation. Each rank
registers the shared arena for its streamed pinned transfer. After
actual pause, it streams each tensor to HBM, performs hardware Snappy decompression,
transforms the XOR mask into the physical weight layout, and applies in place.
GPU Zstd decompression is not part of this path.

Miles negotiates the immutable plan and original participant cohort once when
connecting; learned updates reuse that plan rather than sort and hash it again.
The receiver advertises an opaque engine-host arena identity. Miles supplies only
that engine's local union of canonical tensor names. Ranks within an engine share
CPU preparation; independent engines have separate arenas and may duplicate host
bytes. No host-wide cache lock or release barrier couples separate engines.
Each creator bounds queued decode futures to `4 * WEIGHT_DELTA_CPU_WORKERS`;
it does not enqueue one unbounded future for every tensor/frame.
The global canonical inventory must have complete, unique owner coverage. Duplicate
exports, including overlapping PP/MTP names, are rejected rather than deduplicated.

One Miles coordinator exclusively owns the original engine endpoints during an
update; concurrent administration, other mutations and external pause/resume are
unsupported. Each engine independently prepares while its old version serves,
waits for its own original ranks to be PREPARED, then calls
`update_weights_from_delta(session_id)`. That engine closes admission, pauses,
fences readers, retracts requests, flushes caches and applies. A failed reader
fence never reclaims KV. Its own all-rank APPLIED certificate authorizes
`resume_weights_from_delta(session_id, receipts)`; the engine records the new
version and resumes without waiting for other engines. Ordinary pause/continue
APIs remain unchanged. Each engine's local TP/EP participants still synchronize
for safe activation; independent replicas may temporarily serve different versions.

The trainer awaits every engine coroutine and advances its pinned baseline only
after all original RESUMED receipts. On failure it still settles the other engine
tasks before raising. A preparation failure aborts only that engine's preparation;
an uncertain reply after apply dispatch is terminal for that engine: do not abort,
replay XOR, automatically resume or recover. Other engines may already have resumed
successfully; failure does not roll them back or report overall success. A failed
update never advances the common sender baseline. Session/version/incarnation
checks do not prove full weight-content equality.
External cancellation can leave outstanding remote work; it is incomplete and
does not authorize automatic retry, cleanup or recovery.

The sender's N matrix bytes still incur new export D2H plus old/new H2D (3N total),
followed by final compressed D2H. Raw bypass avoids both matrix H2D uploads. Export,
bulk compression, publication and activation block training; compression starts
after all exports and does not overlap later exports. Receiver preparation runs
before pause; streamed H2D, Snappy decode and in-place mutation block rollout.
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
  APPLIED and RESUMED receipts. It includes its reader fence, retraction/cache
  flush, application, its own engine's APPLIED wait and resume. Failed/open intervals are
  never reported as completed pauses.
- `receiver_reader_fence_s` and `receiver_paused_apply_host_wall_s` have separate
  rank distributions. `receiver_host_prepare_s` covers background preparation
  before pause. Other `receiver_host_*` summaries retain their source span names;
  registration and cache wait are per-rank work.
- `creator_host_*` distributions sample the single creator in each engine-host arena,
  excluding attaching ranks' zero counters. `host_cache_creators` records coverage;
  byte sums count each creator once. `creator_cpu_workers` records active creator
  worker counts. `host_outer_zstd_validate_s` and `worker_decode_sum_s` are sums
  of worker intervals, while `host_outer_zstd_decode_s` is builder wall time
  including submission, validation, raw copies and joining tasks. These overlap.
- `creator_host_{shared,encoded}_allocation_s/{min,p50,max}` samples only host
  creators; corresponding `allocation_calls/sum` and `allocation_bytes/sum`
  count cold/growth allocations once per engine-host arena. Warm fitting updates report zero
  allocation work while retaining their arenas.
- `host_shared_arena_bytes`, `host_shared_capacity_bytes`, and
  `host_encoded_capacity_bytes` report `{min,p50,max,sum}` across distinct engine-host arenas,
  counting each arena once across its ranks. Independent engines may duplicate
  physical bytes; `receiver_host_arenas` is not a physical-node count.
  Used bytes and retained capacity are separate quantities.
- `creator_host_payload_decode_hash_s` measures the combined CPU decode/hash
  wall span; `creator_host_payload_hash_wait_s` measures only the hash tail
  waited after decode. SHA duration and decode wall overlap and must not be
  added. The worker decode sum remains summed worker elapsed time, not CPU
  utilization. Each span has `{min,p50,max}` across engine-host creators.
- `receiver_host_plan_cache_reused/{min,p50,max}` reports per-rank static-plan
  cache reuse; each publication still validates its dynamic frame metadata.
  `creator_host_frames_validate_s` is creator-only and nested inside arena build;
  `creator_host_frames_validations/sum` counts one dynamic geometry validation
  per created arena, not zero-weighted follower rank medians.
- `receiver_host_shared_{register_calls,registered_bytes,registration_reused,
  mapping_reused,registration_capacity_bytes}/{min,p50,max}` are per-rank
  distributions. `registered_bytes` counts newly registered bytes for this
  update (zero on warm reuse); `registration_capacity_bytes` remains the active
  per-process capacity. These rank capacities are never summed as physical host
  memory. Receipts lacking optional capacity fields omit those metrics.
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
completes, every trainer drains its summary and the usual TP0/last-PP/effective
DP-CP0 logging rank submits it at the last trained rollout's existing step axis.
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
export WEIGHT_DELTA_CODEC=snappy-zstd
python tests/manual/bench_gpu_delta.py inventory \
  --model /models/GLM5.2-NVFP4 --output /data/gpu-delta/inventory
python tests/manual/bench_gpu_delta.py fixture \
  --model /models/GLM5.2-NVFP4 \
  --inventory /data/gpu-delta/inventory/inventory.json \
  --output /data/gpu-delta/fixture --ratio 0.002 --versions 3
```

The fixture calibrates sparse finite mantissa/packed-FP4 perturbations against
plain CPU Zstd level-1 sample frames. It measures the actual Snappy-Zstd publication
ratio separately; it does not force that ratio to 0.2%. Static draft and calibration
scales stay unchanged in the proxy; native tests cover scalar/scale updates.
The altered checkpoint contains the final cumulative version and does not alias
original checkpoint files.

The builder uses the production GPU encoder: pinned old/new CPU snapshots,
bounded GPU XOR/Snappy batches, then one owner-wide GPU Zstd pass per version.
Only compact Snappy survives between batches. Raw scalars/vectors bypass both
codecs. It records inner/outer bytes, alignment and final manifest/file sizes.
Fixture creation is setup, excluded from receiver timing and not a distributed
producer measurement. Native fixture tests independently replay CPU Zstd/Snappy
and verify exact altered targets and source/draft immutability.

Only protocol4 `codec=snappy-zstd` publications are admitted. Saved fixtures with
that schema can be reused without recomputing weights. Obsolete codec profiles
are not accepted by production; historical artifact migration is external setup,
not a fallback in this harness.

## Full-model receiver benchmark

By default, each invocation starts one fresh TP8/DP8/EP8 GLM5.2 W4A16 engine
with static MTP, CuTe DSL MoE and no MoE A2A. It applies three cumulative immutable
publications: one first-use update and two warm updates. It saves original-rank
receipts, server logs and generation through DP routes0–7.

```bash
python tests/manual/bench_gpu_delta.py run --model /models/GLM5.2-NVFP4 \
  --fixture /data/gpu-delta/fixture --output /data/gpu-delta/snappy-zstd-new
python tests/manual/bench_gpu_delta.py oracle --model /models/GLM5.2-NVFP4 \
  --fixture /data/gpu-delta/fixture --output /data/gpu-delta/target-oracle
```

For two TP4/DP4/EP4 engines on one eight-GPU host, provide two ports. The first
engine uses GPUs0–3; the second uses GPUs4–7. Both use the same tmpfs base
`WEIGHT_DELTA_HOST_CACHE_DIR`, but each engine owns a distinct subdirectory/arena.
The harness checks one arena per engine, all eight original scheduler identities, each engine's
TP/DP ranks0–3, and captures compute-process PID→GPU UUID observations. If NVML
uses host PIDs unavailable in the container's `NSpid` mapping, that join remains
explicitly unqualified; requested GPU masks are not presented as native proof. Generation
records retain engine IDs explicitly; use `(engine_id, dp_rank)` as the route key.

An EP8 fixture's view-bound plan digest does not describe EP4. First inventory
the new topology, then rebind its views into a **new** immutable fixture directory:

```bash
export WEIGHT_DELTA_HOST_CACHE_DIR=/dev/shm/gpu-delta-benchmark
python tests/manual/bench_gpu_delta.py inventory --model /models/GLM5.2-NVFP4 \
  --ports 31135 31235 --output /data/gpu-delta/ep4-inventory
python tests/manual/bench_gpu_delta.py rebind --model /models/GLM5.2-NVFP4 \
  --inventory /data/gpu-delta/ep4-inventory/inventory.json \
  --fixture /data/gpu-delta/fixture --output /data/gpu-delta/ep4-fixture
python tests/manual/bench_gpu_delta.py run --model /models/GLM5.2-NVFP4 \
  --fixture /data/gpu-delta/ep4-fixture --ports 31135 31235 \
  --output /data/gpu-delta/ep4-snappy-zstd-new
python tests/manual/bench_gpu_delta.py oracle --model /models/GLM5.2-NVFP4 \
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

Separate coordinator wall time, background read/hash/pin/CPU-Zstd preparation,
explicit scheduler pause, H2D, hardware Snappy decode, layout/application and
derived refresh. Pause measures the original scheduler flag-to-resume interval;
it excludes earlier prepare/status handler service and does not quantify serving
interference. Do not sum nested events or concurrent rank durations.

Each host reads/hashes the immutable owner files once and CPU-unwraps the union
of tensors needed by its original ranks into a shared arena. Each scheduler
registers that arena for pinned transfer and reuses tensor-level HBM scratch;
the shared storage stays alive while its consumers use it. The benchmark generates before/after updates, not during preparation;
realized overlap, request latency and production throughput need separate study.
The harness terminates only its own engine processes, retains partial evidence
on failure and never releases a devbox allocation.
