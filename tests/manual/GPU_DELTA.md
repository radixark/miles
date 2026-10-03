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
| `WEIGHT_DELTA_CPU_WORKERS=4` | CPU outer-Zstd workers for the host cache creator; total per host, not per rank. |
| `WEIGHT_DELTA_HOST_CACHE_DIR` | Shared host tmpfs root; defaults to `/dev/shm/sglang-gpu-delta-<uid>`. Its persistent identity groups colocated engines. |

For Ray launches, set the job `runtime_env` environment or use the provided
`execute_train(extra_env_vars=...)` path. The submitting shell alone does not
forward arbitrary variables into existing workers. Trainer-only settings can
use `--train-env-vars`; receiver profiling needs the timing setting on rollout
actors too. The five-layer E2E forwards the codec and checks every learned
publication's protocol and codec. No old codec/encoder setting is migrated.

## Fixed producer and receiver pipeline

The sender retains old canonical weights in pinned CPU RAM and stages each new
export there asynchronously. Routed experts stay on their exporter EP/EDP owner;
non-routed tensors use the existing data-replica sender. Immutable owner geometry
partitions scalar/vector bypass and matrix batches once before learned updates.
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
once per host into shared Snappy storage during background preparation. Each rank
registers the shared arena for its streamed pinned transfer. After
actual pause, it streams each tensor to HBM, performs hardware Snappy decompression,
transforms the XOR mask into the physical weight layout, and applies in place.
GPU Zstd decompression is not part of this path.

Miles negotiates the immutable plan and original participant cohort once when
connecting; learned updates reuse that plan rather than sort and hash it again.
The receiver advertises an opaque shared-cache host identity. Miles supplies each
host's union of canonical tensor names so engines sharing that cache can reuse
host preparation without decoding experts assigned only to other hosts.
The owner-local exporter hook requires ETP1 only for this protocol; ordinary
upstream direct-exporter ETP support is unchanged.

One Miles coordinator exclusively owns the original engine cohort during an update;
concurrent engine administration, other weight mutations, or external pause/resume
calls are unsupported. Prepare runs while the old version serves, and Miles waits
for every original rank to be PREPARED. `update_weights_from_delta(session_id)` then
closes local admission, pauses scheduling, fences readers, retracts requests,
flushes caches and applies the delta. A failed reader fence never reclaims KV.
Each engine stays paused after APPLIED. Once all original ranks report APPLIED,
`resume_weights_from_delta(session_id, receipts)` validates their compact global
certificate, records the new weight version and resumes. There is no separate
global quiesce or commit round trip; ordinary pause/continue APIs are unchanged.

Only all-original-rank RESUMED receipts commit the sender's pending baseline. A
prepare failure can discard prepared inputs without stopping serving. Failure or
an uncertain reply after apply dispatch is terminal: retain partial artifacts,
do not abort/replay the XOR or automatically resume/recover. A failed apply RPC
may leave an unreachable engine serving the old version while reached engines
remain paused; the operation never publishes a successful new version. Session,
version and incarnation checks do not prove full weight-content equality.

The sender's N matrix bytes still incur new export D2H plus old/new H2D (3N total),
followed by final compressed D2H. Raw bypass avoids both matrix H2D uploads. Export,
bulk compression, publication and activation block training; compression starts
after all exports and does not overlap later exports. Receiver preparation runs
before pause; streamed H2D, Snappy decode and in-place mutation block rollout.
Do not sum nested phases or add sender/receiver times from different workloads.

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
  flush, application, global APPLIED wait and resume. Failed/open intervals are
  never reported as completed pauses.
- `receiver_reader_fence_s` and `receiver_paused_apply_host_wall_s` have separate
  rank distributions. `receiver_host_prepare_s` covers background preparation
  before pause. Other `receiver_host_*` summaries retain their source span names;
  registration and cache wait are per-rank work.
- `creator_host_*` distributions sample the single cache creator on each host,
  excluding attaching ranks' zero counters. `host_cache_creators` records coverage;
  byte sums count each creator once. `creator_cpu_workers` records active creator
  worker counts. `host_outer_zstd_validate_s` and `worker_decode_sum_s` are sums
  of worker intervals, while `host_outer_zstd_decode_s` is builder wall time
  including submission, validation, raw copies and joining tasks. These overlap.
- `coordinator_{prepare,apply_barrier,resume_barrier,activation}_s` are enclosing
  coordinator wall times. `producer_prefix_*` distributions end at the existing
  pre-publication owner gather, before receiver activation.
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

Each invocation starts one fresh TP8/DP8/EP8 GLM5.2 W4A16 engine with static MTP,
CuTe DSL MoE and no MoE A2A. It applies three cumulative immutable publications,
saving original-rank receipts, server logs and generation through DP routes0–7.

```bash
python tests/manual/bench_gpu_delta.py run --model /models/GLM5.2-NVFP4 \
  --fixture /data/gpu-delta/fixture --output /data/gpu-delta/snappy-zstd-new
python tests/manual/bench_gpu_delta.py oracle --model /models/GLM5.2-NVFP4 \
  --fixture /data/gpu-delta/fixture --output /data/gpu-delta/target-oracle
```

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
