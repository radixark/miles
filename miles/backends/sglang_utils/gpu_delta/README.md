# GPU delta receiver

The experimental receiver applies canonical XOR deltas directly to existing
execution-layout buffers, including scalars and vectors. Derived calibration
buffers are recomputed from the updated scalar values.
It does not requantize weights, change checkpoint loading, replace the ordinary
allocator, or alter the independent disk update API.

See [GPU-accelerated Weight Teleportation](https://humansand.ai/blog/gpu-accelerated-weight-teleportation)
for the end-to-end design and performance motivation.

This is the receiver contract. The [user guide](../../../../docs/advanced/gpu-delta-weight-transfer.md)
covers the trainer side, metrics and package ownership, and the
[development guide](../../../../tests/manual/gpu_delta/README.md) the manual checks, E2E
test and benchmark. The companion tree is the paired SGLang branch that registers these
hooks; where stated, admission also identifies SGLang main's storage.

## Admission and supported layouts

Canonical names, shapes and dtypes come from the immutable local safetensors
checkpoint headers, read whenever `get_gpu_delta_info` builds a backend (also after a
clear). The standard loader's selected target and bundled-MTP files are supported;
custom, secondary, encrypted or transformed sources are not. The receiver binds its plan digest to the SHA-256 of
its own primary safetensors shards. A cached content digest beside the checkpoint
(or in the user cache for a read-only mount) is reused while the files retain the
same identities, sizes and timestamps. The startup checkpoint must remain immutable
while the engine uses it.

The checkpoint files must stay immutable between SGLang model loading and the
receiver's first admission, when it binds their digest. Clearing delta resources
retains this identity with the live weights; if the files change, reattachment
requires an engine reload.

The trainer and receiver share this checkpoint prior, not a negotiated receiver
plan. The trainer publishes canonical names, dtypes, shapes and byte encodings.
Each receiver independently maps that inventory to its local TP/EP slices and
backend storage. Publications contain no rank views or backend permutations;
encoding does not require contacting rollout. The receiver binds the static
catalog once, admits only tensors it binds or that another rank's routed experts own,
checks that all its mutable tensors are present, and reuses the binding table on later
updates.

Quantization follows a routed-expert-only contract. Routed experts may be quantized
in a format their kernel backend admits (`kernel_backends/`, below) and their model
mapping routes; every other mutable tensor is high precision (BF16 or F32) and served
unquantized. A shared expert that SGLang fuses into the routed `FusedMoE` counts as one
of its routed experts. Generic dense binding admits only `UnquantizedLinearMethod`,
`UnquantizedEmbeddingMethod` or a plain module without a quantization method; every
quantized dense method is rejected at admission, and F8_E8M0 is not a canonical dtype.
GPU delta never binds static drafts or frozen modules (below), so it places no
constraint on their format; SGLang must still load them. Checkpoints that quantize
dense layers, such as DeepSeek-V4.1's released block-FP8 projections, are out of
scope; RL runs convert them first. The primary recipe is NVFP4 W4A16 routed experts
on FlashInfer's CuTe DSL or MegaMoE runner, with every other tensor high precision; the
receiver tests cover it in depth, and the E2E test serves it on CuTe DSL. The other
admitted formats and runners are implemented for completeness and future extension,
with lighter tests.

`models/` holds one module per SGLang model implementation; its `__init__.py` states
how a model is admitted. A listed architecture is a binding contract, not serving
qualification (see the E2E test in the development guide).

| Module | Served architectures | Mapping-specific contract |
| --- | --- | --- |
| `deepseek_v2.py` | `DeepseekV2ForCausalLM`, `DeepseekV3ForCausalLM`, `DeepseekV32ForCausalLM`, `GlmMoeDsaForCausalLM` | The DeepSeek-V2 runtime that DeepSeek and GLM DSA checkpoints share: canonical name mutation, unequal Q/KV-A fusion, indexer fusion and MLA consumer views; its MLA helpers also serve mappings of other `DeepseekV2AttentionMLA` subclasses |
| `deepseek_v4.py` | `DeepseekV4ForCausalLM`, model type `deepseek_v41` only | DeepSeek-V4.1's BF16 dense projections, with fused rows bound as aliases, and ModelOpt NVFP4 routed experts |
| `kimi_k3.py` | `KimiK3ForConditionalGeneration`, model type `kimi_k3` only | Kimi-K3's KDA and MLA layers, F32 `o_norm` and attention-residual scores as computed images, and compressed-tensors MXFP4 routed experts |

The DeepSeek-V4.1 and Kimi-K3 mappings mirror SGLang main's loaders on a best-effort
basis, as placeholders without end-to-end tests yet; end-to-end testing will start from
them. GDN and other model mappings are not implemented.

The dtype recorded for each tensor in the selected HF checkpoint is canonical.
Miles may train that tensor in higher precision and casts the exported value back
to the checkpoint dtype before computing XOR. For example, DeepSeek-V4.1-Flash's
released compressor norm, indexer key norm and Engram Q/K gains are BF16. Miles
can train them in FP32 and publish BF16 deltas to BF16 serving parameters.
Admission rejects a serving parameter that narrows one of these gains below its
checkpoint precision.

Static tensors never carry trainer updates; the user guide's
[limits](../../../../docs/advanced/gpu-delta-weight-transfer.md#limits) define them. A bundled
draft's shared embedding and head follow the target; a speculative token map, which gives
the draft a gathered copy of the head, is rejected at admission.

Expert byte-view segments (`byte_views.DestinationSegment`) exclude padding and preserve
the canonical byte order; there is no per-update weight or mask permutation. Aliased
images receive one XOR; independent consumer images each receive it.

`kernel_backends/` holds one module per kernel library behind SGLang's MoE runners; its
`__init__.py` states how a runner is admitted, and each library summarizes its formats
in a subsection below. These are byte-layout contracts for existing inference paths, not
new inference kernels. Load-time weight quantization and conversion fallbacks are
rejected.

Attention, routed-expert and shared-expert TP follow their existing loader
partition groups, including padded expert slices. Each rank decodes complete
canonical tensors and applies only its admitted byte views. Attention DCP is
supported within an attention-TP group: its width must divide attention TP.
DeepSeek/GLM MLA's unquantized replicated Q/K consumers receive the same XOR mask
through prepared affine views; local K/V caches refresh from the local KV-B shard.
This adds no apply-time all-gather or kernel specialization. Quantized replicated
Q projections remain unsupported. On SGLang main the local Q projection is a view
of its gathered image, so those rows take the XOR once, through that image, and
the gathered, transposed V image (`w_vc_qrep`) is bound as an affine view.
Admission detects either tree's storage and rejects any other alias.

Prefill CP reuses those module-owned partitions: CP partners may repeat attention
views while routed and shared experts retain their own TP groups. The optional
`enable_cp_decode_attn_tp` optimization is rejected because it caches independent
weight slices and temporarily swaps live parameter storage.

Fixed topology is required when routed experts are bound: no presharded canonical
checkpoint, EPLB, redundant experts or elastic joiners. The launch MoE runner and
dispatcher must be a pair `_MOE_RUNNER_BACKENDS` admits. Dense-only mappings do not
need a MoE configuration. Ordinary globally ordered control broadcast is required;
local broadcast, LoRA, disaggregated serving or pipeline widths above one, a multi-node
engine without attention DP (whose other nodes cannot reach the head node's IPC reply
socket), shared IPC weight storage, HPC-Ops derived-weight caches, DWDP expert prefetch,
DSPARK's quantized lm_head cache (SGLang main's `--speculative-dspark-lm-head-dtype`)
and a speculative token map (`--speculative-token-map`, whose hot-token rows the EAGLE
worker copies out of the target lm_head for the draft) are rejected at admission.
Control replies include every original TP/CP/DP participant.

### FlashInfer (`kernel_backends/flashinfer.py`)

FlashInfer's TRTLLM, routed TRTLLM, MXFP4, CuTe DSL and MegaMoE runners share one
module because they share its row permutations, blocked scale tiles and consumer
fields. The routed and MXFP4 runners consume TRTLLM storage (each runner's storage
family is its `_MOE_RUNNER_BACKENDS` row), so the TRTLLM column covers them. The
admitted matrix, owned by `_PAIRS` and `identify_layout`, is:

| Native weight format | TRTLLM / routed | CuTe DSL | MegaMoE |
| --- | --- | --- | --- |
| BF16 | Yes | No native path admitted | No native path admitted |
| FP8, 128×128 blocks, UE8M0 scales | Yes | No | No |
| MXFP8 | Yes | No | Yes |
| MXFP4 with MXFP8 activation | Yes | No | Yes |
| NVFP4 | W4A4 | W4A4 or W4A16 | W4A4 or W4A16 |

Non-gated experts bind their up projection to the single W13 slot of the BF16 TRTLLM
layout: the 32-row shuffle and 128-byte K blocks apply, without a gate/up interleave,
and a gate projection is rejected.

CuTe DSL serves gated W13 weights and block scales as interleaved up/gate row groups:
64 rows on the companion tree (`interleave_w13_halves`) and on SGLang main without
swap-AB. On SM100 and SM103 SGLang main selects FlashInfer's swap-AB layout
(`use_cutedsl_moe_swap_ab`, FlashInfer 0.7.4 or later), whose
`interleave_cute_dsl_moe_weights` writes 16-row groups. Admission reads the decision the
warmed `CuteDslMoEWrapper` records (`swap_ab`; earlier wrappers have only the 64-row
layout) and rejects that layout, which no CUDA check has compared yet.

Block128 FP8 accepts exact UE8M0 scale values in compact U8 wire storage and expands
them losslessly into the consumer's FP32 API image; arbitrary FP32 scale factors
are unsupported. NVFP4 derived calibration reuses the ordinary inference helpers.
MegaMoE rejects changes to its captured input-normalization scalar and incompatible
gate/up alphas before any live writes; under an activation clamp (`gemm1_clamp_limit`),
TRTLLM NVFP4 likewise rejects untied gate/up global scales. CuTe wrappers, MegaMoE
kernels and derived buffers must already be initialized; SGLang main builds each
layer's MegaMoE kernel on its first forward (CUDA graph capture or warmup).

TRTLLM MXFP4 covers two SGLang methods. `Mxfp4FlashinferTrtllmMoEMethod` serves the
FP4 routed experts of `Fp8Config` checkpoints. Its loader, like MegaMoE's FP4 experts,
casts a checkpoint `weight_scale_inv` numerically into FP32 parameters, which does not
keep U8 codes, so an MXFP4 expert scale named `weight_scale_inv` is rejected; the
layout tests bind these kernels' scale images from U8 `weight_scale` bytes. The
companion SGLang tree keeps the loader's FP32 scales and writes the kernel image to
`w*_weight_scale_inv_shuffled`.
SGLang main writes its image over `w*_weight_scale_inv`, typed `float8_e4m3fn`, pads the
intermediate to 128 outside every destination and keeps the GEMM1 clamp on the
method; its MegaMoE MXFP4 likewise binds the DeepGEMM-packed scales to
`w*_weight_scale_inv`. Admission identifies the tree from the live scale storage
and rejects any other. Both trees interleave the gate/up rows only under
`SGLANG_MXFP4_USE_OFFICIAL_SHUFFLE` (default true); admission rejects the legacy
shuffle's plain halves. SM100 `Mxfp4MoEMethod` (`_fi_kernel == "trtllm_sm100"`)
serves MXFP4 experts from separate gate/up halves, with `float8_e4m3fn` scale views;
its Marlin, DeepGEMM and MegaMoE transforms are rejected.

## Ownership and lifecycle

Miles owns model updates, pause/resume, memory residency and topology. Each receiver
commits a pair of stream ID and publication version. Every rank must match the
publication's source before apply. Forward uses its base as the source and commits
its target. Explicit reverse uses the target as the source and commits its base,
as when eval cancels a snapshot. The immutable HF base is `(null, 0)`. A trainer
restored at `(X, S)` can publish from that exact parent into a new stream.

The three HTTP POST endpoints are an internal interface to Miles:

| Endpoint | Purpose |
| --- | --- |
| `get_gpu_delta_info` | Return original rank identities, committed stream/version, state and plan digest. An optional session ID retrieves its last completed activation reply. |
| `update_weights_from_gpu_delta` | Prepare, apply and resume one publication, returning compact APPLIED acknowledgements, full RESUMED receipts and server timings. |
| `clear_gpu_delta_state` | Release retained delta resources while preserving weights and committed version. |

Each participant identity carries the engine ID, a random rank ID, hostname, host
cache ID, PID, process start ticks and TP/DP/PP ranks.
The update request carries the admitted participants, caller-generated session
ID, manifest path/SHA256, base and target stream/version pairs, and plan digest.
`validate_payload_hash` defaults to true and `reverse` to false. Rolling updates and
HF-base recovery use the same request shape;
successful updates reuse the original admission. Each engine's
tokenizer owns the following internal phases:

1. **Prepare:** the CPU process authenticates the manifest and builds shared
   payloads and numeric metadata. Rank-local background work qualifies the canonical
   plan and prepares small GPU metadata, uncompressed XOR masks and derived calibration.
   It takes no model-update writer
   lock and neither writes model/KV storage nor pauses generation. The tokenizer
   aborts a preparation that has not settled within 30 minutes (`_PREPARE_TIMEOUT_S`).
2. **Apply:** close generation admission, pause scheduling, fence readers, pause
   generation in the requested mode (retracting requests by default) and optionally
   flush caches. Allocate two decoded-mask slots, bind
   pointers, decode and apply. `APPLIED` requires GPU completion and successful
   native size/status checks. A failed fence cannot reclaim KV/cache.
3. **Resume:** after every original rank of that engine applies, the tokenizer
   sends the internal resume message. SGLang records the model version and reopens
   generation. Miles advances its baseline only after all engines finish.
   Independent engines need not pause together.

Registered weights may reside on CPU or CUDA; the serving worker selects the
CUDA execution device independently of parameter placement. Preparation leaves
both kinds of live storage unchanged. After the reader fence, each touched CPU
storage is copied once to a temporary CUDA mirror. The existing affine XOR
kernels update those mirrors, preserving views and padding; successful native
status and GPU completion precede copyback into the original CPU storage.
Cross-device derived consumers finish updating before `APPLIED`. Cancellation
uses the same XOR operation for either residency.

This requires spare HBM for all touched CPU storage capacities during apply;
it is not a bounded-memory CPU-offload implementation. Prefetched group
offloading is rejected because it retains independent GPU copies. Dynamic
offloader integration is not covered by the mixed-residency tensor tests.

Dedicated eval uses the same API, phases and host arena; the receiver does not
interpret eval or rollout modes. Miles sequences it as the
[eval contract](../../../../docs/advanced/gpu-delta-weight-transfer.md#dedicated-eval-snapshots)
describes.

The tokenizer serializes update, info and clear controls independently of generation.
After a lost HTTP reply, Miles can read the original activation receipts when the
session and all committed ranks match. A stale source is rejected before preparation
and leaves the last committed state intact. Before apply dispatch, the tokenizer
can abort the matching preparation, including one cancelled while waiting for
pause admission. It waits for every original rank to drain its work and close the
backend, releases the shared host cache, then reattaches at the verified checkpoint
and source version. The live weights and committed canonical shadows survive this
reset. A transient storage error or prepare deadline permits one new preparation
attempt. Caller cancellation and invalid publications propagate to the caller.
An unverified reset, a failed drain or an uncertain write requires receiver
replacement. Apply/resume failures keep admission closed.

Updates default to `pause_mode="retract"` and `abort_all_requests=false`.
`pause_mode` selects SGLang's own scheduler pause, run after the reader fence.
`retract` drains the overlap result, requeues running requests, frees their KV and
flushes the prefix and multimodal caches. `in_place` keeps running batches, their KV
(computed with the old weights), the overlap result and the prefix cache; the normal
loop resumes them on the new weights. Cache handling follows the pause mode; there is
no separate flush option. In-place updates retain multimodal caches too. Miles uses
this mode only for rollout syncs under `--pause-generation-mode in_place`, as its
[user guide](../../../../docs/advanced/gpu-delta-weight-transfer.md#activation-and-failures)
states. Receiver-owned derived images and caches are refreshed during apply in either mode.
Reader fencing and the generation pause always run; optional abort occurs after new
admission closes. Updates preserve the pause present at the apply boundary. An
internally retained pause commits the version
and releases per-update resources without reopening generation; its
`RESUMED` receipt has `generation_paused=true` and null `resumed_ns`/`blocked_s`.
The optional `weight_version` labels admitted generations independently of the
transport's target sequence; omission uses the transport target version.

`clear_gpu_delta_state` is valid when idle, successfully resumed, or draining an aborted pre-apply session. It drains each
rank's FIFO release work, closes workers/host allocations and drops
planning, decoder and stream objects. After every rank acknowledges, each
engine-host's arena owner (the rank that ran its CPU child) deletes the shared encoded
and prepared caches. Weights, canonical shadows, publications, committed version,
process identity, small provenance and pause state remain unchanged. Describe
recreates the backend at that version; until it does, prepare is rejected and the
receiver stays cleared. Framework allocator/JIT caches are not cleared.

HF-base recovery requires a fresh version-0 engine serving the matching immutable
checkpoint; the publication path must be visible to all ranks. Delta resources are
retained by default (`release_state=false`). `release_state=true` clears them after a
successful activation, with the same all-rank close/cache-release barriers and outside
the returned phase timings; the response keeps the APPLIED/RESUMED receipts when that
clear succeeds and reports only the clear failure otherwise.

## Payload and cache contracts

The receiver takes the codec and frame size (a positive integer of at most 4 MiB,
`publication.MAX_FRAME_BYTES`) from each publication's manifest; it has no codec
environment variable or default (the user guide lists the codecs). Frames for the
decompression engine (DE) must fit the device's compressed and decoded limits, and all
paths validate geometry, alignment and capacity. Streams can change codec
without changing their canonical plan; native decoders are cached by inner codec.

Wrapped matrix frames use independent GPU-compressed outer Zstd chunks with at
most 1 MiB output. CPU workers unwrap them into shared host storage. Plain
LZ4/Bitcomp has an empty outer `frames` list and equal encoded/decoded extents;
workers copy the inner compressed bytes. DE reads its host arena directly;
CUDA decoders receive batch-sized H2D copies through two GPU slots.
There is no full-publication HBM copy.
Unchanged frames are omitted. Scalars/vectors and compact UE8M0 scales use
`xor_raw_bytes`: uncompressed canonical XOR masks, omitted when unchanged.
Preparation computes complete scalar targets for derived calibration without
changing live weights. Dtype admission runs on each rank when
`get_gpu_delta_info` constructs its backend: binding checks every bound tensor of
the full local checkpoint inventory, including tensors whose delta stays empty,
against the served dtypes its direct or packed layout allows. A rank reports all
of its mismatches in one error prefixed `tp_rank=<t> dp_rank=<d>: `, and the
failed query returns every rejecting rank's message before any publication is
read; the CPU child performs no dtype admission. Matrices require matching
logical dtypes; no matrix values are scanned for cast compatibility. BF16
scalars/vectors may use FP32 storage: the dtype policy is independent of model or
parameter name. It relies on
served values remaining lossless checkpoint images and masks using checkpoint
dtypes; arbitrary edits to served FP32 buffers are outside this contract.
Preparation performs no live-value validation or host synchronization for dtype
admission. BF16 widening and the explicit UE8M0-to-FP32 scale representation
produce physical XOR masks before pause; apply kernels still consume only bytes
and runtime layout metadata.

A model mapping may bind a canonical tensor that the loader serves only through
arithmetic (for example an F32 weight rounded into a BF16 parameter) to a
canonical shadow: receiver-owned exact canonical rows on the consumer's device.
Only a version-0 admission creates a shadow, and its first
preparation fills it from the immutable checkpoint. The session retains shadows
across `clear_gpu_delta_state`, so they always hold the committed version; updates
XOR them like any other raw or matrix target. A computed image rebuilds each
served value from shadows and live canonical storage with the loader's own
function. It runs during the pause after the matrix batches, under the sticky
error gate, and every admission recomputes each computed image and requires
bytewise equality with its live consumer. Re-admission after version 0 without its
shadows requires an HF-base restart. Every shadow must feed a computed image, which
admission rejects otherwise, so computed images and their shadows are
CUDA-resident; shadows are freed only with the process. Other downcasts and
numerical casts are not supported.

One engine-host creator reads and optionally SHA256-checks owner files in parallel
into retained tmpfs slices. Source identity/extent is checked across each read;
all tasks drain before READY or failure. Namespace/publication records bind the
original cohort, stream, path, digest, session, versions and hash policy. Mapped
encoded bytes remain owned until every submitted worker drains.

For every codec, one fixed original rank per engine-host (the lowest TP rank, DP
rank, PID and rank ID) owns a persistent CPU child, without Torch or CUDA, that parses
the manifest and prepares the complete canonical image; only descriptors cross its IPC
channel. It publishes READY after building the image, and a release marks only that
READY publication REUSABLE. The child never rebuilds a token recorded in its cache
state, whose XOR mask may already be applied. Cancellation terminates and reaps the
child, ending its write lease; a late abort may retain a complete READY image but never
makes it reusable. Clear drains and closes the child, and parent death terminates it.
Every rank maps that same prepared tmpfs image; followers create no payload pool, and
abort cancels their owner wait.

DE ranks own private CUDA 13 `HOST_NUMA`/`PINNED` allocations with
`CU_MEM_CREATE_USAGE_HW_DECOMPRESS`; these are never exported/imported. Each DE rank
maps the shared prepared image without registration and copies only its required
tensor spans into its private arena, without another file read, hash or
outer decode. All arenas and caches fit the initial extent, rounded to allocation
granularity. Growth reserves twice the required extent; fitting
updates reuse storage and CUDA registrations. Shared prepared capacity is counted
once per engine-host, while DE rank-private capacity is counted per rank.

Successful all-rank apply precedes resume. The owner queues the release of that
publication token after its resume, and ordered release/view cleanup on the same
FIFO executor completes before the next preparation can reuse the arena.
BUILDING precedes overwrite; failed/aborted publications remain nonreusable, and
late old releases cannot release newer bytes. Explicit clear removes retained
encoded and prepared files only after the rank-close barrier. Each rank share-locks its
cohort directory until close; the owner's first preparation removes cohort directories
that no live rank locks (replaced or crashed incarnations, earlier jobs on the same
engine ID, and an older build's cohorts, which hold no lock).

## Fixed decode/apply pipeline

Compressed tensors are grouped by active model layer, with separate embedding,
head and remaining standalone groups first. Empty groups are omitted. Each batch
must fit HBM; there is no intra-layer streaming. The first publication prepares
affine geometry for every local matrix and physical small-mask destination,
including tensors without frames.
Later publications reuse exact active batches and repack the offsets and pointer
tables of batches whose membership changes. Frame lengths, arena offsets and
omitted ranges come from the current publication.

Preparation builds one numeric frame table, validates workspace/output geometry,
uploads small metadata/XOR masks, prepares derived values and prebinds launches of the
fixed Triton apply kernels (`miles/utils/gpu_delta/apply.py`); no model geometry enters
their compilation-cache keys, and there is no online autotuning or pause-time scratch
fitting.

After the reader fence, two largest-batch decoded HBM slots are allocated when
compressed work exists, and small XOR updates run first on the apply stream, which owns
the sticky error gate. DE decodes one slot while the apply stream consumes the other; on
CUDA, one compute stream decodes and applies while a copy stream prefetches the next
batch's compressed input into one of two slots. `receiver.py` documents the ordering.

Only omitted frame gaps/tails are zeroed. A fully covered batch skips zeroing; one
status kernel checks native sizes/statuses before each grouped XOR. Source binding
requires 16-byte aligned decoded slots, patches the prepared pointer slab and uploads
it once; a misaligned slot is rejected because there is no fallback descriptor path.
Derived consumer copies are gated by the sticky error. Computed images run in the
same phase, after the prepared calibration copies. No image reads another derived
destination; a computed image reads only canonical-bound storage and never overlaps
a canonical binding.
Completion joins transfer, decode and apply before scratch release; failed cleanup
attempts to drain both streams. Foreign tensor aliases retain allocations and registered
producer/consumer streams are fenced before free in their original CUDA context.
The ordinary Torch allocator and its settings remain unchanged.

Prepared XOR kernels update existing buffers during pause. The receiver qualifies the canonical
plan digest once and validates fresh frames and native decode status every update.
Before apply, identity checks cover every live parameter
slot and consumer object, with tensor geometry deduplicated at admission. Pointer
identity cannot detect unrelated same-buffer writes by another updater.

## Configuration and observations

Feature environment variables are development controls, not a stable public API.
Keep them fixed for the stream's lifetime. Batch size is captured at backend
creation, worker count when its pool is first needed, and sorting by each decoder.

| Setting | Default | Meaning |
| --- | --- | --- |
| `GPU_DELTA_LAYERS_PER_BATCH` | `1` | Positive count of active adjacent model layers per batch |
| `GPU_DELTA_CPU_WORKERS` | `32` | Reusable workers on one owner per engine-host, for every codec; range 1–64 |
| `GPU_DELTA_SORT_BEFORE_HW_DECOMPRESS` | `0` | `0`/`1`, native sorting on the paused DE call |
| `GPU_DELTA_HOST_CACHE_DIR` | `/dev/shm/sglang-gpu-delta-<uid>` | Private user-owned tmpfs visible to the engine's host-local cohort |
| `SGLANG_ENABLE_NVTX_OPERATIONS` | `0` | Standard SGLang detailed operation annotations, including GPU delta |

Profiling the CPU child requires the optional NVIDIA `nvtx` Python package:
`python -m pip install --no-deps nvtx==0.2.17`. It is imported only when
`SGLANG_ENABLE_NVTX_OPERATIONS=1`; ordinary execution does not require it.

Payload verification is selected per update by `validate_payload_hash=True` on
`update_weights_from_gpu_delta`; Miles sets it per publication, as its
[user guide](../../../../docs/advanced/gpu-delta-weight-transfer.md#activation-and-failures)
states, and SGLang does not read a hash-policy environment variable. Every rank sharing
a publication must agree. After ordered cleanup releases the completed publication, the
next publication may choose another policy; unchecked bytes cannot satisfy a checked
publication. A receiver requiring hashes rejects `payload_checksum_format="none"`; a
skipping receiver may accept hashed publications. Skipping trusts contents without
payload SHA authentication.
Manifest SHA, exact file extent/identity, path containment, Zstd output length and
native decode checks still apply.

CPU timings, bytes and counters remain available for W&B independently of NVTX.
The paused APPLIED acknowledgement retains identity, publication provenance, state,
scheduler timing and `result.applied`/`result.target_version`. Full metrics stay in
the session and are returned by RESUMED and status, avoiding their serialization
before the all-rank apply barrier. Receipts contain no diagnostic CUDA timing
events. Host spans are nested and worker sums overlap; do not add them or infer
device bandwidth from host durations.

| Receipt fields | Interpretation |
| --- | --- |
| `host_prepare_s` | Full background preparation |
| `host_manifest_read_parse_s`, `host_plan_validate_s`, `host_tensor_prepare_s` | Manifest, canonical-plan qualification, local tensor planning |
| `host_plan_cache_reused`, `host_batch_plan_reused` | Admitted digest / current active-geometry cache reused; no old frames retained |
| `host_rank_prepare_s` | Publication access and child work or follower wait, rank-local canonical admission, mapping and local input preparation |
| `host_rank_metadata_release_s` | CPU child drops the parsed manifest and encoded view after shared preparation |
| `host_cpu_worker_spawn_s`, `host_cpu_worker_wait_s` | Owner's process-launch call / complete preparation IPC wait; wait includes child work and is not additive with its phases |
| `host_cpu_worker_pid`, `host_cpu_worker_start_ticks` | Shared child identity, repeated on the engine's ranks; followers do not own another child |
| `host_rank_prepare_call_s` | DE-local copy loop or CUDA tensor-view binding after layout/allocation; nested inside the rank preparation span |
| `host_encoded_cache_access_s` | Publication access through shared numeric-metadata attachment; includes child work or follower wait |
| `host_rank_layout_s`, `host_rank_allocation_s` | Local ordering/offset planning / arena growth |
| `host_raw_pack_s`, `raw_tensors`, `raw_bytes`, `raw_h2d_bytes` | CPU raw packing, raw tensor count, direct bytes, aligned arena upload bytes |
| `host_metadata_prepare_s`, `host_metadata_wait_s` | Small GPU input/status setup including its own stream waits / those waits alone |
| `host_apply_kernel_prepare_s`, `apply_kernel_cache_misses`, `apply_kernel_cache_reused` | Record construction and fixed-module cache lookup; misses are Python cache misses, not necessarily compilation |
| `paused_setup_host_s`, `paused_apply_host_wall_s` | Slot allocation/binding/uploads / complete apply through completion and scratch release |
| `h2d_bytes`, `apply_metadata_h2d_bytes`, `apply_descriptor_h2d_bytes`, `decoder_metadata_h2d_bytes` | Explicit raw, metadata, CUDA compressed-input and CPU-resident mirror uploads; descriptor bytes are a subset of apply metadata bytes |
| `cpu_weight_h2d_bytes`, `cpu_weight_d2h_bytes` | Touched CPU-resident storage copied to temporary CUDA mirrors after the reader fence / copied back after successful apply; present only with CPU-resident targets |
| `compressed_input_buffers`, `compressed_input_capacity_bytes`, `compressed_input_h2d_bytes` | CUDA input-slot count, total input-slot capacity and copied bytes including alignment padding; zero on DE |
| `de_host_input_bytes` | Compressed bytes read directly by DE, not an explicit H2D transfer |
| `decoded_buffers`, `decoded_scratch_bytes`, `decoder_workspace_bytes` | Actual slots (0 for raw-only); total slot bytes; temporary workspace, not peak HBM |
| `decoded_zero_ranges`, `decoded_zero_bytes` | Omitted canonical ranges cleared before decode |
| `host_matrix_enqueue_s`, `host_apply_completion_wait_s` | Host enqueue of the paused matrix batches / host wait for paused GPU completion |
| `compressed_batches`, `compressed_tensors`, `layers_per_batch`, `decoder_metadata_uploads` | Actual batches and tensors, configured layers per batch; decoders upload descriptors once in preparation and output pointers once in the pause, and CUDA also uploads input pointers |
| Group/contract/grid counters, `computed_images` | Actual runtime work, including computed images refreshed after the matrix batches; kernel cache reuse is a per-rank boolean |

CPU slab construction and launch prebinding are outside
`host_apply_kernel_prepare_s`. Shared-cache fields have distinct scopes:

- `host_encoded_cache_created`/`reused` identify creator and followers;
  `host_encoded_cache_wait_s` is the mutex wait and `access_s` includes
  wait/build/attachment through lock release. `build_s` repeats the cached creator
  duration on followers and must not be summed across ranks.
- Creator-only `host_encoded_cache_read_hash_s` encloses submission and joins;
  `read_worker_sum_s` and `sha256_worker_sum_s` sum overlapping tasks. `hash_files`
  and `hash_bytes` count work once. `skip_payload_hash` is reported on every rank;
  skipped SHA time/counts are zero.
- `host_rank_outer_zstd_decode_s` encloses task submission/join, including raw
  copies; `host_rank_outer_zstd_worker_decode_sum_s` sums each worker's native
  batch decode and scatter into the shared image.
  `host_rank_outer_zstd_{encoded_bytes,decoded_bytes,tensors,frames}` counts
  full-image wrapped work only on the owner and zero on followers.
  Plain LZ4/Bitcomp uses `host_rank_encoded_copy_s` and
  `host_rank_encoded_copy_{worker_sum_s,bytes,tensors}` instead; copy counters
  exclude uncompressed XOR masks. Inactive codec counters are zero.
- `host_rank_{arena_bytes,capacity_bytes,capacity_generation,mapping_reused}` and
  `host_rank_allocation_{s,calls,bytes}` describe local mappings and required bytes.
  CUDA ranks share physical capacity; its allocation calls count registrations.
  `host_rank_private_capacity_bytes` counts only private DE input storage.
  `host_rank_staging_copy_{s,bytes}` measures DE's local copy from the shared image;
  it is zero for CUDA decode.
  `host_encoded_cache_capacity_{bytes,generation}` and creator-only
  `host_encoded_cache_allocation_{s,calls,bytes}` describe shared tmpfs.
  `host_prepared_cache_*` describes the shared prepared image; `build_s` repeats
  its owner duration on followers and must not be summed. `host_rank_cpu_workers`
  records pool size, zero for followers on either path. Capacity counters exclude
  temporary native Zstd output slabs and do not measure total host RAM usage.

`scheduler_timing.blocked_s` is the pause metric: process-local `monotonic_ns` from
the start of the paused apply, just before it sets the pause flag and fences readers,
through resume clearing that flag. It includes the generation pause (retraction by
default), any flush, apply and the wait for that engine's ranks, excluding background
prepare and post-resume cleanup. Open,
failed or caller-retained pauses have null `resumed_ns`/`blocked_s`. Resume returns the
completed receipts without an extra RPC; this is neither GPU idle time nor HTTP latency
or first-token recovery.

NVTX range semantics are owned by `miles/utils/gpu_delta/profiling.py`, and those of
the asynchronous `gpu_delta.pause` range by `DeltaSession.apply` (`session.py`); the
[benchmark](../../../../tests/manual/gpu_delta/README.md#end-to-end-benchmark-from-an-hf-base)
shows how to profile a receiver.

## Validation

CPU tests live in `tests/fast/utils/gpu_delta` and
`tests/fast/backends/sglang_utils/gpu_delta`; the
[development guide](../../../../tests/manual/gpu_delta/README.md#manual-checks) lists
the manual CUDA checks.
