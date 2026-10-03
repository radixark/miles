# GPU-delta development benchmark

The experimental `--update-weight-transfer-mode gpu-delta` uses the paired
SGLang `update_weights_from_delta` API. `disk-delta` continues to reconstruct a
checkpoint and use the existing disk reload API. The two paths do not fall back
to each other.

## Environment

Use a CUDA 13 Miles development image, the paired SGLang branch on `PYTHONPATH`,
and eight Blackwell GPUs. The benchmark starts one TP8/DP8/EP8 engine with
GLM5.2 NVFP4 W4A16, CuTe DSL MoE, no MoE A2A and a static bundled MTP draft.
The original checkpoint must already be available and is never modified.

Install the prebuilt encoder/decoder without changing the image's dependency closure:

```bash
python -m pip install --no-deps nvidia-libnvcomp-cu13==5.3.0.16
```

No custom C++/CUDA extension is built. GPU production uses nvCOMP CUDA
compression for both codecs; Zstd decoding uses its CUDA backend; Snappy
explicitly requests hardware decompression and rejects unsupported hardware or
allocation modes. Both decode from HBM. Pinned host memory supplies H2D copies.

| Environment variable | Meaning |
| --- | --- |
| `WEIGHT_DELTA_CODEC=zstd\|snappy` | Producer codec; each immutable frame records its actual codec, including raw fallback for incompressible frames. Default: `snappy`. |
| `WEIGHT_DELTA_ENCODER=gpu\|cpu` | GPU XOR/compression with a pinned CPU baseline, or the explicit CPU reference encoder. Default: `gpu`. |
| `WEIGHT_DELTA_SNAPPY_OUTER=cpu\|gpu` | Snappy outer Zstd execution. Default: `cpu`; `gpu` requires the GPU Snappy producer and protocol-4 receiver. |
| `WEIGHT_DELTA_TIMING=1` | Record per-phase CUDA events for profiling. Default: off; event instrumentation can perturb timing. |

For Ray launches, pass these settings explicitly in the job's
`runtime_env={"env_vars": ...}` (or `execute_train(extra_env_vars=...)`). Setting
only the submitting shell does not forward arbitrary variables to workers on an
existing Ray cluster. The five-layer W4A16 E2E explicitly forwards the codec,
encoder and outer compressor, and checks every learned publication's protocol and
codec profile. An existing `--train-env-vars` override can set producer-only
values on trainer actors; rollout profiling additionally needs
`WEIGHT_DELTA_TIMING=1` in the rollout/job environment. Receivers select the outer
envelope from the publication descriptor.

Codec and producer execution are independent settings. All four combinations
are supported by the existing GPU-delta publication path:

| Producer | Configuration | Compression implementation |
| --- | --- | --- |
| CPU Zstd | `WEIGHT_DELTA_ENCODER=cpu WEIGHT_DELTA_CODEC=zstd` | CPU Zstd worker |
| CPU Snappy | `WEIGHT_DELTA_ENCODER=cpu WEIGHT_DELTA_CODEC=snappy` | CPU Snappy + CPU Zstd envelope |
| GPU Zstd | `WEIGHT_DELTA_ENCODER=gpu WEIGHT_DELTA_CODEC=zstd` | nvCOMP CUDA compression |
| GPU Snappy (default) | `WEIGHT_DELTA_ENCODER=gpu WEIGHT_DELTA_CODEC=snappy` | nvCOMP CUDA Snappy + CPU Zstd envelope |

The CPU choices compute XOR and inner compression on CPU; the GPU choices
compute them on GPU. The Snappy outer envelope defaults to CPU Zstd. CPU Snappy requires the existing `python-snappy` dependency.
Receiver decode depends on the codec, not the encoder location: either CPU- or
GPU-produced Zstd uses CUDA decoding, and either Snappy producer requires the
qualified Blackwell hardware decoder. Default Snappy publications use a per-tensor CPU Zstd level-1 envelope (protocol 3,
`snappy-independent-1mib-zstd-v1`). Background preparation unwraps locally needed
tensors into pinned buffers before hardware Snappy decode. Native Zstd keeps protocol 2.

The experimental `WEIGHT_DELTA_SNAPPY_OUTER=gpu` producer opt-in keeps compact
Snappy tensor arenas on device across all owner batches, then compresses their
independent <=1 MiB outer chunks together in one nvCOMP Zstd call. Only the final
wrapped wire bytes return to pinned CPU memory for hashing/writing and owner
gathering; there is no intermediate Snappy host payload slab. Raw scalar/vector
CPU writes overlap the matrix GPU work as before. Protocol 4 records
`snappy-independent-1mib-gpu-zstd-v1`, natural tensor boundaries and explicit
outer chunk offsets/lengths. SHA-256 covers final owner files; inner Snappy bytes
are not read back merely to hash them. The receiver verifies the final files and CPU-decompresses the authenticated
Zstd chunks into pinned Snappy tensor buffers during background preparation,
including when the producer used GPU Zstd. Its existing streamed H2D and hardware
Snappy apply path is unchanged: GPU outer decoding is not implemented. The GPU
sender avoids its intermediate Snappy host slab; the receiver still owns pinned
Snappy buffers. There is no implicit producer codec/backend fallback.

For a matched full-model receiver fixture, run the setup-only adapter below.
It rewraps retained, authenticated Snappy arenas and verifies exact recovered
bytes; fixture construction is excluded from update timings.

```bash
PYTHONPATH=. python tests/manual/rewrap_gpu_delta_fixture.py \
  --fixture /data/benchmarks/cpuouter/fixture.json \
  --fixture-sha256 <fixture-sha256> \
  --output /data/benchmarks/gpuouter-new \
  --source-digest <verified-source-manifest-sha256> --device 0
```

Run the usual receiver benchmark with `WEIGHT_DELTA_SNAPPY_OUTER=gpu` to select
this fixture's `snappy-gpu-zstd` entries, still CPU-unwrapped by the receiver.
Benchmark the default and opt-in
independently; no performance preference is implied by the experimental mode.
There is no silent execution fallback.

Both GPU codecs use the same bulk encoder. It replaces the earlier per-tensor
GPU implementation; there is no legacy GPU selection or fallback. The CPU
encoders remain independent choices in the matrix above.

The GPU producer keeps old canonical bytes in pinned CPU memory and stages the
full new snapshot D2H during export. It then encodes name-sorted batches using
the existing `update_weight_buffer_size` target: upload old and new bytes,
compute XOR/counts, compress independent 1 MiB frames across tensors, and copy
encoded payloads back to pinned CPU memory. A tensor larger than the target
stands alone. Snappy uses one owner CPU worker to wrap/hash/write completed
batches while later GPU batches encode; Zstd file hash/write follows encoding.
All owner work drains before publication sealing. Only successful receiver
activation commits the new baseline; old and pending CPU snapshots remain
separate until then. This bulk path removes per-tensor compression fences but
adds new-byte H2D: N canonical bytes require 3N transfer bytes (new D2H, old/new
H2D), plus compressed payload D2H. Export, bulk encoding, publication and
activation still block the trainer. GPU compression no longer overlaps later
exports; the CPU reference keeps its existing worker overlap. Earlier
per-tensor GPU timings do not measure this new pipeline.
Routed experts retain exporter EP/EDP ownership; non-routed tensors retain the
existing data-replica sender. The producer-only comparison is documented in
[bench_gpu_delta_producer.md](bench_gpu_delta_producer.md). It compares the four
1 MiB encoder/codec combinations plus GPU Zstd/Snappy 64 KiB and 2 MiB controls,
using the same three cumulative targets across all eight arms. Frame size is an
internal benchmark control; production retains 1 MiB with no new tuning knob.
The paired receiver admits at most 1 MiB decoded frames: 2 MiB publications are
producer-only experiments and are not receiver-compatible. Rotating eight arms
over three versions is not fully balanced repeated sampling of a fixed target.

CPU SHA-256 checks encoded files during background preparation. Runtime updates
do not hash old or new weights. Session/version/incarnation checks prevent stale
or repeated application; they do not establish exact weight-content equality.

## Persistent fixture

Run from the Miles checkout. Replace the paths below with your checkpoint,
paired source checkout and persistent storage. The output directory must be new.
Allow space for one complete altered checkpoint and both codec publications.

```bash
export PYTHONPATH=/workspace/sglang/python:/workspace/miles
python tests/manual/bench_gpu_delta.py inventory \
  --model /models/GLM5.2-NVFP4 --output /data/gpu-delta/inventory
python tests/manual/bench_gpu_delta.py fixture \
  --model /models/GLM5.2-NVFP4 \
  --inventory /data/gpu-delta/inventory/inventory.json \
  --output /data/gpu-delta/fixture --ratio 0.002 --versions 3
```

The fixture calibrates sparse finite mantissa/packed-FP4 perturbations against
representative 1 MiB Zstd frames. It reports the actual complete publication ratio,
including frame metadata/padding. Zstd and Snappy encode the same successive
changed weights; Snappy is not independently tuned to 0.2%. Scale tensors and
static draft weights remain unchanged in the large proxy; focused receiver tests
cover scale changes. The altered checkpoint contains the final version.

The builder writes wrapped Snappy directly through the production publication
writer, preserving canonical tensor boundaries and inner raw fallback. Fixture
Snappy frames come from CPU `snappy.compress`; this is not a GPU-producer
measurement. Native interoperability tests and the separate five-layer producer
benchmark cover actual GPU-origin frames. Publication accounting reports inner
frame bytes, outer stored/decoded bytes and complete file/manifest bytes
separately; outer compression does not change the canonical denominator.

Default Snappy selects the `snappy-zstd` fixture entry and validates protocol 3
and its codec profile before launching an engine. `WEIGHT_DELTA_SNAPPY_OUTER=gpu`
selects the rewrapped `snappy-gpu-zstd` entry and validates protocol 4; both use
CPU outer decoding followed by streamed hardware Snappy decoding. A previously saved wrapped
fixture can be reused with `--fixture` without reading/re-exporting model weights;
its target checkpoint, plan and versions remain unchanged. Historical plain
`snappy` entries are never a fallback. New fixtures use the same deterministic
key, so no fixture migration command is needed.

## Compare both receiver codecs

Each invocation starts a fresh engine from the same original checkpoint. It
applies all three immutable publications to the engine and saves
all original-rank receipts, server logs and generation through DP routes 0–7.

```bash
WEIGHT_DELTA_CODEC=zstd WEIGHT_DELTA_TIMING=1 \
  python tests/manual/bench_gpu_delta.py run --model /models/GLM5.2-NVFP4 \
  --fixture /data/gpu-delta/fixture --output /data/gpu-delta/zstd-tensor
WEIGHT_DELTA_CODEC=snappy WEIGHT_DELTA_TIMING=1 \
  python tests/manual/bench_gpu_delta.py run --model /models/GLM5.2-NVFP4 \
  --fixture /data/gpu-delta/fixture --output /data/gpu-delta/snappy-tensor
python tests/manual/bench_gpu_delta.py oracle --model /models/GLM5.2-NVFP4 \
  --fixture /data/gpu-delta/fixture --output /data/gpu-delta/target-oracle
```

The oracle starts from the final altered checkpoint and keeps the original static
draft. Its generation/logprobs are an untimed functional comparison, not proof of
every live weight byte. Exact decode/layout/application comparisons belong to
receiver unit tests. The updated five-layer W4A16 E2E additionally exercises real
training-driven publications; it is not a full-model RL validation.

Separate coordinator wall time, background read/hash/pin preparation,
**Explicit scheduler pause** (pause flag to resume), GPU H2D, decode,
layout/application and derived refresh. The pause excludes prepare/status-handler
service before the pause and does not quantify rollout interference.
Do not add nested event spans or sum concurrent ranks. Receivers stream only the
current locally needed tensor to HBM. Every scheduler still reads/hashes all
compressed owner files, then unwraps only locally needed tensors and retains
their pinned arenas through commit; HBM scratch is reused per tensor. Compare
transferred bytes as well as timing. No throughput claim follows from
this idle-engine benchmark; preparation overlap under generation load needs its
own measurement.

The harness terminates only its own engine processes. It does not release a
cluster allocation. Failed observations and partial fixtures remain on disk.
