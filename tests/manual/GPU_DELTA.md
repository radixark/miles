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
| `WEIGHT_DELTA_TIMING=1` | Record per-phase CUDA events for profiling. Default: off; event instrumentation can perturb timing. |
| `WEIGHT_DELTA_SNAPPY_ZSTD=1` | Opt in to a CPU Zstd envelope around each tensor's GPU Snappy/raw frames. Default: off; requires GPU Snappy and the paired protocol-3 receiver. |

For Ray launches, pass these settings explicitly in the job's
`runtime_env={"env_vars": ...}` (or `execute_train(extra_env_vars=...)`). Setting
only the submitting shell does not forward arbitrary variables to workers on an
existing Ray cluster. The five-layer W4A16 E2E explicitly forwards the codec,
encoder and outer-Zstd flag, and checks every learned publication's protocol and
codec profile. An existing `--train-env-vars` override can set producer-only
values on trainer actors; rollout profiling additionally needs
`WEIGHT_DELTA_TIMING=1` in the rollout/job environment. Receivers select the outer
envelope from the publication descriptor, not from a receiver-side opt-in flag.

Codec and producer execution are independent settings. All four combinations
are supported by the existing GPU-delta publication path:

| Producer | Configuration | Compression implementation |
| --- | --- | --- |
| CPU Zstd | `WEIGHT_DELTA_ENCODER=cpu WEIGHT_DELTA_CODEC=zstd` | CPU Zstd worker |
| CPU Snappy | `WEIGHT_DELTA_ENCODER=cpu WEIGHT_DELTA_CODEC=snappy` | CPU Snappy worker |
| GPU Zstd | `WEIGHT_DELTA_ENCODER=gpu WEIGHT_DELTA_CODEC=zstd` | nvCOMP CUDA compression |
| GPU Snappy (default) | `WEIGHT_DELTA_ENCODER=gpu WEIGHT_DELTA_CODEC=snappy` | nvCOMP CUDA compression |

The CPU choices compute XOR and compression on CPU; the GPU choices compute
both on GPU. CPU Snappy requires the existing `python-snappy` dependency.
Receiver decode depends on the codec, not the encoder location: either CPU- or
GPU-produced Zstd uses CUDA decoding, and either Snappy producer requires the
qualified Blackwell hardware decoder. There is no silent execution fallback.

Both GPU codecs use the same bulk encoder. It replaces the earlier per-tensor
GPU implementation; there is no legacy GPU selection or fallback. The CPU
encoders remain independent choices in the matrix above.

The GPU producer keeps old canonical bytes in pinned CPU memory and stages the
full new snapshot D2H during export. It then encodes name-sorted batches using
the existing `update_weight_buffer_size` target: upload old and new bytes,
compute XOR/counts, compress independent 1 MiB frames across tensors, and copy
encoded payloads back to pinned CPU memory. A tensor larger than the target
stands alone. File hash/write follows all GPU batches. Only successful receiver
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

## Matched outer-Zstd receiver comparison

Derive a new fixture from a completed saved Snappy fixture. Pin the saved
`fixture.json` SHA from its receipt; do not recreate or re-export model weights:

```bash
python tests/manual/bench_gpu_delta.py wrap-fixture \
  --fixture /data/gpu-delta/fixture \
  --fixture-sha256 <saved-fixture-json-sha256> \
  --output /data/gpu-delta/fixture-snappy-outer
```

This CPU-only operation reads existing encoded payload files, validates their
SHA-256s, and uses the production `PublicationWriter` to add one CPU Zstd level-1
envelope per nonempty canonical tensor. It preserves natural tensor boundaries,
inner frame sizes, the negotiated plan, stream/version metadata, and the same
final altered checkpoint. It then decompresses every written envelope and
checks each reconstructed Snappy/raw frame byte for byte against the saved
source. No model tensor is read or hashed, no GPU compression is run, and no
checkpoint is copied or changed. The new `derivation.json` retains complete or
failed/partial evidence. Existing publications remain intact and addressable
from the new fixture alongside its new `snappy-zstd` entries.

**Input provenance:** these standalone fixture Snappy frames were produced by
the historical CPU `snappy.compress` builder in the same independent-block
format consumed by nvCOMP. They are not a new GPU-producer measurement. The
matched five-layer producer benchmark measures actual GPU Snappy plus outer
Zstd; native producer/receiver interoperability tests cover that exact chain.

Run each receiver arm in its own fresh engine, with matching timing settings:

```bash
WEIGHT_DELTA_ENCODER=gpu WEIGHT_DELTA_CODEC=snappy \
WEIGHT_DELTA_SNAPPY_ZSTD=0 WEIGHT_DELTA_TIMING=1 \
  python tests/manual/bench_gpu_delta.py run --model /models/GLM5.2-NVFP4 \
  --fixture /data/gpu-delta/fixture-snappy-outer --output /data/gpu-delta/snappy-control
WEIGHT_DELTA_ENCODER=gpu WEIGHT_DELTA_CODEC=snappy \
WEIGHT_DELTA_SNAPPY_ZSTD=1 WEIGHT_DELTA_TIMING=1 \
  python tests/manual/bench_gpu_delta.py run --model /models/GLM5.2-NVFP4 \
  --fixture /data/gpu-delta/fixture-snappy-outer --output /data/gpu-delta/snappy-outer
```

The opt-in chooses the new descriptor through the normal activation API. It
does not select an alternative benchmark apply implementation. Compare source
file bytes, CPU outer decode/preparation time, reconstructed inner bytes, and
the unchanged Snappy/raw H2D bytes as well as coordinator time and scheduler
blocked time. CPU outer decoding occurs during preparation; it must not be
added to nested coordinator or pause spans. Keep version 1 first-use allocation
separate from warm versions 2/3. The same fresh-target oracle remains valid
because the derived fixture points to the identical altered checkpoint, but
each new arm still needs its own exact output comparison. Repeat with timing
disabled as a separate run before attributing effects to instrumentation.

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
actual scheduler pause, GPU H2D, decode, layout/application and derived refresh.
Do not add nested event spans or sum concurrent ranks. Receivers stream only the
current locally needed tensor to HBM and skip unowned expert payloads. Compare
transferred bytes as well as timing. No throughput claim follows from
this idle-engine benchmark; preparation overlap under generation load needs its
own measurement.

The harness terminates only its own engine processes. It does not release a
cluster allocation. Failed observations and partial fixtures remain on disk.
