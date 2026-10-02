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

The producer keeps old canonical bytes in pinned CPU memory. Two workers per
owner overlap old-byte H2D, GPU XOR/compression, new-baseline D2H and encoded
payload writes with subsequent exports. Export, worker backpressure, final
encoding drain and publication/activation barriers still block the trainer.
Routed experts retain exporter EP/EDP ownership; non-routed tensors retain the
existing data-replica sender. The producer-only comparison is documented in
[bench_gpu_delta_producer.md](bench_gpu_delta_producer.md).

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
