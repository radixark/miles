# GLM-5.2 producer comparison on one node

This manual benchmark compares **CPU XOR + Zstd**, **CPU XOR + Snappy**,
**GPU XOR + Zstd**, and **GPU XOR + Snappy** using the actual Megatron direct
exporter and GPU-delta publication protocol. Eight controlled arms retain all four
encoder/codec combinations at 1 MiB framing and add GPU Zstd/Snappy at 64 KiB and 2 MiB.
It runs on eight GPUs with TP1/PP1/CP1/EP8/ETP1 and the
native GLM-5.2 five-layer model (three dense layers and two routed MoE layers).
It constructs and loads the real model, exports W4A16 NVFP4 using TE 4over6,
and writes immutable owner publications. It does not launch a receiver or run
forward/backward, an optimizer, or an activation RPC.

- **Ownership:** the existing exporter quantizes each routed expert on its
  EP × EDP owner before the expert gather. This topology has EDP1. Nonrouted
  tensors are emitted by global rank 0 on the same node. Every rank participates
  in the normal exporter collectives.
- **Inputs:** supply matching prepared NVFP4 HF and native-DSA Megatron
  `torch_dist` checkpoints. The script reads them and keeps them unchanged.
  Its plan is built from names emitted by a real discovery export and original
  checkpoint headers. Static/unemitted calibration scales and draft weights
  are not added to the mutable plan. The canonical full-view plan is benchmark
  metadata, not a live SGLang admission/capability result.
- **Changes:** once per version, select approximately `--perturb-fraction`
  of each floating matrix's elements and multiply them by
  `1 + --perturb-relative-scale`. The deterministic selection depends on the
  global parameter name and version, so replicated dense weights agree.
  The model is then unchanged while all eight arms export it. Defaults select
  about 0.1% of elements and multiply those by 1.03125. These are controlled
  model perturbations, not learned optimizer steps, and do not target a fixed
  post-quantization density or compression ratio.
- **Comparison:** each arm has an independent canonical CPU baseline and
  publication stream. After all arms finish each version, compare their
  complete pending canonical CPU targets byte for byte, outside the measured
  intervals. GPU committed baselines still contain the previous version at this
  point. Differences fail the run. Only after all ranks pass does the harness
  explicitly acknowledge each sealed publication and commit its pending baseline
  for the next version. This is a producer-only simulated acknowledgment, not a
  receiver activation or acceptance proof. Arm order rotates between versions,
  starting from
  CPU Zstd, CPU Snappy, GPU Zstd, GPU Snappy, GPU Zstd 64 KiB, GPU Snappy
  64 KiB, GPU Zstd 2 MiB, GPU Snappy 2 MiB. The default three cumulative versions do not fully balance all eight
  execution positions and are not repeated measurements of one fixed target. The initial discovery and eight
  baseline exports warm the exporter before measurement;
  their durations are reported separately.
- **Compression:** nvCOMP GPU compression runs GPU SM kernels for both codecs.
  Blackwell's decompression engine does not accelerate compression. The CPU
  arms use the existing owner worker pool, CPU XOR with Zstd or Snappy, and publication writer.
  All GPU arms use the sole production bulk GPU encoder described below;
  the four 64 KiB/2 MiB arms change only the per-instance frame size. The earlier
  per-tensor GPU encoder has been removed; its saved results are historical controls.

The GPU path first stages the full new canonical snapshot to pinned CPU memory
as export proceeds. After that D2H work finishes, it encodes deterministic
name-sorted batches using the existing `update_weight_buffer_size` target
(512 MiB in this benchmark). A larger individual tensor stands alone; the
value is a batching target, not a hard workspace-memory ceiling. Each batch
uploads old and new bytes, computes XOR/counts, compresses all independent 1 MiB
frames together (64 KiB or 2 MiB in the explicit framing controls), and returns encoded
payloads to pinned CPU memory. After all
GPU batches complete, the owner hashes/writes the payloads and seals the
publication. The old CPU snapshot stays unchanged until acknowledgment.

This trades per-tensor compression dispatch/fences for bulk work, but GPU
encoding no longer overlaps later exports. For N canonical bytes, it transfers
N new bytes D2H plus N old and N new bytes H2D (3N total), in addition to encoded
payload D2H. The CPU control keeps its existing export-overlapped worker path.
No new batching knob or performance conclusion is introduced here; measure the
new pipeline separately from earlier per-tensor GPU results.

## Supported producer configurations

All four combinations use the existing `gpu-delta` publication protocol. CPU
Snappy is selected through the existing CPU encoder; no separate implementation
or fallback is added by this benchmark.

| Arm | `WEIGHT_DELTA_ENCODER` | `WEIGHT_DELTA_CODEC` | Frame bytes | XOR/compression execution |
| --- | --- | --- | --- | --- |
| `cpu-zstd` | `cpu` | `zstd` | 1,048,576 | CPU worker pool, Zstd |
| `cpu-snappy` | `cpu` | `snappy` | 1,048,576 | CPU worker pool, Snappy |
| `gpu-zstd` | `gpu` | `zstd` | 1,048,576 | GPU XOR, nvCOMP CUDA compression |
| `gpu-snappy` | `gpu` | `snappy` | 1,048,576 | GPU XOR, nvCOMP CUDA compression |
| `gpu-zstd-64k` | `gpu` | `zstd` | 65,536 | Same GPU encoder, per-instance framing control |
| `gpu-snappy-64k` | `gpu` | `snappy` | 65,536 | Same GPU encoder, per-instance framing control |
| `gpu-zstd-2m` | `gpu` | `zstd` | 2,097,152 | Experimental producer-only framing control |
| `gpu-snappy-2m` | `gpu` | `snappy` | 2,097,152 | Experimental producer-only framing control |

The last four arms are benchmark variants, not a new production environment or
CLI knob. Production retains 1 MiB. Each variant owns a separate encoder and
publication stream; no process-global frame-size monkeypatch can leak into a
later arm. The publication records its frame profile explicitly:
`<codec>-independent-64kib-v1`, `<codec>-independent-1mib-v1`, or
`<codec>-independent-2mib-v1`; the benchmark checks this against the arm.

**Receiver compatibility:** the paired SGLang receiver currently admits at most
1 MiB decoded frames. The 2 MiB arms are producer-only experiments and their
publications must not be sent to that receiver. Native producer round trips
can qualify compression bytes independently; they do not establish receiver
admission. Production retains 1 MiB and the receiver interoperability suite
remains scoped to 64 KiB/1 MiB.

Receiver decode is determined by codec, independently of the producer's CPU/GPU
choice: Zstd uses CUDA decoding; Snappy requires Blackwell hardware decompression.
The benchmark itself has no receiver. Production defaults remain GPU Snappy.

## Run

Use the same explicitly versioned CUDA 13 Miles development image and source
as the production path under test, with matching Megatron and Transformer
Engine (including its NVFP4 4over6 support). Install the nvCOMP runtime used by
`miles.utils.gpu_delta_nvcomp`; it uses the prebuilt library, not a runtime
compiled extension. Record the image tag/digest with the result. The script
records package versions, GPU model, topology arguments, and the required
quantization environment. It rejects conflicting environment values. If sources
are overlaid onto an image checkout, set `GPU_DELTA_SOURCE_DIGEST` to the externally
verified source manifest digest. The benchmark does not require a Git checkout;
the recorded source digest identifies the externally verified source tree.

From the Miles checkout, with its dependencies and native-DSA checkpoint
conversion already prepared:

```bash
export PYTHONPATH="$PWD:/root/Megatron-LM:/root/TransformerEngine${PYTHONPATH:+:$PYTHONPATH}"
torchrun --standalone --nproc-per-node=8 \
  tests/manual/bench_gpu_delta_producer.py \
  --hf-checkpoint /data/models/GLM-5.2_5layer-NVFP4 \
  --load /data/models/GLM-5.2_5layer-megatron-dsa_torch_dist \
  --output /data/benchmarks/gpu-delta-producer-001 \
  --versions 3 \
  --perturb-fraction 0.001 \
  --perturb-relative-scale 0.03125 \
  --timing
```

The output directory must be new. Raw per-arm/version receipts and publications
are kept even if a later arm fails. `result.json` is written only after all eight
arms in every version pass the identical-target check. `setup.json` records initialization,
discovery, baseline capture, rank ownership, the ordered `arms` list and exact
`arm_order_by_version`. `arm_configs` records each arm's encoder, codec and
frame bytes. `receiver_compatibility` identifies the two producer-only 2 MiB
arms and current receiver frame limit. It also binds the `producer_pipelines` labels and existing
`gpu_batch_target_bytes` (also per GPU arm) to the captured source. `plan.json`
contains the exact mutable canonical inventory;
every completed version repeats its actual `order`. Three versions produce 24
arm/version observations, not repeated samples of a fixed target.
Every completed version also gets its own JSON
and a concise JSON line on stdout. Nonzero `torchrun` exits remain failures;
do not use an earlier successful partial receipt as complete-run acceptance.

The flags select only this benchmark. Codec and encoder choices are fixed
explicitly for all eight arms; no production defaults or live serving settings
are changed. To measure event overhead, rerun into a separate output directory
without `--timing`, using the same inputs and perturbation arguments. Never
merge timing-enabled and timing-disabled samples into one distribution.

## Timings and ratios

Per-rank wall spans use that process's monotonic clock. Report median and range
over all eight ranks, retaining rank 0 separately because it owns ordinary
tensors as well as experts.

| Field | Meaning |
| --- | --- |
| `producer_blocked_s` | Caller time from update setup through export, background-work tail, immutable publication and final completion fence; excludes pre-arm barrier, target perturbation and correctness comparisons. |
| `export_loop_s` | Real exporter loop including conversion/quantization and collectives. GPU: stages the new snapshot D2H. CPU: overlaps encoding workers and includes their queue backpressure. |
| `encoding_tail_s` | GPU: final export D2H wait, bulk encoding, encoded-payload hash/write, and collective agreement. CPU: remaining owner futures and collective agreement. |
| `seal_and_visibility_s` | Payload/shard sealing, metadata gather and final manifest publication. |
| `conversion_host_s` | Sum of host intervals advancing the actual conversion/quantization iterator. Nested inside the exporter loop. |
| `conversion_cuda_ms` | Optional same-stream event spans around those conversions, read only after the final fence. Includes dispatch gaps or intervening work; not a pure quantization-kernel measurement. |
| `publication.producer_metrics` | Production per-owner diagnostics, byte movement and optional GPU phase timings. GPU includes `export_staging_wait_s`, `bulk_encode_s`, `encoded_hash_write_s`, and `encoder_batches`; these are nested in caller phases. CPU worker spans can overlap each other and the exporter. |
| `sizes.changed_bytes / sizes.canonical_bytes` | Actual post-export canonical byte-change fraction. Separate from selected training elements. |
| `(sizes.payload_bytes + sizes.manifest_bytes) / sizes.canonical_bytes` | Actual complete compressed-publication fraction, including manifest metadata. Raw-fallback frames count as wire bytes. |

The nonoverlapping caller wall phases plus setup/final fence compose the
blocked interval. **Do not add conversion, compression, H2D/D2H or worker sums
to it:** those are nested or overlapping diagnostics. Likewise do not subtract
monotonic timestamps from different ranks. GPU events can perturb timing and
do not establish GPU-idle time. There is no concurrent trainer workload in this
benchmark, so it does not establish the training throughput benefit of overlap.

With `--timing`, owner `tensor_phases` retains GPU `baseline_h2d_s`,
`current_h2d_s`, `xor_count_s`, `compression_s`, and `encoded_pack_d2h_s` spans
plus host `metadata_wait_s` and `payload_wait_s`. Shared batch timings appear
only on the batch's first entry with `timing_scope="batch"`; do not multiply
these timings by its tensor count. `encoded_pack_d2h_s` includes GPU payload
packing and its D2H copy, not just PCIe transfer. The final encoded file
hash/write span is reported once per owner. Keep host waits separate from GPU
work; waits include GPU completion and possibly host scheduling. No nested
span sum is an additional producer latency.

The receiver benchmark in [GPU_DELTA.md](GPU_DELTA.md) measures another part
of the pipeline. Producer-only results do not establish end-to-end RL weight
sync speed, serving pause time, receiver correctness or numerical quality.
Changing weights rather than replaying a fixed quantized file is intentional:
the real exporter and quantizer costs remain inside the measured path.
