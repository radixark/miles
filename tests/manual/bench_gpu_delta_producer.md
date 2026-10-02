# GLM-5.2 producer comparison on one node

This manual benchmark compares **CPU XOR + Zstd**, **GPU XOR + Zstd**, and
**GPU XOR + Snappy** using the actual Megatron direct exporter and GPU-delta
publication protocol. It runs on eight GPUs with TP1/PP1/CP1/EP8/ETP1 and the
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
  The model is then unchanged while all three arms export it. Defaults select
  about 0.1% of elements and multiply those by 1.03125. These are controlled
  model perturbations, not learned optimizer steps, and do not target a fixed
  post-quantization density or compression ratio.
- **Comparison:** each arm has an independent canonical CPU baseline and
  publication stream. After all arms finish each version, compare their
  canonical CPU snapshots byte for byte, outside the measured intervals.
  Differences fail the run. Arm order rotates between versions. The initial
  discovery and three baseline exports warm the exporter before measurement;
  their durations are reported separately.
- **Compression:** nvCOMP GPU compression runs GPU SM kernels for both codecs.
  Blackwell's decompression engine does not accelerate compression. The CPU
  arm uses the existing owner worker pool, CPU XOR/Zstd and publication writer.
  Both GPU arms use the same production bounded asynchronous encoder path.

## Run

Use the same explicitly versioned CUDA 13 Miles development image and source
as the production path under test, with matching Megatron and Transformer
Engine (including its NVFP4 4over6 support). Install the nvCOMP runtime used by
`miles.utils.gpu_delta_nvcomp`; it uses the prebuilt library, not a runtime
compiled extension. Record the image tag/digest with the result. The script
records package versions, GPU model, topology arguments, and the required
quantization environment. It rejects conflicting environment values. If sources
are overlaid onto an image checkout, set `GPU_DELTA_SOURCE_DIGEST` to the externally
verified source manifest digest. `checkout_git_head` records checkout metadata
only and must not be used as the tested revision of an overlaid source tree.

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
are kept even if a later arm fails. `result.json` is written only after every
version passes the identical-target check. `setup.json` records initialization,
discovery, baseline capture, and rank ownership; `plan.json` contains the exact
mutable canonical inventory. Every completed version also gets its own JSON
and a concise JSON line on stdout. Nonzero `torchrun` exits remain failures;
do not use an earlier successful partial receipt as complete-run acceptance.

The flags select only this benchmark. Codec and encoder choices are fixed
explicitly for all three arms; no production defaults or live serving settings
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
| `export_loop_s` | Real exporter loop including conversion/quantization, its collectives and encoder queue backpressure. Background encoding overlaps this span. |
| `encoding_tail_s` | Remaining owner futures plus collective error agreement after the exporter returns. |
| `seal_and_visibility_s` | Payload/shard sealing, metadata gather and final manifest publication. |
| `conversion_host_s` | Sum of host intervals advancing the actual conversion/quantization iterator. Nested inside the exporter loop. |
| `conversion_cuda_ms` | Optional same-stream event spans around those conversions, read only after the final fence. Includes dispatch gaps or intervening work; not a pure quantization-kernel measurement. |
| `publication.producer_metrics` | Production per-owner encoding diagnostics, byte movement and optional GPU phase timings. Worker wall spans and streams can overlap one another and the exporter. |
| `sizes.changed_bytes / sizes.canonical_bytes` | Actual post-export canonical byte-change fraction. Separate from selected training elements. |
| `(sizes.payload_bytes + sizes.manifest_bytes) / sizes.canonical_bytes` | Actual complete compressed-publication fraction, including manifest metadata. Raw-fallback frames count as wire bytes. |

The nonoverlapping caller wall phases plus setup/final fence compose the
blocked interval. **Do not add conversion, compression, H2D/D2H or worker sums
to it:** those are nested or overlapping diagnostics. Likewise do not subtract
monotonic timestamps from different ranks. GPU events can perturb timing and
do not establish GPU-idle time. There is no concurrent trainer workload in this
benchmark, so it measures available overlap between export and encoding, not
the training throughput benefit of overlap.

With `--timing`, owner `tensor_phases` retains `baseline_h2d_s`, `xor_count_s`,
`compression_s`, `baseline_d2h_s` and `encoded_d2h_s` CUDA spans plus host
`metadata_wait_s`, `payload_wait_s` and `encoded_hash_write_s`. Keep the host
waits separate from GPU work; the waits can include GPU completion and host
scheduling. Phase sums across two worker streams are work-accounting diagnostics,
not the elapsed producer critical path.

The receiver benchmark in [GPU_DELTA.md](GPU_DELTA.md) measures another part
of the pipeline. Producer-only results do not establish end-to-end RL weight
sync speed, serving pause time, receiver correctness or numerical quality.
Changing weights rather than replaying a fixed quantized file is intentional:
the real exporter and quantizer costs remain inside the measured path.
