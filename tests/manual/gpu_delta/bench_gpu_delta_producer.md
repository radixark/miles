# GLM-5.2 GPU-delta producer benchmark on one node

This benchmark uses the actual Megatron GPU-delta iterator and publication
protocol on eight GPUs: configurable TP/CP with PP1/EP8/ETP1, a native GLM-5.2
five-layer model (three dense and two routed MoE layers), and NVFP4 TE 4over6 quantization.
It loads the real model and writes three cumulative immutable publications.
It does not launch a receiver or run forward/backward, an optimizer, or activation.

- **Codec:** `GPU_DELTA_CODEC` selects `snappy-zstd` (default), `lz4-zstd` or plain `lz4`.
  Matrices use GPU XOR and GPU Snappy or LZ4. Wrapped codecs then use owner-wide
  GPU Zstd. All compression runs on GPU SMs; hardware DE is a receiver feature.
- **Ownership:** routed experts remain on their EP × EDP exporter owners before
  expert gathering; this benchmark has EDP1. Nonrouted tensors use the GPU-delta
  iterator's fixed native owner plan after TP reconstruction. Discovery records
  every native conversion, checks its assigned owner, and verifies exactly-once
  canonical output coverage across all ranks. No global-rank0 ownership is assumed.
- **Inputs:** supply matching prepared NVFP4 HF and native-DSA Megatron
  `torch_dist` checkpoints. They remain immutable. A discovery export and original
  checkpoint headers define the exact mutable inventory. Unemitted calibration
  scales and the static draft are excluded. This plan is benchmark metadata,
  not a live SGLang capability or identity proof.
- **Perturbations:** once per version, select approximately `--perturb-fraction`
  of each floating matrix and multiply by `1 + --perturb-relative-scale`.
  Selection depends on parameter name/version, so replicated weights agree.
  Defaults select 0.1% of elements and multiply by 1.03125. These are controlled
  changes, not learned optimizer steps or a fixed post-quantization delta ratio.
- **Correctness boundary:** after sealing, check every rank's complete pending
  target names/byte sizes against discovered ownership, outside timing. Only then
  simulate receiver acknowledgment and commit the pending baseline for the next
  version. Native tests independently decode and compare payload bytes. This
  single-codec benchmark does not prove target equality with another run.

## Pipeline and memory

The [production pipeline](README.md#producer-and-receiver-pipeline) stages each
new owner-local export into pinned CPU memory. Per-bucket stream dependencies
and source leases protect asynchronous D2H copies. Cached matrix batches follow
baseline callback order and become eligible when their export event completes,
so compression can overlap later exports. `update_weight_buffer_size` is a
512 MiB input-batch target here; larger individual tensors remain whole.

Each batch uploads old/new snapshots, computes XOR/counts with parallel 64 KiB
tiles, and submits its inner frames together. `--frame-bytes` defaults to
1 MiB; optional outer Zstd chunks stay at most 1 MiB. Compact inner arenas remain
in HBM until owner-wide finalization: optional GPU Zstd, one final pack/D2H,
then CPU writes and optional hashing. Raw scalar/vector targets use a separate
CPU writer. All jobs drain before sealing, and the old snapshot stays unchanged
until the benchmark's explicit acknowledgment.

For C matrix bytes and R scalar/vector bytes, transfers are C+R new bytes D2H,
C old plus C new bytes H2D, and final encoded payload D2H. Raw targets never enter
GPU compression. Budget the pinned snapshots, full compact owner payload,
bounded canonical scratch and codec workspaces; there is no OOM fallback.
This benchmark does not execute the receiver pipeline described in the README.

## Run

Use a matching explicit Miles CUDA 13 image, paired feature checkout, prebuilt
nvCOMP 5.x (at least 5.3), compatible FlashInfer and TransformerEngine, and the prepared
five-layer checkpoints. Install nvCOMP without changing the image dependency
closure as documented in [README.md](README.md).

```bash
export PYTHONPATH=/workspace/sglang/python:/workspace/miles:/root/Megatron-LM
export GPU_DELTA_CODEC=snappy-zstd
python -m torch.distributed.run --standalone --nproc-per-node=8 \
  tests/manual/gpu_delta/bench_gpu_delta_producer.py \
  --hf-checkpoint /models/GLM5.2-5layer-NVFP4 \
  --load /models/GLM5.2-5layer-megatron-torch_dist \
  --output /data/gpu-delta/producer-new --versions 3 \
  --tensor-model-parallel-size 2 --context-parallel-size 2 \
  --perturb-fraction 0.001 --perturb-relative-scale 0.03125
```

This TP2/CP2/PP1/EP8/ETP1 command has DP2 and EDP1. The benchmark defaults to
TP1/CP1. Match both topology flags, checkpoints, perturbations and timing mode
when comparing sources.
Rank ownership may change between sources, so compare canonical target bytes by
name across the global inventory rather than requiring identical per-rank shards.

The `GPU_DELTA_*` environment variables are development/debug controls, not a
stable user-facing configuration API. Output must be a new directory.
The harness records runtime package versions,
model flags, source digest (from `GPU_DELTA_SOURCE_DIGEST` when provided),
GPU memory counters, original rank ownership and all per-version measurements.
`--timing` enables CUDA phase events; default timing is off to avoid event overhead.
Select `GPU_DELTA_CODEC=lz4-zstd` or `GPU_DELTA_CODEC=lz4` with a new output
directory and identical workload flags/checkpoints for a codec comparison.
The harness always uses the production GPU encoder.

## Timing interpretation

| Field | Scope |
|---|---|
| `producer_blocked_s` | Update setup through export, encoding tail, immutable publication and final completion fence; excludes pre-update barrier, perturbation and inventory check. |
| `measurement.rank_max_s` | Maximum of each caller host interval over all eight ranks; raw per-rank values remain in `measurement.ranks`. |
| `export_loop_s` | Actual conversion/quantization/collectives and new snapshot D2H staging. |
| `encoding_tail_s` | Remaining export D2H, bulk GPU encoding, owner hash/write tails and collective agreement. |
| `seal_and_visibility_s` | Owner shard sealing, metadata gather and final manifest publication. |
| `publication.manifest_seal_s` | Root manifest validation, serialization, hash and exclusive publication, nested in seal/visibility. |
| `conversion_host_s` | Host time advancing the real conversion iterator, nested in export. |
| `conversion_cuda_ms` | Optional same-stream conversion events, read after final fence; includes dispatch gaps and intervening work. |
| `publication.producer_metrics` | Per-owner nested encoding/wait/raw/write phases and source-accounted copy bytes. |
| `gpu_memory_bytes` | Before/after/peak PyTorch allocated/reserved bytes over the isolated update. |
| `sizes` | Changed canonical bytes, raw bytes, inner codec/wire-envelope/payload/manifest sizes. Plain LZ4 has `outer_frame_bytes=0`; its stored and decoded arena lengths are equal. |

The `finalize_wall_s` counter encloses finalization for every codec.
`outer_zstd_wall_s` measures Zstd submission, metadata wait and output
selection; it is zero for plain LZ4. `pack_d2h_wall_s` measures packing
and the final pinned D2H wait separately. These are host intervals, not pure
GPU kernel timings. Plain LZ4 reports zero `outer_zstd_frames` and no Zstd CUDA
phase. `matrix_payload_bytes` counts the stored matrix envelope, while
`inner_arena_bytes` counts its decoded or directly stored inner arena.

The protocol also retains rank-local `publication_metrics["metadata_gather_s"]`
after the existing gather completes. It measures metadata serialization/transport
and rank wait, not payload transfer, and is too late for that same gather's owner
prefix. External diagnostic observers can collect it outside the measured span;
the production path adds no collective just for this clock.

Report all rank ranges/medians plus rank 0 separately. V1 includes first-use
allocation/compilation; V2/V3 are warm cumulative versions, not independent
fixed-target trials. Discovery and baseline export warm the exporter before
measurement and are reported separately. No concurrent trainer workload exists,
so this cannot establish training throughput or realized overlap.

Nonoverlapping caller phases compose blocked time. Do not add nested conversion,
compression, transfer or worker spans, or subtract monotonic timestamps across
ranks. CUDA events can perturb timing and do not measure GPU idle time. Logical
copy-byte counters are not bus measurements. `resident_inner_hbm_bytes` records
unique retained inner storage at finalization entry, not allocator peak/workspace totals.
The existing pre-update fence brackets peak-stat reset; reset does not empty the
CUDA cache, and reserved memory can include earlier work. No host RSS/pinned-peak
or untracked native allocation measurement is claimed.

Compare the [full-model receiver benchmark](README.md) separately. Its input
fixture and timings differ from this five-layer producer workload; do not subtract
one from the other to claim end-to-end RL savings.

## Full-model and multi-node scope

The five-layer TP/CP-configurable PP1/EP8/ETP1 run is a correctness and profiling
proxy, not full-model capacity, multi-node performance or training-throughput
validation. Production ownership remains PP-stage-local: ETP1 experts use their
EP × EDP owners; ETP>1 uses the gather-before-convert path. Canonical inventory
must retain complete, unique coverage, including across PP/MTP boundaries.

`update_weight_buffer_size` is a matrix input-batch target, not a total memory
cap. Budget the pinned old/pending snapshots, the full owner-local compact inner-codec
payload and codec scratch/workspaces on the intended topology. Keep the exact
inventory and reconstructed-target checks when extending this benchmark; the
measured proxy's owner imbalance or compression ratio is not a sizing rule.
