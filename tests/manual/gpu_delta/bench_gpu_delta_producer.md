# GLM-5.2 Snappy-Zstd producer benchmark on one node

This benchmark uses the actual Megatron direct exporter and GPU-delta publication
protocol on eight GPUs: TP1/PP1/CP1/EP8/ETP1, native GLM-5.2 five-layer model
(three dense layers and two routed MoE layers), NVFP4 TE 4over6 quantization.
It loads the real model and writes three cumulative immutable publications.
It does not launch a receiver or run forward/backward, an optimizer, or activation.

- **Codec:** Snappy-Zstd is the sole supported codec. Matrices use GPU XOR,
  GPU Snappy, then owner-wide GPU Zstd. Both
  compression algorithms execute on GPU SMs. Hardware Snappy acceleration is a
  receiver decompression property, not GPU compression acceleration.
- **Ownership:** routed experts remain on their EP × EDP exporter owners before
  expert gathering; this benchmark has EDP1. Global rank 0 owns nonrouted
  tensors in this PP1 proxy. Owner-local GPU-delta consumers skip expert
  gathers uniformly; ordinary export keeps its existing gather path.
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
  single-codec benchmark does not prove target equality with a historical run.

## Pipeline and memory

Export stages the complete new owner-local snapshot into pinned CPU memory.
One caller-to-staging stream dependency covers each exporter bucket; every
source allocation retains its stream lease through its asynchronous D2H copy.
Canonical scalars/vectors bypass XOR and both compressors; an owner CPU worker
compares them and writes complete target values only when changed. Classification
is fixed during baseline setup, outside the update hot loop.

After export D2H completes, name-sorted matrix batches use the existing
`update_weight_buffer_size` target (512 MiB here). Larger individual tensors remain
whole. Upload old/new pinned snapshots, compute XOR/counts and compress all
independent 1 MiB Snappy frames in one nvCOMP submission per batch. XOR/counting
uses disjoint 64 KiB tiles per frame and reduces only their small count array.
Only compact
aligned Snappy arenas remain in HBM as batches finish. Compress all owner-local
outer Zstd chunks together, read sizes/status once, pack final bytes and perform
one pinned D2H. CPU workers hash/write the final outer payload; no intermediate
Snappy D2H or large Snappy host slab is needed. Every owner drains before sealing.
The old canonical snapshot stays unchanged until acknowledgment.

For C matrix bytes and R scalar/vector bytes, this transfers C+R new bytes D2H,
C old plus C new bytes H2D, and final outer payload D2H. Raw values never enter
GPU compression. The full owner-local compact Snappy payload stays in HBM until
outer encoding completes, alongside bounded canonical scratch and codec workspaces.
There is no OOM fallback. No receiver GPU Zstd path is introduced: the receiver
CPU-decodes into final pinned Snappy buffers, then streams per-tensor H2D and
hardware Snappy decode/apply during the normal safe pause.

## Run

Use a matching explicit Miles CUDA13 image, paired feature checkout, prebuilt
nvCOMP >=5.3, compatible FlashInfer and TransformerEngine, and the prepared
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
  --perturb-fraction 0.001 --perturb-relative-scale 0.03125
```

The `GPU_DELTA_*` environment variables are development/debug controls, not a
stable user-facing configuration API. Output must be a new directory.
The harness records runtime package versions,
model flags, source digest (from `GPU_DELTA_SOURCE_DIGEST` when provided),
GPU memory counters, original rank ownership and all per-version measurements.
`--timing` enables CUDA phase events; default timing is off to avoid event overhead.
There are no codec/encoder/outer arm selectors. Earlier comparison artifacts remain
historical controls, not executable alternate production paths.

## Timing interpretation

| Field | Scope |
|---|---|
| `producer_blocked_s` | Update setup through export, encoding tail, immutable publication and final completion fence; excludes pre-update barrier, perturbation and inventory check. |
| `export_loop_s` | Actual conversion/quantization/collectives and new snapshot D2H staging. |
| `encoding_tail_s` | Remaining export D2H, bulk GPU encoding, owner hash/write tails and collective agreement. |
| `seal_and_visibility_s` | Owner shard sealing, metadata gather and final manifest publication. |
| `publication.manifest_seal_s` | Root manifest validation, serialization, hash and exclusive publication, nested in seal/visibility. |
| `conversion_host_s` | Host time advancing the real conversion iterator, nested in export. |
| `conversion_cuda_ms` | Optional same-stream conversion events, read after final fence; includes dispatch gaps and intervening work. |
| `publication.producer_metrics` | Per-owner nested encoding/wait/raw/write phases and source-accounted copy bytes. |
| `gpu_memory_bytes` | Before/after/peak PyTorch allocated/reserved bytes over the isolated update. |
| `sizes` | Changed canonical bytes, raw bytes, inner Snappy/outer Zstd/payload/manifest sizes. |

The protocol also retains rank-local `publication_metrics["metadata_gather_s"]`
after the existing gather completes. It measures metadata serialization/transport
and rank wait, not payload transfer, and is too late for that same gather's owner
prefix. External diagnostic observers can collect it outside the measured span;
the production path adds no collective just for this clock.

Report all rank ranges/medians plus rank0 separately. V1 includes first-use
allocation/compilation; V2/V3 are warm cumulative versions, not independent
fixed-target trials. Discovery and baseline export warm the exporter before
measurement and are reported separately. No concurrent trainer workload exists,
so this cannot establish training throughput or realized overlap.

Nonoverlapping caller phases compose blocked time. Do not add nested conversion,
compression, transfer or worker spans, or subtract monotonic timestamps across
ranks. CUDA events can perturb timing and do not measure GPU idle time. Logical
copy-byte counters are not bus measurements. `resident_snappy_hbm_bytes` records
unique retained inner storage at outer entry, not allocator peak/workspace totals.
The existing pre-update fence brackets peak-stat reset; reset does not empty the
CUDA cache, and reserved memory can include earlier work. No host RSS/pinned-peak
or untracked native allocation measurement is claimed.

Compare the [full-model receiver benchmark](README.md) separately. Its input
fixture and timings differ from this five-layer producer workload; do not subtract
one from the other to claim end-to-end RL savings.

## Full-model and multi-node scope

The five-layer TP1/PP1/CP1/EP8/ETP1 run is a correctness and profiling proxy;
it does not validate full-model capacity, multi-node performance or training
throughput. GPU delta reuses Megatron export ownership: ETP1 experts are consumed
on their EP × EDP owners, ETP>1 uses the existing gather-before-convert sender path,
and non-routed senders are PP-stage-local. The global canonical inventory still
requires complete, unique ownership; duplicate exports, including overlapping
PP/MTP names, are rejected rather than deduplicated.

`update_weight_buffer_size` is a matrix input-batch target, not a total memory
cap. Budget the pinned old/pending snapshots, the full owner-local compact Snappy
payload and codec scratch/workspaces on the intended topology. Keep the exact
inventory and reconstructed-target checks when extending this benchmark; the
measured proxy's owner imbalance or compression ratio is not a sizing rule.
