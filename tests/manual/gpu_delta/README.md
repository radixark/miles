# GPU-delta development guide

Manual checks and the end-to-end benchmark for `--update-weight-transfer-mode gpu-delta`.
[GPU-Delta Weight Transfer](../../../docs/advanced/gpu-delta-weight-transfer.md) describes
the trainer pipeline, recovery, eval snapshots and `perf/gpu_delta/*` metrics; the
[receiver contract](../../../miles/backends/sglang_utils/gpu_delta/README.md) owns layout
admission, control APIs, cache lifecycle, decoder scheduling, receipts and receiver tests.

## Environment

GPU-delta's CUDA decoder path, including the default Bitcomp-Zstd codec, does not require
Blackwell. The GLM5.2 NVFP4 W4A16 benchmark needs eight Blackwell GPUs, a CUDA 13 Miles
development image and the paired SGLang branch on `PYTHONPATH`; it uses CuTe DSL MoE, no
MoE A2A and static bundled MTP. Install nvCOMP without changing the image dependency
closure; manual tests also need `python-snappy`, `lz4` and `zstandard` as CPU oracles:

```bash
python -m pip install --no-deps nvidia-libnvcomp-cu13==6.0.0.12
```

These `GPU_DELTA_*` variables are development/debug controls, not a stable configuration
API; runtime defaults suffice for normal use.

| Development variable | Meaning |
| --- | --- |
| `GPU_DELTA_CODEC=bitcomp-zstd` | Trainer codec for ordinary learned updates: `snappy-zstd`, `lz4-zstd`, `lz4`, `bitcomp-zstd` or `bitcomp`. `*-zstd` adds GPU Zstd wrapping; `lz4` and `bitcomp` skip it. Read once at launch; the receiver selects the codec from each authenticated publication. |
| `GPU_DELTA_INITIAL_SYNC_CODEC=bitcomp-zstd` | Trainer codec for initial sync, engine recovery, checkpoint companions and eval snapshots; accepts the same codecs. Initial synchronization is mandatory. |
| `GPU_DELTA_SKIP_PAYLOAD_HASH=1` | Miles-only opt-in: omit owner-file SHA256 generation and send `validate_payload_hash=false` with each receiver update. Accepts `0` (default) or `1`; any other value raises. Such publications declare `payload_checksum_format="none"` and null file hashes; SGLang accepts them only with this explicit request option and does not read the variable. Manifest SHA256 and native decoder checks remain mandatory. |

Receiver variables are in the receiver's
[configuration table](../../../miles/backends/sglang_utils/gpu_delta/README.md#configuration-and-observations).
Ray workers do not inherit the submitting shell: use `runtime_env`,
`execute_train(extra_env_vars=...)` or, for trainer-only variables, `--train-env-vars`.

## Five-layer E2E

The dedicated test performs the required initial sync, then five rollouts with
deterministic rewards. Four GPUs train; two TP2/DP1/EP1 engines shard attention, routed
experts and BF16 shared experts over TP2, and DCP2 uses TRT-LLM DSA with a replicated Q
projection, covering its derived Q/K weights (prefill CP has unit coverage only). The test
keeps the default codecs, forwards the hash control, checks every manifest's phase codec
and excludes the startup publication from its change gate. Warm updates on every TP rank
must reuse the prepared Triton kernels, keep each rank's scheduler pause under 150 ms and
each engine's all-TP apply/resume under 200 ms, and keep each rank's preparation and every
matrix-producing owner's encoding stages under 1.5 s. These limits are about 3x the worst
recent B300 warm maxima, leaving headroom for cluster CPU variance. Publications go to
`/root/delta-updates/<run_id>/gpu_delta` and checkpoints carry delta companions. Without
fault tolerance or engine restarts, receiver recovery is not covered, and attention FP8
conversion stays disabled. The test takes no CLI options:

```bash
python tests/e2e/megatron/test_glm5_2_744b_a40b_5layer_nvfp4_w4a16.py
```

## Manual sender checks

```bash
python tests/manual/gpu_delta/nvcomp_oracle.py
python -m pytest -q tests/manual/gpu_delta/test_gpu_delta_encoder.py
```

The qualification script (not a pytest module) round-trips native nvCOMP Zstd, Snappy, LZ4
and Bitcomp compression through independent CPU decoders and checks asynchronous
submission; the encoder test applies its `decode()` oracle to producer and recovery
payloads. Receiver tests are listed in the receiver contract's Validation section.

## End-to-end benchmark from an HF base

The benchmark needs only an existing NVFP4 serving checkpoint with bundled MTP, which it
never modifies:

```bash
export PYTHONPATH=/workspace/sglang/python:/workspace/miles
python tests/manual/gpu_delta/bench_gpu_delta.py --model /models/GLM5.2-NVFP4 \
  --output /data/gpu-delta/new-run --prepared-inputs /data/gpu-delta/prepared-inputs
```

- `--output` must be new. `--prepared-inputs` (default `OUTPUT/inputs`) builds or reuses
  the inventory and cumulative versions; three versions take two canonical copies plus one
  full checkpoint. Codec, hashing, frame size and sender GPUs can change without a rebuild.
- Defaults: two serial Bitcomp-Zstd arms with payload hashing on and off, 1 MiB frames,
  three versions and two TP4/attention-DP4/EP4 engines (one `--ports` value: one TP8
  engine), set by `--codecs`, `--hashing`, `--frame-bytes`, `--versions` and `--ports`.
  Versions are deterministic mantissa/packed-FP4 mutations calibrated by `--ratio`.
- Arms encode with the production `GpuBatchEncoder` and `PublicationWriter` over simple HF
  ownership on `--sender-gpus` (no Megatron export or distributed gather); fresh receivers
  then prepare, apply and resume each version. `--validate-final-target` compares final
  outputs with the target checkpoint; otherwise equivalence stays `NOT_VALIDATED`.
  `REPORT.md` has a row per arm and version, `summary.json` every owner and rank metric;
  a failure stops the run and keeps partial output without retrying.

### Saved-delta apply mode

`--reuse-fixtures` skips encoding and replays a completed fresh-encoding run's sealed
publications on fresh receivers; fresh encoding remains the mode for sender performance:

```bash
python tests/manual/gpu_delta/bench_gpu_delta.py --model /models/GLM5.2-NVFP4 \
  --prepared-inputs /data/gpu-delta/prepared-inputs --output /data/gpu-delta/new-replay \
  --reuse-fixtures /data/gpu-delta/original-run --reuse-producer-spec-sha256 REVIEWED_SHA256 \
  --reuse-producer-spec /data/gpu-delta/original-run/producer-spec.json \
  --codecs bitcomp-zstd --hashing on --frame-bytes 1048576 --versions 3 --sender-gpus 8
```

- **Provenance:** every fresh encoding run writes `producer-spec.json` with its `argv`, the
  input-cache `cache_files` hashes, `prepared_settings_sha256` and the producer source
  `guard_files` hashes; replay requires its reviewed SHA256. For an older run, a reviewed
  provenance adapter may supply the same four fields and link the original producer
  evidence. Package versions stay unknown unless recorded separately.
- **Admission before any receiver starts:** the original fixture phase must have passed with
  SHA-on payloads and identical inputs, settings, plan, codec, frames, version chain,
  mutations and owner partition. Every payload SHA256 and stable file identity, the sealed
  manifest SHA256, the canonical plan and every owner record are verified; symlinks, path
  escapes, partial seals, missing owners and an unfinished producer workflow fail. Fresh
  receivers still check each publication against their inventory and bindings.
- **Outputs:** nothing is copied. Each `arms/ARM/fixture-reuse.json` references the verified
  original fixture, its workflow phase reads `REUSED` / `SKIPPED_ENCODING`, and
  `fixture-verification.json` records the current verification wall time. Reused sender
  measurements are labeled `historical_saved_fixture`, never current sender performance.
  External capture or reduction scripts must resolve `fixture-reuse.json` and handle the
  `NOT_RUN` oracle status used without `--validate-final-target`.

For a receiver profile, add `--nsys` to the replay command with a new output directory. It
captures warm version 2 (`--profile-version N` selects another but does not enable
profiling) with CUDA, NVTX, OS-runtime, GIL, CPU and Python stack tracing, from preparation
through both resume replies plus a 0.2 s arming lead-in. The script owns the `nsys` launch
(do not wrap it; `nsys` must be on `PATH` with CPU profiling permitted), records it in
`nsys-command.json` and writes `receiver-profile.nsys-rep`. Keep profiled timings separate.

### Timing scopes and limits

Timing scopes overlap and must not be added. Generation checks selected outputs, not every
weight byte, and without serving traffic or training the workload measures neither
serving interference nor RL throughput.

- **Sender:** owner maxima of compression/publication wall: encoding, finalization,
  optional hashing, writes and fsync, excluding input reads and manifest sealing.
- **Receiver:** background preparation, CPU outer decode, matrix-application host wall and
  scheduler pause; GPU intervals come only from Nsight profiles.
- **Coordinator:** the complete prepare/apply/resume RPC lifetime.
