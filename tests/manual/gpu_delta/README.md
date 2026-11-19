# GPU-delta development guide

Manual checks and the end-to-end benchmark for `--update-weight-transfer-mode gpu-delta`.
[GPU-Delta Weight Transfer](../../../docs/advanced/gpu-delta-weight-transfer.md) is the user guide; the
[receiver contract](../../../miles/backends/sglang_utils/gpu_delta/README.md) owns receiver behavior.

## Environment

The benchmark needs eight Blackwell GPUs, a CUDA 13 Miles development image, which installs nvCOMP
(`docker/Dockerfile`), and the paired SGLang branch on `PYTHONPATH`. Manual tests also need
`python-snappy`, `lz4` and `zstandard` as CPU oracles.

These `GPU_DELTA_*` variables are development/debug controls, not a stable configuration API:

| Development variable | Meaning |
| --- | --- |
| `GPU_DELTA_CODEC=bitcomp-zstd` | Trainer codec for ordinary learned updates, one of the [user guide's codecs](../../../docs/advanced/gpu-delta-weight-transfer.md#usage). Read once at launch; receivers take the codec from each authenticated publication. |
| `GPU_DELTA_INITIAL_SYNC_CODEC=bitcomp-zstd` | Trainer codec for initial sync, engine recovery, checkpoint companions and eval snapshots. |
| `GPU_DELTA_SKIP_PAYLOAD_HASH=0` | `1` makes the trainer write publications without owner-file SHA256 (`payload_checksum_format="none"`), which [requesters follow](../../../docs/advanced/gpu-delta-weight-transfer.md#activation-and-failures). A trainer-only variable: launch validation and each trainer reject values other than `0` and `1`. See the receiver's [payload verification](../../../miles/backends/sglang_utils/gpu_delta/README.md#configuration-and-observations). |

Receiver variables are in the receiver's
[configuration table](../../../miles/backends/sglang_utils/gpu_delta/README.md#configuration-and-observations).
Ray workers do not inherit the submitting shell: use `runtime_env`,
`execute_train(extra_env_vars=...)` or, for trainer-only variables, `--train-env-vars`.

## Five-layer E2E

`tests/e2e/megatron/test_glm5_2_744b_a40b_5layer_nvfp4_w4a16.py` serves the primary recipe on two TP2
CuTe DSL engines (DCP2 with a replicated Q projection) after the required initial sync, for five
deterministic-reward rollouts with checkpoint companions. It fails if every learned publication is a no-op
and enforces the warm-update limits stated in the file; receiver recovery is not covered. It takes no
options:

```bash
python tests/e2e/megatron/test_glm5_2_744b_a40b_5layer_nvfp4_w4a16.py
```

## Manual checks

CPU mocks do not prove CUDA/native behavior. Each test's docstring states its hardware and package
prerequisites; without CUDA the layout and apply tests skip, and any other missing prerequisite fails.
`nvcomp_oracle.py` round-trips native nvCOMP compression through independent CPU decoders and checks
asynchronous submission; the encoder test applies its `decode()` oracle. From the Miles checkout, run
each command in a fresh process:

```bash
python tests/manual/gpu_delta/nvcomp_oracle.py
for test in test_gpu_delta_encoder test_gpu_delta_layout_cuda test_host_cuda test_apply_cuda; do
  python -m pytest -q "tests/manual/gpu_delta/$test.py"
done
# DE allocation and decoding, once per ambient allocator mode.
for conf in backend:native,expandable_segments:True backend:cudaMallocAsync,expandable_segments:False; do
  PYTORCH_ALLOC_CONF="$conf" python -m pytest -q \
    tests/manual/gpu_delta/test_device_allocator_cuda.py tests/manual/gpu_delta/test_codec.py
done
```

See [NVIDIA's DE requirements](https://docs.nvidia.com/cuda/nvcomp/decompression_engine_faq.html).

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

`--reuse-fixtures RUN` skips encoding and replays a completed fresh-encoding run's sealed publications
on fresh receivers; fresh encoding remains the mode for sender performance:

```bash
python tests/manual/gpu_delta/bench_gpu_delta.py --model /models/GLM5.2-NVFP4 \
  --prepared-inputs /data/gpu-delta/prepared-inputs --output /data/gpu-delta/new-replay \
  --reuse-fixtures /data/gpu-delta/original-run --reuse-producer-spec-sha256 REVIEWED_SHA256 \
  --reuse-producer-spec /data/gpu-delta/original-run/producer-spec.json --codecs bitcomp-zstd --hashing on
```

Sealed publications must include the base stream and the content-bound checkpoint digest. Regenerate
older fixtures once from their prepared inputs before using receiver-only replay.

Every fresh-encoding run writes `producer-spec.json` with its argv and the SHA256 of the prepared
inputs' `READY.json`. Before any receiver starts, replay requires the spec's reviewed SHA256, that the
spec names RUN and these prepared inputs, and that each arm's fixture covers versions 1 through
`--versions` at this `--frame-bytes` with payload hashing on. Receivers then authenticate every
manifest and payload file. Nothing is copied: `arms/ARM/fixture-reuse.json` references the fixture and
the spec, and the reused sender timings are reported as `historical_saved_fixture`.

Add `--nsys` to a one-arm replay without `--validate-final-target` to capture warm update 2.
`--profile-version N` selects a different warm update; it does not enable profiling by itself.
Nsight Systems must be on `PATH`, with permission for system-wide CPU sampling. The launcher enables
detailed NVTX, CPU/context-switch and Python/GIL sampling, CUDA and OS-runtime tracing. Its capture
range opens 0.2 s before the selected update and closes after every engine's resume reply; the reported
update time excludes the lead-in. Capture stops without terminating the benchmark, which finishes
the remaining updates and cleanup. Keep profiled timings separate from ordinary measurements:

```bash
python tests/manual/gpu_delta/bench_gpu_delta.py --model /models/GLM5.2-NVFP4 \
  --prepared-inputs /data/gpu-delta/prepared-inputs --output /data/gpu-delta/new-profile \
  --reuse-fixtures /data/gpu-delta/original-run --reuse-producer-spec-sha256 REVIEWED_SHA256 \
  --reuse-producer-spec /data/gpu-delta/original-run/producer-spec.json \
  --codecs bitcomp-zstd --hashing on --frame-bytes 1048576 --nsys
```

The output includes `receiver-profile.nsys-rep`, the exact command in `nsys-command.json`, and
`REPORT.md` with the captured update marked profiled.

### Timing scopes and limits

Timing scopes overlap and must not be added. Generation checks selected outputs, not every
weight byte, and without serving traffic or training the workload measures neither
serving interference nor RL throughput.

- **Sender:** owner maxima of compression/publication wall: encoding, finalization,
  optional hashing, writes and fsync, excluding input reads and manifest sealing.
- **Receiver:** background preparation, CPU outer decode, matrix-application host wall and
  scheduler pause; GPU intervals come only from Nsight profiles.
- **Coordinator:** the complete prepare/apply/resume RPC lifetime.
