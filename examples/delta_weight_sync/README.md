# Delta Weight Sync

Delta weight sync sends changed weight positions and values instead of every
parameter during non-colocated actor-to-rollout updates.

Disk transport is intended for trainer/rollout disaggregation over a shared
filesystem:

```bash
DELTA_ARGS=(
  --update-weight-mode delta
  --update-weight-transport disk
  --update-weight-encoding deltas_zstd
  --update-weight-delta-dir /shared/fs/delta-updates
)
```

NCCL transport is useful inside one cluster:

```bash
DELTA_ARGS=(
  --update-weight-mode delta
  --update-weight-transport nccl
  --update-weight-encoding indices
)
```

Receiver-side SGLang options:

```bash
SGLANG_ARGS=(
  --sglang-update-weight-delta-chunk-bytes $((2 * 1024 * 1024 * 1024))
  --sglang-update-weight-delta-read-workers 4
)
```

See [docs/advanced/delta-weight-sync.md](../../docs/advanced/delta-weight-sync.md)
for the protocol and constraints.
