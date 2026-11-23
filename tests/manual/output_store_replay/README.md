# Output store replay profile

A manual A/B of how trainer ranks consume forward replay rows (routed experts, indexer top-k) that rollouts wrote to the output store. No CI job runs it. Use it for a change to how those rows are written, read or filled: run the change's base and its head back to back on one machine and compare.

## Workloads

| | `indexer` | `r3` |
|---|---|---|
| Model | GLM-5 4-layer | GLM-4.7-Flash |
| Replay | `--use-rollout-indexer-replay` | `--use-rollout-routing-replay` |
| Trainer | TP2 PP2 EP2 CP1, colocated on 4 GPUs | TP1 CP4 PP1 EP4 on 4 GPUs, 4 more for FP8 engines |
| Batch per step | 64 prompts x 4 samples, up to 4096 tokens | 32 prompts x 16 samples, up to 16384 tokens |
| Rows per step | about 35 GB per trainer rank | about 3.7 GB per DP shard |
| Shows | time and host memory: the fill is a large part of a step | bytes per CP rank and correctness under CP: the fill takes about a second, so time does not move |

A change that only matters with PP > 1 or CP > 1 needs the workload that has it. The sizes assume 8 H200s and about 1 TB of host memory; `launch.py` holds every flag.

## Run

From the repository root, once per workload and then once per tree:

```bash
python -m tests.manual.output_store_replay.launch indexer prepare
python tests/manual/output_store_replay/sample_memory.py mem-base.jsonl &
python -m tests.manual.output_store_replay.launch indexer execute 2>&1 | tee base.log
kill %1
python tests/manual/output_store_replay/step_metrics.py base.log
```

- Keep the image, the SGLang tree and the machine the same for both trees, with nothing else on the GPUs or the host.
- A run is 3 steps (`PROFILE_NUM_ROLLOUT`). Step 0 warms up; compare steps 1 and 2.
- Run each tree at least twice, alternating. A difference counts when the two trees' ranges over those steps do not overlap.
- `MOONCAKE_PROTOCOL=rdma MOONCAKE_DEVICE=<device>` moves the store off TCP.

## Compare

| Metric | Source | Reading |
|---|---|---|
| `perf/step_time` | log | the headline |
| `perf/rollout_time` | log | generation does not depend on the trainer's read path: its spread is the noise of the pair |
| `perf/data_preprocess_time`, `perf/replay_fill_time` | log | where the rows are read and registered |
| `perf/actor_train_time`, `perf/train_time` | log | must not grow by what the fill saved |
| Trainer rank `VmHWM`, cgroup anon + shmem, GPU used | `sample_memory.py` | peaks; `nvidia-smi` samples are a lower bound |
| `train/train_rollout_logprob_abs_diff`, `Replay check` lines | log | stay in the base's range |

`replay_fill` is its own timer. It runs before `train` when every micro batch is filled up front and inside it when a micro batch is filled as the forward reaches it, so read `perf/step_time` first.

## Where the time goes

```bash
py-spy record --nonblocking --rate 50 --format raw --duration 300 --pid <trainer rank pid> --output rank0.txt
python tests/manual/output_store_replay/pyspy_top.py rank0.txt --under attach_replay_data
```

Sample a live run with `--nonblocking` only. `sample_memory.py` writes the trainer rank pids; `--under` narrows the summary to one call path, and `--gil --threads` on `py-spy record` shows which thread holds the GIL when a fill overlaps the forward.
