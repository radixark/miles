# Monarch cluster backend

**Status.** Experimental.

Install the optional dependency from the checkout with `pip install -e ".[monarch]"`, then pass `--cluster-backend monarch`:

```bash
MILES_SCRIPT_CLUSTER_BACKEND=monarch python scripts/run_qwen3_4b.py train
```

- **One process mesh per cell.** The driver spawns a worker manager actor, which spawns every cell as one
  Monarch proc mesh and watches its liveness: a cell that loses a process is torn down and reported as not
  alive, exactly as under Ray.
- **Gpu placement.** Monarch assigns no gpus, so Miles packs its gpu slots onto hosts in mesh order, as a
  Ray `PACK` placement group would, and every gpu worker selects the devices it was placed on.
- **Worker calls.** The driver calls its workers through Monarch endpoints, and a trainer cell's ranks in
  one mesh call; `--worker-comm-backend rpc` calls the http server every worker serves instead.
- **Data plane.** There is no Ray object store, so `--object-store-backend` is forced to `mooncake`, as
  under Kubernetes; the launcher starts the Mooncake master.
- **Hosts.** A single-node run spawns everything on the driver's host. A multi-node run is submitted to
  SLURM as one Monarch job (below).
- **Environment.** Every worker gets the variables the launcher exports to the driver (`PYTHONPATH`,
  `--extra-env-vars`, ...), on every host, as Ray's runtime env would pass them on.
- **Not available.** The dashboard, `--use-distributed-post` and multi-LoRA need a Ray cluster and are
  refused; the Prometheus collector, a Ray actor too, is skipped with a warning, as under Kubernetes; hot
  restart is Kubernetes-only.

## Multi-node on SLURM

Submit the launcher command from a login node; the job runs it unchanged on its first node:

```bash
python -m miles.utils.workers.monarch.slurm --nodes 2 --gpus-per-node 8 \
    --slurm-arg=--partition=gpu --slurm-arg=--time=4:00:00 -- \
    python scripts/run_qwen3_4b.py train --num-gpus-per-node 8
```

- **One allocation.** Monarch starts a worker loop on every node, the driver attaches to them as its
  hosts, and SLURM releases the nodes once the launcher command exits, or on `scancel <job id>`.
  `--dry-run` checks and prints the submission without submitting it.
- **Logs.** The job's output, the driver's included, goes to `slurm_<job id>_*.out` and `.err` in a
  directory of its own under `--log-dir`, which the submission prints.
- **Head node.** The driver, the worker manager and the Mooncake master run on the first node, and
  workers on the other nodes reach them by its address. Commands a launcher runs on every node,
  such as a multi-node checkpoint conversion, run there through `srun`.
- **Shared filesystem.** Nothing is copied to the nodes: the venv, the checkout, the run's paths and
  `--log-dir`, where the job's logs and saved state go, must resolve to the same files on every node.
- **Site settings.** Partition, account, time limit and the like go in `--slurm-arg`; the flags the
  submission sets itself (`--nodes`, `--gpus-per-node`, `--job-name`, ...) are refused there.
- **Whole nodes.** The job reserves its nodes whole. With `--no-exclusive` it shares them with other jobs,
  asks only for `--gpus-per-node` gpus, and needs the cpus and memory in `--slurm-arg`
  (`--slurm-arg=--cpus-per-task=32 --slurm-arg=--mem=200G`). Two jobs on one node also need their own
  worker loop port (`--monarch-port`); a run refuses to start when another job holds the Mooncake master's
  port on its head node, which a multi-node run cannot move.
- **Dual-stack networks.** When a node's hostname resolves to IPv6 on the node and to IPv4 on the driver's
  node, the driver never reaches the worker loops; `--bind-to ::` makes them listen on every address.
