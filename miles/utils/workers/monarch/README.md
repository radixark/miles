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
- **Data plane.** There is no Ray object store, so `--object-store-backend` is forced to `mooncake`, as
  under Kubernetes; the launcher starts the Mooncake master.
- **Hosts.** A run spawns everything on the driver's host.
- **Not available.** The dashboard, `--use-distributed-post` and multi-LoRA need a Ray cluster and are
  refused; the Prometheus collector, a Ray actor too, is skipped with a warning, as under Kubernetes; hot
  restart is Kubernetes-only.
