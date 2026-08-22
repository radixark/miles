---
title: Ray and Kubernetes Backend
description: Run the same training script on Ray, or let Kubernetes schedule every worker of the run.
---
Miles runs a job on one of two cluster backends. Ray is the default and needs no configuration.
Kubernetes installs the run as a helm release, so the cluster schedules every worker. The training
script is the same either way.

<Warning>

**Status.** Under active development: flags, chart values and failure semantics still change. Ray
also runs *inside* Kubernetes, and that is the well-trodden path — take this one only if you want
Miles to create the cluster's objects itself.

</Warning>

## Launch

Run everything from the repository root, with `kubectl` and `helm` on your PATH.

Install a workbench — the long-lived pod you launch from — once per namespace, from your
cluster's [`infra.yaml`](#for-cluster-administrator):

```bash
export MILES_NS="miles-$USER"
python -m miles.utils.external_utils.miles_workbench install -n "$MILES_NS" -f infra.yaml
```

It creates the namespace if missing, checks your rights, installs, and waits for Ready. Runs
launched from the pod inherit those values.

Launch from inside it:

```bash
python -m miles.utils.external_utils.miles_workbench exec -n "$MILES_NS" -- \
  bash -lc "cd /root/miles && python scripts/run_qwen3_4b.py train"
```

## Observability

**Built in**

- The launcher follows every pod's logs, prints status changes and warning events, and ends with
  the run's exit code.
- On a failure, collect the namespace's logs, describes and events into one directory — before
  cleaning up, and the whole namespace, because the explanation is usually the pod next to it:
  `python -m miles.utils.external_utils.miles_workbench collect-diagnosis -n "$MILES_NS" --output-dir ~/artifacts/miles`

**External**

- A run's pods are ordinary pods, so whatever the cluster already runs — a metrics stack, a log
  collector, the platform's own dashboards — sees them with no wiring from Miles.
- Prefer it at scale. The built-in following is meant for watching one run, not hundreds of pods.

## Clean up

```bash
python -m miles.utils.external_utils.miles_workbench stop -n "$MILES_NS" 260811-143000-042
python -m miles.utils.external_utils.miles_workbench uninstall -n "$MILES_NS"
```

`stop` removes the run and frees its GPUs; `uninstall` removes the workbench.

## Folder convention

A run is many pods on many machines, and they share nothing but the volumes `infra.yaml` mounts.
A path that is on none of them is the most common way a run fails.

- Every path your script names — `/root/models`, `/root/datasets` — has to be under one of the
  mounts, and so does `infra.paths.runsRoot`, where the launcher keeps each run's directory.
- Copying a file into a pod is pointless: pods come and go, the mount survives.
- To run your own branch instead of the image's copy, mount it over the path the image imports
  from — `/root/miles`, `/root/Megatron-LM`, `/sgl-workspace/sglang`. All three are editable
  installs, so a copy mounted anywhere else is silently never imported.
- Static workers run from `/root/miles`, so repository-local modules such as custom generate
  functions under `examples/` resolve from the same Miles checkout mounted at that path.

## For cluster administrator

Everything above assumes this was done once.

**Install LWS.** Miles deploys its worker pools as
[LeaderWorkerSets](https://github.com/kubernetes-sigs/lws). Install the CRDs and controller, and
grant users rights over them explicitly: LWS ships no aggregation labels, so a namespace `admin`
role does not include them.

**Give each user a namespace.** The namespace is the real boundary, not the Role: anything that
may create workloads can name another ServiceAccount and read its token. Keep privileged accounts
out of it.

The run chart gives pod observers a dedicated ServiceAccount that can only get, list, and watch
pods. The orchestrator keeps its separate pod-deletion and optional uninstall-job permissions;
worker processes that only report pod state do not inherit them.

**Write one `infra.yaml`.** The same file drives every Miles chart:

```yaml
infra:
  image:
    repository: radixark/miles
    tag: dev
  # TODO: describe a volume and its mounts here.
  volumes:
    - name: cluster-storage
      hostPath: {path: /cluster-storage, type: Directory}
      mounts:
        - {mountPath: /cluster-storage}
        - {mountPath: /root/miles, subPath: alice/miles}
  paths:
    runsRoot: /cluster-storage/miles_data
```

`charts/miles-run/values.yaml` shows the full shape, and each chart's `values.schema.json` is the
authoritative field list.
