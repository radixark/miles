---
title: Google TPUs
description: Run Miles on Google Cloud TPU v7x through torch_tpu, on one host with sglang-torchtpu or on KubeRay with vllm-torchtpu.
---
Miles runs on Google Cloud TPU v7x through [torch_tpu](https://github.com/google-pytorch/torch_tpu).
Two layouts are supported:

- **One host, shared slice**: the torchtitan backend trains and
  [sglang-torchtpu](https://github.com/sgl-project/sglang-torchtpu) serves rollouts as an external
  engine on the same 2x2x1 host; weights move over ICI.
- **KubeRay on GKE**: the FSDP (or torchtitan) backend trains and
  [vllm-torchtpu](https://github.com/vllm-project/vllm-torchtpu) serves rollouts in separate Ray
  placement groups; weights move host to host with tpu-sync. See
  [KubeRay on GKE (TPU7x)](#kuberay-on-gke-tpu7x).

Megatron-LM is not supported on TPU.

## Layout

One v7x 2x2x1 host has 4 chips with 2 TensorCores each, so 8 devices. The trainer runs as 4 ranks
on devices 0-3 and the engine serves on devices 4-7. Both jobs share the host's slice, so weights
move from trainer to engine over ICI (`--update-weight-transfer-mode ici`) without touching disk.

## Image

The engine and the trainer share one Python environment, built by `docker/Dockerfile.tpu` on
sglang-torchtpu's release image. CI rebuilds it next to the CUDA image and publishes it as
`radixark/miles:dev-tpu` (and `latest-tpu`). To build it yourself:

```bash
python docker/build.py --variant tpu --image-tag dev    # radixark/miles:dev-tpu
```

To set the environment up by hand instead, see
[`scripts/tpu/README.md`](https://github.com/radixark/miles/blob/main/scripts/tpu/README.md#environment).

## Launch training

The TPU launchers live under `scripts/tpu/`, one per model (Qwen3-4B, Qwen3-30B-A3B, and a 4-layer
GLM-4.7-Flash). Each launcher has a `prepare`, a `sglang` and a `train` command. `prepare` fetches the model and
datasets. Start the engine in its own shell, then start training a minute or two later without waiting for the
engine's `/health`: the engine compiles only after the trainer has joined the slice.

```bash
python3 scripts/tpu/run_qwen3_4b_torchtitan.py prepare
python3 scripts/tpu/run_qwen3_4b_torchtitan.py sglang               # devices 4-7
python3 scripts/tpu/run_qwen3_4b_torchtitan.py train --recipe full  # devices 0-3
```

The first step of a cold run compiles the trainer's programs, and sglang has to be restarted with
every training run. The
[README](https://github.com/radixark/miles/blob/main/scripts/tpu/README.md) covers both, along with
the recipes, the compile cache, and per-model trainer settings.

## KubeRay on GKE (TPU7x)

[`scripts/tpu/run_qwen3_0_6b_kuberay.py`](https://github.com/radixark/miles/blob/main/scripts/tpu/run_qwen3_0_6b_kuberay.py)
runs GRPO on Qwen3-0.6B against an existing KubeRay `RayCluster`: Miles launches the vLLM engine
itself (`--rollout-engine vllm`), places the trainer and the engine in separate placement groups,
and syncs weights with `--update-weight-transfer-mode tpu-sync`. Its docstring lists the flags.

### Cluster shape

- **Head**: a CPU node. It runs the driver (`train.py`), the router and the rollout controller, and
  never opens a TPU. `MILES_ACCELERATOR=tpu` is safe there: torch_tpu imports without a device.
- **TPU workers**: one Ray worker per v7x 2x2x1 host (GKE node selectors
  `cloud.google.com/gke-tpu-accelerator: tpu7x` and `cloud.google.com/gke-tpu-topology: 2x2x1`,
  `google.com/tpu: 4`). Each chip exposes two TensorCores, so start Ray with
  `--resources='{"TPU": 8}'`. KubeRay labels each host's worker group with a `tpu-group-<n>` resource;
  Miles reads those labels to pin each placement group to one slice.
- **Shared storage**: a volume every pod mounts at the same path (for example GCS FUSE at `/data`)
  for the checkout, the model and the datasets.
- **Image**: torch_tpu, vllm-torchtpu, tpu-sync and Miles' Python dependencies (`requirements.txt`).
  The FSDP backend runs on the torch 2.13 images vllm-torchtpu ships; the torchtitan backend needs
  the environment in [`scripts/tpu/README.md`](https://github.com/radixark/miles/blob/main/scripts/tpu/README.md#environment).

### Layouts

The launcher asks for 4 trainer devices and one TP 4 engine. Placement picks the layout from the
live slices:

| Live 2x2x1 slices | Trainer | Engine |
|---|---|---|
| Two or more | devices 0-3 of one slice | devices 0-3 of another |
| One | devices 0-3 (sub-host offset 0) | devices 4-7 (sub-host offset 4) |

On one slice the two jobs run as independent 4-device torch_tpu sessions
(`TORCH_TPU_TOPOLOGY=1,2,1,2`, `ALLOW_MULTIPLE_LIBTPU_LOAD=1`), each on its own block of local
ports. tpu-sync moves weights over TCP in both layouts, so neither needs the job-local-world
torch_tpu build that `ici` does.

### Submit a run

Run the launcher inside the head pod, against the cluster's own Ray: it submits `train.py` by its
path in the checkout, so the checkout has to sit on the shared volume at the same path on every pod.
Install the Python dependencies the image lacks on the same volume and put them on `PYTHONPATH`,
which the launcher forwards to every Ray worker (on `ray-verl-tpu` images that is `polars`, which
needs its `polars-runtime-32` binary next to it):

```bash
HEAD=$(kubectl get pod -l ray.io/cluster=<cluster>,ray.io/node-type=head -o name)
# stage the checkout on the shared volume
kubectl cp . "${HEAD#pod/}":/data/tmp/miles -c ray-head
kubectl exec -it "$HEAD" -c ray-head -- bash -c '
  pip install --target /data/tmp/miles-pydeps --no-deps polars==1.42.1 polars-runtime-32==1.42.1 &&
  cd /data/tmp/miles &&
  PYTHONPATH=/data/tmp/miles-pydeps:$PYTHONPATH MILES_SCRIPT_EXTERNAL_RAY=1 \
    python3 scripts/tpu/run_qwen3_0_6b_kuberay.py train --recipe smoke --compile-cache-dir /data/tmp/miles-cache'
```

`MILES_SCRIPT_EXTERNAL_RAY=1` is required: without it the launcher stops Ray and starts its own,
which on the head pod takes the cluster down. The launcher refuses to run in a KubeRay pod
without it. Follow a run in the Ray dashboard (`kubectl port-forward svc/<cluster>-head-svc 8265`).

To submit `train.py` from a workstation instead, skip the launcher and pass the checkout as the
job's working directory through a port-forward, with the same flags the launcher builds (its
recorded command is in `tests/snapshots/launch_scripts/py/scripts/tpu/run_qwen3_0_6b_kuberay.py/train.txt`).
The dependency overlay then goes in the runtime environment:

```bash
kubectl port-forward svc/<cluster>-head-svc 28265:8265 &
# an exported RAY_ADDRESS wins over --address, so set it for this command
RAY_ADDRESS=http://127.0.0.1:28265 ray job submit --working-dir . \
  --runtime-env-json='{"env_vars": {"MILES_ACCELERATOR": "tpu", "LIBTPU_INIT_ARGS": "--xla_tpu_bf16_emission_mode=PROMOTE_F32"},
    "pip": {"packages": ["polars==1.42.1", "polars-runtime-32==1.42.1"], "pip_check": false,
            "pip_install_options": ["--no-deps"]}}' \
  -- python3 train.py <flags from the recorded command>
```

A job submitted with a `ray` CLI newer than the cluster's has been seen to work, but keep the two
versions close.

`prepare` downloads `zhuzilin/gsm8k`, whose `messages` and `label` columns the recipe reads.
veRL-style gsm8k parquet files keep the answer in a nested `reward_model.ground_truth`, which
`--label-key` cannot address.

### What to expect

Measured with the smoke recipe on one 2x2x1 slice (4+4 sub-host), Qwen3-0.6B:

| Stage | Time |
|---|---|
| vLLM engine to `/health`, cold | ~370 s |
| vLLM engine to `/health`, `--compile-cache-dir` on `/data` warm | ~240 s |
| tpu-sync weight update, first step (receivers register) | ~0.8 s |
| tpu-sync weight update, later steps | ~0.2 s |
| Whole smoke run (3 rollouts) | ~7 min |

The first training step also compiles the trainer's programs. `--compile-cache-dir` keeps the trainer's
and the engine's compiled programs on the shared volume, one directory per role and slot, so they
survive a preempted node; use a fresh directory for each torch_tpu or vLLM build.

### Caveats

- **Spot preemption**: a spot TPU node that is reclaimed takes its slice with it, and the run
  fails. Spot 2x2x1 nodes have been observed to live only 2-8 minutes when capacity is short, less
  than a cold start plus a run; keep `--compile-cache-dir` on the shared volume so each attempt
  starts warmer, or use reserved capacity. A stocked-out zone can leave a worker group pending; a
  single live slice still runs the 4+4 layout.
- **`LIBTPU_INIT_ARGS`**: the launcher sets `--xla_tpu_bf16_emission_mode=PROMOTE_F32`. Without it
  v7x evaluates bf16 SiLU and sigmoid in bf16, and the engine's logprobs drift from the trainer's.
  An exported `LIBTPU_INIT_ARGS` replaces the launcher's set, so keep that flag in it.
- **Pod-wide TPU environment**: GKE injects whole-host variables (`TPU_WORKER_HOSTNAMES`,
  `TPU_PROCESS_ADDRESSES`, `MEGASCALE_*`, `TORCH_TPU_DEVICES_PER_HOST`, ...) that break a 4-device
  sub-host session. Miles removes them in each trainer and engine process before torch_tpu
  initializes; Ray's `runtime_env` cannot unset variables, so do not try to clear them there.
- **Stopping a run**: stop it with `ray job stop`, not `kill -9`. A killed TPU process can leave
  the host's VFIO devices busy for a few seconds, and a run started right away fails to open them.
- **Fractional TPU requests**: Ray rejects TPU quantities other than 1, 2, 4 or 8, so a custom
  provider or spec must not ask for a fraction of a TPU.

## Next steps

- [`scripts/tpu/README.md`](https://github.com/radixark/miles/blob/main/scripts/tpu/README.md) — the full recipe and operating notes.
- [Docker build](/developer/ci/02-docker-build) — the TPU Dockerfile, variants, and tags.
