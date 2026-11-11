---
title: Google TPUs
description: Run Miles on Google Cloud TPU v7x through torch_tpu, with the torchtitan trainer and sglang-torchtpu sharing one host.
---
Miles runs on Google Cloud TPU v7x through [torch_tpu](https://github.com/google-pytorch/torch_tpu).
The torchtitan backend trains and [sglang-torchtpu](https://github.com/sgl-project/sglang-torchtpu)
serves rollouts as an external engine. Megatron-LM is not supported on TPU.

## Layout

One v7x 2x2x1 host has 4 chips with 2 TensorCores each, so 8 devices. The trainer runs as 4 ranks
on devices 0-3 and the engine serves on devices 4-7. Both jobs share the host's slice, so weights
move from trainer to engine over ICI (`--update-weight-transfer-mode ici`) without touching disk.

## Image

The engine and the trainer share one Python environment, built by `docker/Dockerfile.tpu` on
sglang-torchtpu's release image:

```bash
python docker/build.py --variant tpu --image-tag dev    # radixark/miles:dev-tpu
```

To set the environment up by hand instead, see
[`scripts/tpu/README.md`](https://github.com/radixark/miles/blob/main/scripts/tpu/README.md#environment).

## Launch training

The TPU launchers live under `scripts/tpu/`, one per model (Qwen3-4B, Qwen3-30B-A3B, and a 4-layer
GLM-4.7-Flash). Each launcher has a `sglang` and a `train` command. Start the engine in its own shell, then start
training a minute or two later without waiting for the engine's `/health`: the engine compiles only
after the trainer has joined the slice.

```bash
python3 scripts/tpu/run_qwen3_4b_torchtitan.py sglang               # devices 4-7
python3 scripts/tpu/run_qwen3_4b_torchtitan.py train --recipe full  # devices 0-3
```

The first step of a cold run compiles the trainer's programs, and sglang has to be restarted with
every training run. The
[README](https://github.com/radixark/miles/blob/main/scripts/tpu/README.md) covers both, along with
the recipes, the compile cache, and per-model trainer settings.

## Next steps

- [`scripts/tpu/README.md`](https://github.com/radixark/miles/blob/main/scripts/tpu/README.md) — the full recipe and operating notes.
- [Docker build](/developer/ci/02-docker-build) — the TPU Dockerfile, variants, and tags.
