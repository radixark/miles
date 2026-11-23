# miles on Trainium

GRPO training on one trn3.48xlarge (16 Neuron devices, 64 logical NeuronCores at LNC 2): the torchtitan trainer runs
one rank per core on cores 0-7 (Qwen3-4B) or 0-31 (Qwen3-30B-A3B), and sglang-neuron serves rollouts with TP 4 on
cores 32-35 as an external engine. Weights reach the engine through disk-delta (`--update-weight-transfer-mode
disk-delta`): the trainer publishes per-sync deltas, the engine's `/pull_weights` patches its host-local checkpoint,
and `update_weights_from_disk` copies it into the compiled graphs' parameters in place, so no graph recompiles.

## Environment

The engine and the trainer share one image, `docker/Dockerfile.neuron`, built on AWS's sglang-neuron release image
(Ubuntu 26.04: torch-neuronx 2.14 needs glibc 2.43). It adds:

- `docker/neuron_patch/sglang_neuron_weight_sync.patch`: the in-place disk reload and `/pull_weights`, which the
  released sglang-neuron lacks (it refuses every weight update);
- torchtitan at the commit miles pins, under the image's torch;
- sglang-miles in its own tree (`/opt/sglang-miles/python`, the launchers' `--sglang-miles-pythonpath`), on the
  trainer's `PYTHONPATH` only: miles parses SGLang's server arguments the way sglang-miles defines them, while the
  engine keeps the image's SGLang.

Run the container with every `/dev/neuron*` device, host networking and `--ulimit memlock=-1`.

## Run

One launcher per model, each with the same `prepare`, `sglang` and `train` commands:

| Launcher | Model | Trainer |
|---|---|---|
| `run_qwen3_4b_torchtitan.py` | Qwen3-4B | FSDP on 8 cores, Adam on the device |
| `run_qwen3_30b_a3b_torchtitan.py` | Qwen3-30B-A3B | FSDP on 32 cores, params, grads and Adam on the host (`--optimizer-cpu-offload`) |

```bash
python scripts/trainium/run_qwen3_4b_torchtitan.py prepare
python scripts/trainium/run_qwen3_4b_torchtitan.py sglang          # stays in the foreground
python scripts/trainium/run_qwen3_4b_torchtitan.py train --recipe smoke
```

The engine compiles its graphs on first start (about 25 minutes for Qwen3-4B, 45 for Qwen3-30B-A3B) into
`--engine-compile-cache-dir`; sglang-neuron refuses a NEFF cache it did not stamp, so the engine keeps its own. The
trainer compiles each decoder block on its first step (about 6 minutes for Qwen3-30B-A3B).

## What runs differently on Neuron

`MILES_ACCELERATOR=neuron` selects `miles/platforms/neuron.py`, whose capabilities route the trainer:

- Attention runs on the nkilib flash kernels with sequence-packing bounds (`neuron_attention.py`): each token
  attends to the earlier tokens of its own packed sequence, with no `[T, T]` score matrix.
- Routed experts run on the NKI blockwise MoE kernels (`neuron_moe.py`): neuronx-cc cannot lower
  `torch._grouped_mm`. The kernels take a multiple of 8192 tokens per call, and every microbatch pads to
  `--seq-length`, so MoE models train at `--seq-length 8192`.
- Static shapes: each decoder block compiles once with the `neuron` backend.
- Vocab-parallel cross entropy selects the target with a one-hot mask (`Platform.gather_backward`): gather's
  scatter_add backward races the collectives on Neuron.
- The engine samples with `SGLANG_NEURON_SAMPLER=temperature`, exact full-vocabulary sampling on the device.
