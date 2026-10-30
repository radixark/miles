# miles on TPU

GRPO training of Qwen3-4B on one TPU v7x 2x2x1 host (2 chips per half, 2 TensorCores per chip):
the torchtitan trainer runs as 4 ranks on devices 0-3 (chips 0-1), and sglang-torchtpu serves
rollouts on devices 4-7 (chips 2-3) as an external engine. Weights move trainer → server
through shared memory (`--update-weight-transfer-mode shm`, about 3 s per sync for Qwen3-4B;
`disk` takes about 5 s).

## Environment

The engine and the trainer share one Python environment. `docker/Dockerfile.tpu` (`python docker/build.py
--variant tpu`, `radixark/miles:dev-tpu`) builds it on sglang-torchtpu's release image
(`lmsysorg/sglang:v0.1.0-tpu`) with the torch that `docker/Dockerfile` uses (2.14.1; the torchtitan backend
needs torch >= 2.13). To set it up by hand, on that image:

```bash
# torch 2.14.1 (CPU wheel; the image's torch_tpu ships glue for torch 2.11-2.15)
pip install torch==2.14.1 torchvision==0.29.1 --extra-index-url https://download.pytorch.org/whl/cpu

# torchtitan v0.3.0 (the commit miles pins) with miles' torch_tpu patch: device calls, FSDP reduce, compile
git clone https://github.com/pytorch/torchtitan.git && cd torchtitan
git checkout $(sed -n 's/^ARG TORCHTITAN_COMMIT=//p' /path/to/miles/docker/Dockerfile)
patch --fuzz=0 -p1 < /path/to/miles/docker/tpu_patch/torchtitan_torch_tpu.patch
pip install -e . && cd ..

# miles itself (its transformers pin replaces the image's)
pip install -e /path/to/miles

# miles' router build (the sglang_router wheel of the radixark/miles-wheels release docker/Dockerfile.tpu
# names): the PyPI sglang-router drops return_routed_experts, which routing replay (MoE) needs
pip install --no-deps --force-reinstall /path/to/sglang_router-*.whl
```

## Run

Model and data default to `/root/models` and `/root/datasets` (`--model-dir`, `--data-dir`);
`train` downloads Qwen3-4B, dapo-math-17k and aime-2024 there if missing.

1. Start sglang on devices 4-7, in its own shell, and wait for
   `curl localhost:30000/health` to return 200 (about 5 minutes, mostly compilation):

   ```bash
   python3 scripts/tpu/run_qwen3_4b_torchtitan.py sglang
   ```

2. Train on devices 0-3:

   ```bash
   python3 scripts/tpu/run_qwen3_4b_torchtitan.py train --recipe full
   ```

   `--recipe smoke` runs 3 short rollouts to check the setup; `--recipe bench` runs 4 rollouts
   at full shapes without eval, for timing. Pass extra `train.py` flags through `--extra-args`.

   The trainer's first step compiles its programs for minutes. `--compile-cache-dir DIR` keeps them
   on disk (on a shared filesystem, for every host), so a later run with the same shapes, flags and
   torch_tpu build loads them instead. It grows with every new shape, flag set and build (tens of GB over
   many runs): delete a `torch_tpu-<build>` directory once no job uses that build.
   `--compile-warmup`, which the launcher passes, compiles them on a synthetic batch while
   eval-before-train and rollout 0 run, instead of after rollout 0 arrives.

wandb is on when `WANDB_API_KEY` is set or `~/.netrc` holds the key; `--wandb-project ""`
turns it off.

## Performance

Qwen3-4B, 128 samples × up to 7168 response tokens per step, `--recipe bench`:

| ref log-probs | log-probs | actor train | train step |
|---|---|---|---|
| 10-12 s | 13 s | 32-34 s | 62-67 s |

The trainer compiles each block with `torch.compile` on the `tpu` backend (always on TPU), gathers
FSDP parameters once per pass (`--titan-fsdp-gather-once`), and runs packed-document attention on
native SDPA (`miles/backends/torchtitan_utils/tpu_doc_attention.py`, installed automatically on TPU).

## Notes

- **Stop a training run gracefully while sglang is running.** The host's TPU network session is
  shared, and a `kill -9`'d TPU process can take the other half down with it. Stop a run with
  `ray stop` (without `--force`) and wait for its devices to be released before starting the next:
  `train` force-stops any ray processes still left from an earlier run (it leaves sglang alone).
- **Do not mask chips with `TPU_VISIBLE_CHIPS`**: it hangs libtpu's session setup. Both halves
  address their devices through `TORCH_TPU_TOPOLOGY=1,2,1,2` instead; the launcher sets it.
- **Keep crash dumps off small disks** (`ulimit -c 0`): core files from crashed TPU processes
  can be tens of GB.
