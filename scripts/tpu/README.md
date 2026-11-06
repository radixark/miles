# miles on TPU

GRPO training on one TPU v7x 2x2x1 host (2 chips per half, 2 TensorCores per chip):
the torchtitan trainer runs as 4 ranks on devices 0-3 (chips 0-1), and sglang-torchtpu serves
rollouts on devices 4-7 (chips 2-3) as an external engine. The two jobs share the host's slice, so
weights move trainer → server device to device over ICI (`--update-weight-transfer-mode ici`).

## Environment

The engine and the trainer share one Python environment. `docker/Dockerfile.tpu` (`python docker/build.py
--variant tpu`, `radixark/miles:dev-tpu`) builds it on sglang-torchtpu's release image with the torch that
`docker/Dockerfile` uses (the torchtitan backend needs torch >= 2.14). To set it up by hand, on that image:

```bash
# the CUDA image's torch (CPU wheel; the image's torch_tpu ships glue for torch 2.11-2.15)
pip install torch==2.14.1 torchvision==0.29.1 --extra-index-url https://download.pytorch.org/whl/cpu

# torchtitan at the commit miles pins, with miles' torch_tpu patch (torch.tpu device calls)
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

Weight sync over ICI also needs:

- `--update-weight-transfer-mode ici` in miles (radixark/miles#4589);
- a torch_tpu build with job-local worlds (`TORCH_TPU_JOB_DEVICE_IDS`), so each job compiles and runs its
  collectives over its own devices (google-pytorch/torch_tpu#6110; `docker/Dockerfile.tpu` installs it as the
  `TORCH_TPU_WHEELS_TAG` wheel);
- sglang-torchtpu's shared-slice receiver (sgl-project/sglang-torchtpu#1884);
- libtpu 0.0.50 ran Qwen3-4B this way (2026-11-04). 0.0.49 was reported to halt the SparseCore when two
  jobs share the slice, which a dense model may not exercise; MoE models on 0.0.49+ are unverified.

Compiled programs are not cached (libtpu cannot load an executable built for part of a slice), so the
first step of every run compiles everything: about 25-30 minutes for Qwen3-4B on a 2x2x1 host.

## Run

`run_qwen3_4b_torchtitan.py` runs Qwen3-4B with Adam on the device, through its `sglang` and `train`
commands; `torchtitan_launch.py` holds what a launcher for another model would share. Model and data
default to `/root/models` and `/root/datasets` (`--model-dir`, `--data-dir`); `train` downloads the
model, dapo-math-17k and aime-2024 there if missing.

1. Start sglang on devices 4-7, in its own shell. It waits for the trainer to join the slice before it
   compiles, so `curl localhost:30000/health` returns 200 only after step 2 has started; start `train`
   a minute or two after sglang, without waiting for the health check:

   ```bash
   python3 scripts/tpu/run_qwen3_4b_torchtitan.py sglang
   ```

2. Train on devices 0-3:

   ```bash
   python3 scripts/tpu/run_qwen3_4b_torchtitan.py train --recipe full
   ```

   `--recipe smoke` runs 3 short rollouts to check the setup; `--recipe bench` runs 4 rollouts
   at full shapes without eval, for timing. Pass extra `train.py` flags through `--extra-args`.

wandb is on when `WANDB_API_KEY` is set.

## Notes

- **Restart sglang with every training run.** Trainer rank 0 hosts the slice's TPU controller, so when a
  run exits the engine loses it: `/health` still answers 200, but the engine's devices stay in the old
  session and the next trainer waits forever ("still in session"). Stop the run with `ray stop` (without
  `--force`; a `kill -9`'d TPU process can take the other half down with it), stop sglang, then start
  both again.
- **Do not mask chips with `TPU_VISIBLE_CHIPS`**: it hangs libtpu's session setup. Both jobs
  address the slice through `TORCH_TPU_TOPOLOGY` and the shared-slice specs; the launcher sets them.
- **Keep crash dumps off small disks** (`ulimit -c 0`): core files from crashed TPU processes
  can be tens of GB.
