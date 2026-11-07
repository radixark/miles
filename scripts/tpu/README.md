# miles on TPU

GRPO training on one TPU v7x 2x2x1 host (4 chips, 2 TensorCores each, so 8 devices): the torchtitan trainer runs
as 4 ranks on devices 0-3 (chips 0-1), and sglang-torchtpu serves rollouts on devices 4-7 (chips 2-3) as an external
engine. The two jobs share the host's slice, so weights move from trainer to engine device to device over ICI
(`--update-weight-transfer-mode ici`).

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
# names): the PyPI sglang-router drops return_routed_experts, which routing replay needs
pip install --no-deps --force-reinstall /path/to/sglang_router-*.whl
```

Weight sync over ICI needs a torch_tpu build with job-local worlds (`TORCH_TPU_JOB_DEVICE_IDS`), so each job compiles
and runs its collectives over its own devices: google-pytorch/torch_tpu#6110, which `docker/Dockerfile.tpu` installs
as the `TORCH_TPU_WHEELS_TAG` wheel. sglang-torchtpu's shared-slice receiver (sgl-project/sglang-torchtpu#1884) is in
the pinned sglang-torchtpu. The recipe was tested with libtpu 0.0.50; 0.0.49 was reported to halt the SparseCore when
two jobs share the slice.

The first step of a run compiles the trainer's programs, about 20-30 minutes. The trainer's ranks share each compiled
program, and `--compile-cache-dir <dir>` keeps them across runs, so a rerun with the same model and settings starts
its first step in seconds. Use a fresh directory for each torch_tpu build: the cache key covers the graph, shapes,
compiler options, libtpu and kernel bodies, but not torch_tpu's own lowering code.

## Run

`run_qwen3_4b_torchtitan.py` runs Qwen3-4B with Adam on the device, through its `sglang` and `train`
commands; `torchtitan_launch.py` holds what a launcher for another model would share. Model and data
default to `/root/models` and `/root/datasets` (`--model-dir`, `--data-dir`); `train` downloads the
model, dapo-math-17k and aime-2024 there if missing.

1. Start sglang on devices 4-7, in its own shell. It waits for the trainer to join the slice before it
   compiles, so `/health` returns 200 only after `train` has started; start `train` a minute or two after
   sglang, without waiting for it:

   ```bash
   python3 scripts/tpu/run_qwen3_4b_torchtitan.py sglang
   ```

2. Train on devices 0-3:

   ```bash
   python3 scripts/tpu/run_qwen3_4b_torchtitan.py train --recipe full
   ```

   `--recipe smoke` runs 3 small rollouts (32 samples each) to check the setup; `--recipe bench` runs 4 rollouts
   at full shapes without eval, for timing. `--compile-cache-dir <dir>` reuses compiled programs across runs.
   Pass extra `train.py` flags through `--extra-args`.

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
