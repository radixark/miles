---
title: Launch Scripts
description: What a miles-diffusion launch script does when you run it, how it is structured, and the ways to override a recipe.
---
Every recipe ships as a Python launch script under `scripts/`, and starting a run is one command:

```bash
python3 scripts/run_diffusion_grpo_pickscore_5gpu_flowgrpo_aligned.py
```

This page explains what that command does and how to change what it runs. For the meaning of
individual flags, see the [CLI Reference](/diffusion/user-guide/cli-reference).

## How a launch script starts a training job

A launch script is a recipe, not the training process. It assembles the full
`train_diffusion.py` command line, starts a local Ray cluster, and submits the command as a Ray
job — submitted rather than run directly, so the driver lives inside the cluster and sees every
node's GPUs.

| Layer | Location | Role |
|---|---|---|
| Launch script | `scripts/run_*.py` | Holds the recipe: the flag blocks and tuned values |
| Command utilities | `miles/utils/external_utils/command_utils.py` | Starts Ray and submits the job |
| Training entrypoint | `train_diffusion.py` | The train loop (~90 lines), run inside the Ray job |

## The structure of a launch script

Every launcher follows the same layout:

```python
@dataclass
class ScriptArgs(U.ExecuteTrainConfig):     # inherits output_dir, num_nodes, ...
    num_rollout: int = 400
    extra_args: str = ""                     # appended verbatim to the command line

def prepare(args) -> str:                    # download the dataset, return its local path
    ...

def execute(args, data_dir) -> None:
    ckpt_args      = "..."                   # named groups of train_diffusion.py flags
    rollout_args   = "..."
    ...
    U.execute_train(train_args=" ".join(groups), num_gpus_per_node=5, config=args)

@U.dataclass_cli
def main(args): execute(args, prepare(args))
```

### ScriptArgs — script options as flags and `MILES_SCRIPT_*` env vars

`@U.dataclass_cli` exposes each `ScriptArgs` field twice: as a `--kebab-case` CLI option
(`--num-rollout`) and as an environment variable with the `MILES_SCRIPT_` prefix
(`MILES_SCRIPT_NUM_ROLLOUT`). A command-line value beats the env var, which beats the field
default. The env form is how wrappers and cluster tooling inject machine-specific values
without editing the script.

Options shared by every launcher, from the `ExecuteTrainConfig` base class and repo convention:

| Option | Default | Purpose |
|---|---|---|
| `--num-nodes` | `$SLURM_JOB_NUM_NODES` or `1` | Training nodes |
| `--cuda-visible-devices` | inherited from the environment | Physical GPUs the job may use. Applied by exporting it for `ray start`. |
| `--output-dir` | `<repo>/logs` | Where checkpoints and dumps are written |
| `--extra-env-vars` | empty | Extra env vars added to the Ray runtime env |
| `--extra-args` | empty | Extra flags appended to the `train_diffusion.py` command line |

wandb flags are emitted only when `WANDB_API_KEY` is set — logging turns on by exporting the
key, with no script change.

### execute() — the flag groups

The command line is assembled as one string block per concern, then concatenated — the parser
sees a flat list, and the groups follow **concern, not flag prefix** (a `--diffusion-*` flag can
live in `rollout_args`, a `--micro-batch-size-*` flag in `perf_args`).

| Block | What it carries |
|---|---|
| `ckpt_args` | `--hf-checkpoint` (required — also selects the model family), `--save` / `--load` |
| `rollout_args` | Prompt data, batch shape, sampler: steps, guidance, noise level, SDE step strategy |
| `eval_args` | Eval datasets and cadence; eval requests are pure generation, no trajectory |
| `grpo_args` | Advantage estimator, reward normalisation, clip range |
| `optimizer_args` | LR, betas, weight decay |
| `lora_args` | Rank, targets, `--lora-ipc-weight-sync` |
| `reward_args` | `--rm-type` and the reward worker pool |
| `sglang_args` | Engine side: router, concurrency, `--sglang-*` passthrough |
| `train_backend_args` | The three dtype flags, `--update-weight-target-module` |
| `perf_args` | Gradient checkpointing, micro-batch tiling, parser workers |
| `misc_args` | GPU layout, `--colocate`, `--deterministic-mode` |

## Ways to override a recipe

From lightest to heaviest:

1. **Append flags with `--extra-args`.** The value lands at the end of the command line, and for
   an argparse flag given twice the later occurrence wins — so it overrides anything the recipe
   sets:

   ```bash
   python3 scripts/run_diffusion_grpo_pickscore_5gpu_flowgrpo_aligned.py \
       --num-rollout 50 --extra-args "--diffusion-kl-beta 0.02"
   ```

2. **Set a script option, as a flag or a `MILES_SCRIPT_*` env var.**
3. **Edit the script.** The launcher is the canonical home of a recipe's hyperparameters; change
   the flag blocks directly for anything you want to keep.

The Typer flags are also the CI interface: an e2e test names the recipe and passes its knobs
(`script="scripts/run_diffusion_grpo_sd3_ocr_sglang.py", args=["--num-rollout", "2", ...]`), so
launch scripts must stay runnable with no arguments and configurable through `ScriptArgs` — CI
drives them the same way you do.

## What execute_train runs on your machine

1. Kills stale `sglang` / `ray` / `miles` processes.
2. Starts a fresh cluster with `export CUDA_VISIBLE_DEVICES=... && ray start --head` — the
   device list must be in the raylet's own environment; set per job or per actor it never
   reaches the scheduler, which then places work on excluded GPUs.
3. Builds the Ray runtime env: NCCL socket vars, `MASTER_ADDR`, `PYTHONPATH`, plus anything
   from `--extra-env-vars`.
4. Submits the job: `ray job submit -- python3 train_diffusion.py <flags>`.

| Env var | Effect |
|---|---|
| `MILES_SCRIPT_EXTERNAL_RAY=1` | A scheduler already built the Ray cluster: skip the teardown and `ray start`, only submit. `--cuda-visible-devices` must be empty here — export it before each `ray start` instead. Used by the multi-node recipe below. |
| `MILES_SCRIPT_ENABLE_RAY_SUBMIT=0` | Run everything except the submission — shows what a launcher would do |
| `MASTER_ADDR` | Where the single-node `ray start --head` binds, default `127.0.0.1`. The training process group does NOT use it: rank 0 negotiates its real node IP and a free port at runtime, so multi-node needs no setting here |

## The batch-size arithmetic

Diffusion adds a dimension LLM RL does not have: one sample expands into **several train pairs**,
one per trained denoising step.

```
samples per rollout   = rollout_batch_size × n_samples_per_prompt
train pairs           = samples × (number of SDE step indices)
```

The trajectory-level knobs are locked by
[the batch-knob invariant](/diffusion/user-guide/concepts#the-batch-knob-invariant); contradictory values
abort at parse time. `global_batch_size` counts samples and must divide by `dp_size`
(= train world size ÷ `--sequence-parallel-size`).

| Knob | Unit | Meaning |
|---|---|---|
| `--num-steps-per-rollout` | optimizer steps | Windows the pairs are cut into per rollout |
| `--micro-batch-size` | train pairs | Flat: pairs per DiT forward |
| `--micro-batch-size-sample` × `--micro-batch-size-tstep` | samples × timesteps | 2D tile per DiT forward; overrides the flat knob with `S × T` |
| `--diffusion-train-iter-order` | — | Tile visit order (2D only) |
| `--train-dp-split-mode` | — | `contiguous` or `stride` dealing of pairs to DP ranks |

Most recipes use the 2D tile — it states "S samples × T timesteps per forward" directly, which
the flat knob cannot express. Wan2.2 is the exception: `--micro-batch-size 2` is its way of
keeping each forward on one side of the high/low-noise expert boundary.

Worked example (the Qwen-Image recipe):

```
32 prompts × 16 samples             = 512 samples per rollout
512 samples × 2 SDE steps           = 1024 train pairs
1024 pairs ÷ 4 DP ranks             = 256 pairs per rank
256 ÷ num_steps_per_rollout(2)      = 128 pairs per optimizer step per rank
tile = 8 samples × 1 timestep       = 8 pairs per DiT forward → 16 forwards per step
```

Two failure modes to recognize:

| Error | Fix |
|---|---|
| `num_pairs_shard=N not divisible by num_steps_per_rollout=M` | Adjust `rollout_batch_size`, `n_samples_per_prompt`, or `num_steps_per_rollout` so the per-rank pair count splits evenly |
| `Micro-batch mixes denoising phases` (Wan2.2) | A micro-batch spans the expert boundary; set `--micro-batch-size 1` or a phase-pure tiling |

## Adapting a recipe

Start from the closest existing script and change one group at a time:

| Goal | What to change |
|---|---|
| Different reward | `reward_args` — swap `--rm-type`, or point `--custom-rm-path` at your own `async def rm(args, samples) -> list[float]` |
| Different prompts | `--prompt-data`, `--input-key`, and the `prepare()` download |
| Different checkpoint | `--hf-checkpoint`, plus `--diffusion-model-family` if the name carries no family hint |
| Smaller GPU budget | Lower `--rollout-batch-size`, `--n-samples-per-prompt`, `--rollout-microgroup-size`; keep the batch identity satisfied |
| Train more denoising steps | Raise `--diffusion-num-sde-steps` — this multiplies train pairs |
| Try a flag without editing the file | `--extra-args "--flag value"` |

## Multi-node training

The worked example is `scripts/run_diffusion_grpo_wan22_pickscore_17gpu_multinode.py`:
wan2.2-A14B full finetune on 2 nodes × 8 GPUs (train + rollout colocated) plus 1 reward GPU on a
separate node. Multi-node runs submit into a cluster you build yourself; the launcher only submits
(`MILES_SCRIPT_EXTERNAL_RAY=1`).

### Bring up the cluster

Each node runs ONE of the blocks below, chosen by its role. The two `export` lines must be in
the environment of the `ray start` daemon itself, which is why every block repeats them.

On the head node:

```bash
ulimit -n 1000000
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
ray start --head --port=6379 --num-gpus 8 --dashboard-host 0.0.0.0
```

On every other training node:

```bash
ulimit -n 1000000
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
ray start --address=<head-ip>:6379 --num-gpus 8
```

On the reward node:

```bash
ulimit -n 1000000
export CUDA_VISIBLE_DEVICES=0
ray start --address=<head-ip>:6379 --num-gpus 1
```

- **`ulimit -n`**: a non-interactive ssh shell defaults to 1024 open files. The raylet inherits
  it and, once the FSDP actors and engines connect, dies with `epoll: Too many open files` — but
  the driver log shows a misleading `ActorUnavailableError: ... RpcError: Socket closed`. Verify
  with `grep "Max open files" /proc/$(pgrep -f raylet | head -1)/limits` after `ray start`.
- **`CUDA_VISIBLE_DEVICES`** follows the same rule the launcher automates single-node: it must be
  in the raylet's environment, and with an external cluster the launcher refuses a
  `--cuda-visible-devices` of its own.

`ray status` on the head should show every node and the full GPU count.

### Submit

```bash
ulimit -n 1000000   # the driver needs it too
MILES_SCRIPT_EXTERNAL_RAY=1 python3 scripts/run_diffusion_grpo_wan22_pickscore_17gpu_multinode.py
```

Reward workers (`--pickscore-num-workers 4 --pickscore-num-gpus-per-worker 0.25`, no
`--colocate-reward`) are default-scheduled and land on the only free GPU: the reward node.

### Verify the run is healthy

- With `--rollout-patch-group wan` and `--sglang-attention-backend torch_sdpa`,
  `train/model_output_mean_abs_diff` is expected to be exactly `0.0` from the first optimizer step. The 4-GPU proxy E2E
  standard records `0.0`, and the documented 17-GPU runs sustained it over 200 rollouts. Any nonzero value indicates a
  train/rollout parity regression; check the patch group, backend, dtype, and versions on every rank.
- Both training nodes near 100% GPU util during rollout; a 100%-vs-idle split between nodes means
  the reward workers were packed onto one node.

### Multi-node pitfalls

| Pitfall | What to know |
|---|---|
| Stale engine hijacks the port | Killing a driver by name leaves its spawned engine alive; the next driver's health check talks to the old, unpatched server while the new one dies on `EADDRINUSE`. Between runs kill by GPU pid and check `nvidia-smi` shows no compute processes |
| Version strings lie under editable installs | `pip show sglang` reports the install-time commit; trust `git rev-parse HEAD` in the checkout (or the `-e git+...@<sha>` line in `pip freeze`) |
| Weight-sync bucket | `--update-weight-buffer-size 4294967296` (4 GiB). In one documented A14B run, 512 MiB took ~92 s and 4 GiB ~15 s; these are run-specific observations, not a benchmark guarantee. |
| Determinism env vars are trainer-side | `--deterministic-mode` sets `NCCL_DETERMINISTIC` and `CUBLAS_WORKSPACE_CONFIG` on the FSDP actors; check rollout parity with `train/model_output_mean_abs_diff` |

## Next

- [CLI Reference](/diffusion/user-guide/cli-reference) — every flag, grouped.
- [Core Concepts](/diffusion/user-guide/concepts) — what one rollout iteration does, object by object.
- [Dtype Control](/diffusion/advanced/dtype-control) — what the three dtype flags actually do.
