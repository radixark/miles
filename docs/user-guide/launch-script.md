---
title: Launch Script
description: What a Miles launch script does when you run it, how it is structured, and the three ways to override a recipe.
---
Every supported model ships as a python launch script under `scripts/`, and starting a
training run is one command:

```bash
python scripts/run_qwen3_dense.py --model-name Qwen3-4B
```

This page explains what that command does and how to change what it runs. For the
meaning of individual training flags, see [Argument Groups](/user-guide/argument-groups)
and the [CLI Reference](/user-guide/cli-reference).

## How a launch script starts a training job

A launch script is a recipe, not the training process. It assembles the full `train.py`
command line for one model family, starts a local Ray cluster, and submits the command
as a Ray job. The pieces involved:

| Layer | Location | Role |
|---|---|---|
| Launch script | `scripts/run_*.py` | Holds the recipe: per-model values and the flag blocks |
| Model definition | `scripts/models/<type>.py` | Provides the Megatron architecture flags |
| Command utilities | `miles/utils/external_utils/command_utils/` | Starts Ray and submits the job |
| Training entrypoint | `train.py` / `train_async.py` | The actual training process, run inside the Ray job |

{/* FIGURE PLACEHOLDER — horizontal flow diagram:
   scripts/run_*.py → command_utils.execute_train → [ray start --head → ray job submit]
   → train.py (inside the Ray job).
   Side branch: scripts/models/<megatron_model_type>.py —load_model_args→ architecture
   flags spliced into the train.py argv.
   Annotate: everything left of "ray job submit" runs in the head-node shell. */}

When the script starts, it prints its resolved options as a table, then every shell
command it issues with an `EXEC:` prefix — the log is a complete record of what ran.

### Running a v1 launch script

Older Ray launch scripts can keep the free-function API by importing its compatibility module:

```python
import miles.utils.external_utils.command_utils.legacy as U
```

It retains `execute_train`, the command and checkpoint helpers, dataset download,
Mooncake startup, and the original `ExecuteTrainConfig`. New scripts build a backend
with `config.create_backend()`; the compatibility API does not support Kubernetes.

## The structure of a launch script

The sections below walk `scripts/run_qwen3_dense.py`; every launcher follows the same
layout. The module docstring states the prerequisites (a converted checkpoint, the
datasets) and a runnable example, then three parts follow in file order.

### Selecting a recipe with `--model-name`

For some models one launcher provides multiple recipes, selected with `--model-name`.
The per-variant values live in a `_RECIPES` table at the top of the file:

```python
@dataclass(frozen=True)
class _Recipe:
    megatron_model_type: str
    tensor_model_parallel_size: int
    max_tokens_per_gpu: int
    rollout_num_gpus_per_engine: int
    sglang_mem_fraction_static: float
    optimizer_cpu_offload: bool
    num_rollout: int = 3000
    extra_sglang_args: str = ""

_RECIPES: dict[str, _Recipe] = {
    "Qwen3-4B": _Recipe("qwen3-4B", 2, 9216, 2, 0.7, False),
    "Qwen3.5-27B": _Recipe("qwen3.5-27B", 4, 8192, 1, 0.5, True),
    # ... one entry per supported variant
}
```

Single-recipe launchers skip the table and write their values directly in the flag
blocks.

### ScriptArgs — script options as CLI flags and `MILES_SCRIPT_*` env vars

The script's own options are the fields of a `ScriptArgs` dataclass:

```python
@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    run_id: str = U.create_run_id()
    model_name: _MODEL_NAMES = "Qwen3-4B"
    num_gpus_per_node: int = 8
    enable_eval: bool = True
    save: bool = True
    extra_args: str = ""
    data_dir: str = "/root/datasets"
    model_dir: str = "/root/models"
    megatron_path: str = "/root/Megatron-LM"
```

The `@U.dataclass_cli` decorator exposes each field twice: as a `--kebab-case` command
line option (`--model-dir`) and as an environment variable with the `MILES_SCRIPT_`
prefix (`MILES_SCRIPT_MODEL_DIR`). A value given on the command line beats the
environment variable, which beats the field default. The env form is how launch
wrappers and cluster tooling inject machine-specific values without editing the script.

Options shared by every launcher, from the `ExecuteTrainConfig` base class and repo
convention:

| Option | Default | Purpose |
|---|---|---|
| `--model-dir` | `/root/models` | Where checkpoints (HF and `torch_dist`) are read from |
| `--data-dir` | `/root/datasets` | Where prompt and eval datasets are read from |
| `--output-dir` | `/root/shared_data` | Where checkpoints and dumps are written |
| `--num-nodes` | `$SLURM_JOB_NUM_NODES` or 1 | Number of training nodes |
| `--num-gpus-per-node` | 8 | GPUs used on each node |
| `--extra-args` | empty | Extra flags appended to the `train.py` command line |
| `--extra-env-vars` | empty | Extra env vars added to the Ray runtime env |

### execute() — assembling the train.py flags from grouped blocks

The `execute()` function builds the `train.py` command line as one f-string block per
concern, then concatenates them:

```python
ckpt_args = (
    f"--hf-checkpoint {args.model_dir}/{args.model_name} "
    f"--ref-load {args.model_dir}/{args.model_name}_torch_dist "
    f"--load {args.output_dir}/checkpoints "
)

grpo_args = (
    "--advantage-estimator grpo "
    "--use-kl-loss "
    "--kl-loss-coef 0.00 "
    "--eps-clip 0.2 "
    "--eps-clip-high 0.28 "
)

train_args = f"{ckpt_args} {rollout_args} {optimizer_args} {grpo_args} ... {args.extra_args}"
```

Each block maps to a section of [Argument Groups](/user-guide/argument-groups), which
documents the flags themselves:

| Block | Flags documented at |
|---|---|
| `ckpt_args` | [Checkpoint paths](/user-guide/argument-groups#ckpt-args) |
| `rollout_args` | [Sampling and reward](/user-guide/argument-groups#rollout-args) |
| `eval_args` | [Evaluation](/user-guide/argument-groups#eval-args) |
| `perf_args` | [Parallelism and memory](/user-guide/argument-groups#perf-args) |
| `grpo_args` | [RL objective](/user-guide/argument-groups#grpo-args) |
| `optimizer_args` | [Optimizer](/user-guide/argument-groups#optimizer-args) |
| `sglang_args` | [Rollout engine](/user-guide/argument-groups#sglang-args) |

Two blocks have no Argument Groups section: `misc_args` carries the cluster shape
(`--colocate`, `--actor-num-nodes`, `--actor-num-gpus-per-node`), and the wandb flags
come from `U.get_default_wandb_args`, which returns them only when `WANDB_API_KEY` is
set — so wandb logging turns on by exporting the key, with no script change.

## Three ways to override a recipe

From lightest to heaviest:

1. **Append flags with `--extra-args`.** The value is appended to the end of the
   `train.py` command line, and for a flag given twice the later occurrence wins — so
   this overrides any flag the recipe already sets:

   ```bash
   python scripts/run_qwen3_dense.py --model-name Qwen3-4B \
       --extra-args "--lr 2e-6 --num-rollout 100"
   ```

2. **Set a script option, as a flag or an env var.** Anything on `ScriptArgs` can come
   from the command line or from `MILES_SCRIPT_*`:

   ```bash
   MILES_SCRIPT_MODEL_DIR=/mnt/models python scripts/run_qwen3_dense.py --model-name Qwen3-4B
   ```

3. **Edit the script.** The launcher is the canonical home of a recipe's
   hyperparameters and is meant to be read and edited — change the `_RECIPES` values or
   the flag blocks directly for anything you want to keep.

## What execute_train runs on your machine

The launcher hands the assembled flags to `U.execute_train`, which issues the `EXEC:`
commands you see in the log, in order:

1. Kills leftover `sglang`, `miles`, and `redis` processes and stops any previous Ray
   cluster.
2. Starts a fresh cluster with `ray start --head`, using `--num-gpus-per-node` GPUs.
3. Runs the launcher's `before_ray_job_submit` hook, if it has one (used for the ssh
   fan-out below).
4. Builds the Ray runtime env for the job: `PYTHONUNBUFFERED`,
   `CUDA_DEVICE_MAX_CONNECTIONS=1`, `NCCL_NVLS_ENABLE` (your exported value if set,
   otherwise probed with `nvidia-smi`), `MASTER_ADDR`, `no_proxy`, and a `PYTHONPATH`
   containing the repo root and `--megatron-path`, plus anything from
   `--extra-env-vars`.
5. Submits the job: `ray job submit -- python3 train.py <architecture flags>
   <recipe flags>`.
6. On normal driver completion, shuts down only the cells, actors, and subprocesses
   owned by that run's Ray worker manager. The launcher does not sweep host processes
   by name after the job exits.

Two environment variables skip parts of this sequence:

| Env var | Effect |
|---|---|
| `MILES_SCRIPT_EXTERNAL_RAY=1` | The Ray cluster is already running: skip the Ray teardown and `ray start`, only submit |
| `MILES_SCRIPT_ENABLE_RAY_SUBMIT=0` | Run preparation, leave the resulting Ray cluster ready, and skip submission |

The head-node address defaults to `127.0.0.1` and is taken from `MASTER_ADDR`; export
it on multi-node runs so Ray and torch distributed bind to the right interface.

## Multi-step and multi-node launchers

Two launcher shapes go beyond a single `execute()`.

### Subcommand pipelines: prepare-* and full-train

Large-model launchers split the pipeline — download, precision cast, `torch_dist`
conversion, training — into subcommands of one script (underscores in the function name
become dashes on the CLI):

```bash
python scripts/run_deepseek_v4.py prepare-download --model-name DeepSeek-V4-Flash-FP8
python scripts/run_deepseek_v4.py train --model-name DeepSeek-V4-Flash-FP8 \
    --num-nodes 8 --num-gpus-per-node 8
```

The `full-train` subcommand chains all steps and checks a sentinel file before each
one, so a completed step is skipped on re-run — after an interruption, relaunch the
same command and it resumes where it stopped.

### Joining multiple nodes: per-role subcommands and ssh fan-out

A multi-node run needs every node in the Ray cluster before the job is submitted.
Launchers express this in one of two ways:

- **One subcommand per node role.** `scripts/run_nemotron_3_super_120b_a12b.py` has
  `worker` (joins the head's cluster and blocks) and `train` (starts the head, waits
  until the cluster reports every GPU, then submits); you run one command on each node.
- **ssh fan-out from the head.** With `--join-ray-workers`,
  `scripts/run_qwen3_sft.py` sshes every host of an MPI-style hostfile into the
  cluster (`U.ssh_start_ray_workers` as the `before_ray_job_submit` hook), so the whole
  cluster comes up from a single command on the head node.

## Model architecture definitions in scripts/models/

Megatron cannot read the architecture from a HuggingFace checkpoint, so each
`megatron_model_type` has a file in `scripts/models/` named exactly after it, exposing
one function that returns the architecture flags:

```python
def model_args(**kwargs) -> str:
    return """
        --num-layers 36
        --hidden-size 2560
        ...
    """
```

`execute_train` resolves the file by name and splices its output into the `train.py`
command line ahead of the recipe flags. A variant (a layer-pruned debug model, a LoRA
target) derives from its base file with `load_sibling_model_args` instead of copying
it.

<Warning>

**Architecture parameters are not self-validating.** Two checkpoints from the same
family can ship different `--rotary-base`, vocab padding, or normalization epsilon.
Diff the checkpoint's `config.json` against the file in `scripts/models/` before a
first run, and override any drifted value by appending it, e.g.
`--extra-args "--rotary-base 10000"`.

</Warning>

## Next

- [Argument Groups](/user-guide/argument-groups) — which training flags belong to
  which block, and what they mean.
- [CLI Reference](/user-guide/cli-reference) — every flag Miles accepts.
- [Quick Start](/getting-started/quick-start) — downloading and converting a
  checkpoint, the step before any launcher.
- [Customization](/user-guide/customization) — the `--*-path` plug points for custom
  rollout, reward, and filter code.
- [Models](/models/index) — the per-model recipe pages built on these launchers.
