"""Build torchrun command and worker arguments."""

from pathlib import Path

from miles.utils.debug_utils.run_megatron.cli.path_utils import resolve_model_script
from miles.utils.debug_utils.run_megatron.worker.script_args import WorkerScriptArgs


def build_torchrun_cmd(
    *,
    model_type: str,
    megatron_path: str,
    nproc: int,
    worker_args: str,
) -> str:
    """Build the full shell command to launch the worker via torchrun."""
    model_script: Path = resolve_model_script(model_type)
    worker_module: str = "miles.utils.debug_utils.run_megatron.worker.main"

    cmd: str = (
        f'source "{model_script}" && '
        f"PYTHONPATH={megatron_path}:$PYTHONPATH "
        f"CUDA_DEVICE_MAX_CONNECTIONS=1 "
        f"torchrun --nproc-per-node {nproc} "
        f"-m {worker_module} "
        f"${{MODEL_ARGS[@]}} "
        f"--tokenizer-type HuggingFaceTokenizer "
        f"--hidden-dropout 0 --attention-dropout 0 "
        f"{worker_args}"
    )
    return cmd


def build_worker_args(
    *,
    tp: int,
    pp: int,
    cp: int,
    ep: int | None,
    etp: int,
    seq_length: int,
    batch_size: int,
    script_args: WorkerScriptArgs,
    extra_args: str,
) -> str:
    """Build the worker argument string.

    Megatron-native flags are built manually; ``--script-*`` flags come from
    ``WorkerScriptArgs.to_cli_args()`` automatically.
    """
    effective_ep: int = ep if ep is not None else tp

    megatron_parts: list[str] = [
        f"--tensor-model-parallel-size {tp}",
        "--sequence-parallel" if tp > 1 else "",
        f"--pipeline-model-parallel-size {pp}",
        f"--context-parallel-size {cp}",
        f"--expert-model-parallel-size {effective_ep}",
        f"--expert-tensor-parallel-size {etp}",
        f"--seq-length {seq_length}",
        f"--micro-batch-size {batch_size}",
        f"--global-batch-size {batch_size}",
        "--bf16",
        "--no-gradient-accumulation-fusion",
        "--use-miles-router",
    ]

    if script_args.routing_replay_dump_path is not None or script_args.routing_replay_load_path is not None:
        megatron_parts.append("--use-routing-replay")

    parts: list[str] = [p for p in megatron_parts if p]
    parts.append(script_args.to_cli_args())
    if extra_args:
        parts.append(extra_args)

    return " ".join(parts)


def build_dumper_env(
    *,
    output_dir: Path,
    run_backward: bool,
    dumper_filter: str,
) -> dict[str, str]:
    """Build DUMPER_* environment variables for the worker."""
    env: dict[str, str] = {
        "DUMPER_ENABLE": "1",
        "DUMPER_DIR": str(output_dir),
        "DUMPER_EXP_NAME": "standalone",
    }
    if dumper_filter:
        env["DUMPER_FILTER"] = dumper_filter
    if run_backward:
        env["DUMPER_DUMP_GRAD"] = "1"
    return env
