"""``run`` and ``show-model-args`` CLI commands."""

from pathlib import Path
from typing import Annotated

import typer

from miles.utils.debug_utils.run_megatron.cli.commands.args import RunArgs
from miles.utils.debug_utils.run_megatron.cli.parallel_utils import ParallelConfig
from miles.utils.debug_utils.run_megatron.cli.path_utils import resolve_megatron_path, resolve_model_script
from miles.utils.debug_utils.run_megatron.cli.prompt_utils import (
    PromptConfig,
    generate_token_ids,
    write_token_ids_to_tmpfile,
)
from miles.utils.debug_utils.run_megatron.cli.worker_executor import (
    build_dumper_env,
    build_torchrun_cmd,
    build_worker_args,
)
from miles.utils.debug_utils.run_megatron.worker.script_args import WorkerScriptArgs
from miles.utils.misc import exec_command
from miles.utils.typer_utils import dataclass_cli


def register(app: typer.Typer) -> None:
    """Register ``run`` and ``show-model-args`` commands on *app*."""
    app.command()(run)
    app.command(name="show-model-args")(show_model_args)


def run_impl(args: RunArgs) -> None:
    """Core run logic, called by both ``run`` command and ``run_and_compare``."""
    parallel: ParallelConfig = ParallelConfig.from_run_args(args)

    if args.routing_replay_dump_path is not None and parallel.nproc != 1:
        raise ValueError(f"Routing replay dump requires single-rank run (nproc=1), got {parallel}")

    resolved_megatron: Path = resolve_megatron_path(args.megatron_path)

    prompt: PromptConfig = PromptConfig(
        mode=args.prompt_mode,  # type: ignore[arg-type]
        text=args.prompt_text,
        file=args.prompt_file,
        seq_length=args.seq_length,
        apply_chat_template=args.apply_chat_template,
    )
    token_ids: list[int] = generate_token_ids(prompt=prompt, tokenizer_path=args.hf_checkpoint)
    token_ids_file: Path = write_token_ids_to_tmpfile(token_ids)
    print(f"[cli] Token IDs written to {token_ids_file} ({len(token_ids)} tokens)", flush=True)

    script_args: WorkerScriptArgs = WorkerScriptArgs(
        hf_checkpoint=args.hf_checkpoint,
        token_ids_file=token_ids_file,
        role=args.role,
        ref_load=args.ref_load,
        run_backward=args.run_backward,
        source_patcher_config=args.source_patcher_config,
        routing_replay_dump_path=args.routing_replay_dump_path,
        routing_replay_load_path=args.routing_replay_load_path,
        top_k=args.top_k,
        logprob_output=args.logprob_output,
    )
    worker_args_str: str = build_worker_args(
        parallel=parallel,
        sp=args.sp,
        seq_length=args.seq_length,
        batch_size=args.batch_size,
        script_args=script_args,
        extra_args=args.extra_args,
    )

    dumper_env: dict[str, str] = build_dumper_env(
        output_dir=args.output_dir,
        run_backward=args.run_backward,
        dumper_filter=args.dumper_filter,
    )
    env_prefix: str = " ".join(f"{k}={v}" for k, v in dumper_env.items())

    cmd: str = build_torchrun_cmd(
        model_type=args.model_type,
        megatron_path=resolved_megatron,
        nproc=parallel.nproc,
        worker_args=worker_args_str,
    )
    exec_command(f"{env_prefix} {cmd}")
    print(f"[cli] Run completed. Output: {args.output_dir}", flush=True)


@dataclass_cli(env_var_prefix="")
def run(args: RunArgs) -> None:
    """Launch torchrun to run Megatron standalone forward (or forward+backward)."""
    run_impl(args)


def show_model_args(
    model_type: Annotated[str, typer.Option(help="Model type matching scripts/models/{model_type}.sh")],
) -> None:
    """Show the MODEL_ARGS for a given model type (debug helper)."""
    output: str | None = exec_command(
        f'source "{resolve_model_script(model_type)}" && echo "${{MODEL_ARGS[@]}}"',
        capture_output=True,
    )
    if output:
        print(output.strip())
