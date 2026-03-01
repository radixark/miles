"""``run`` and ``show-model-args`` CLI commands."""

from pathlib import Path
from typing import Annotated

import typer

from miles.utils.debug_utils.run_megatron.cli.commands.option_types import (
    ApplyChatTemplateOpt,
    BatchSizeOpt,
    CpOpt,
    DumperFilterOpt,
    EpOpt,
    EtpOpt,
    ExtraArgsOpt,
    HfCheckpointOpt,
    MegatronPathOpt,
    ModelTypeOpt,
    PpOpt,
    PromptFileOpt,
    PromptModeOpt,
    PromptTextOpt,
    RefLoadOpt,
    RoleOpt,
    RoutingReplayDumpOpt,
    RoutingReplayLoadOpt,
    RunBackwardOpt,
    SeqLengthOpt,
    SourcePatcherConfigOpt,
    SpOpt,
    TpOpt,
)
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


def register(app: typer.Typer) -> None:
    """Register ``run`` and ``show-model-args`` commands on *app*."""
    app.command()(run)
    app.command(name="show-model-args")(show_model_args)


def run(
    model_type: ModelTypeOpt,
    hf_checkpoint: HfCheckpointOpt,
    output_dir: Annotated[Path, typer.Option(help="Dump output directory")] = Path("/tmp/run_megatron_dump"),
    ref_load: RefLoadOpt = None,
    tp: TpOpt = 1,
    pp: PpOpt = 1,
    cp: CpOpt = 1,
    ep: EpOpt = None,
    etp: EtpOpt = 1,
    sp: SpOpt = False,
    run_backward: RunBackwardOpt = False,
    prompt_mode: PromptModeOpt = "math",
    prompt_text: PromptTextOpt = None,
    prompt_file: PromptFileOpt = None,
    seq_length: SeqLengthOpt = 137,  # odd + somewhat large; also the fine-structure constant
    batch_size: BatchSizeOpt = 1,
    apply_chat_template: ApplyChatTemplateOpt = False,
    role: RoleOpt = "actor",
    source_patcher_config: SourcePatcherConfigOpt = None,
    routing_replay_dump_path: RoutingReplayDumpOpt = None,
    routing_replay_load_path: RoutingReplayLoadOpt = None,
    dumper_filter: DumperFilterOpt = "",
    megatron_path: MegatronPathOpt = None,
    extra_args: ExtraArgsOpt = "",
) -> None:
    """Launch torchrun to run Megatron standalone forward (or forward+backward)."""
    parallel: ParallelConfig = ParallelConfig(tp=tp, pp=pp, cp=cp, ep=ep, etp=etp)
    resolved_megatron: Path = resolve_megatron_path(megatron_path)

    prompt: PromptConfig = PromptConfig(
        mode=prompt_mode,  # type: ignore[arg-type]
        text=prompt_text,
        file=prompt_file,
        seq_length=seq_length,
        apply_chat_template=apply_chat_template,
    )
    token_ids: list[int] = generate_token_ids(prompt=prompt, tokenizer_path=hf_checkpoint)
    token_ids_file: Path = write_token_ids_to_tmpfile(token_ids)
    print(f"[cli] Token IDs written to {token_ids_file} ({len(token_ids)} tokens)", flush=True)

    script_args: WorkerScriptArgs = WorkerScriptArgs(
        hf_checkpoint=hf_checkpoint,
        token_ids_file=token_ids_file,
        role=role,
        ref_load=ref_load,
        run_backward=run_backward,
        source_patcher_config=source_patcher_config,
        routing_replay_dump_path=routing_replay_dump_path,
        routing_replay_load_path=routing_replay_load_path,
    )
    worker_args_str: str = build_worker_args(
        parallel=parallel,
        sp=sp,
        seq_length=seq_length,
        batch_size=batch_size,
        script_args=script_args,
        extra_args=extra_args,
    )

    dumper_env: dict[str, str] = build_dumper_env(
        output_dir=output_dir,
        run_backward=run_backward,
        dumper_filter=dumper_filter,
    )
    env_prefix: str = " ".join(f"{k}={v}" for k, v in dumper_env.items())

    cmd: str = build_torchrun_cmd(
        model_type=model_type,
        megatron_path=resolved_megatron,
        nproc=parallel.nproc,
        worker_args=worker_args_str,
    )
    exec_command(f"{env_prefix} {cmd}")
    print(f"[cli] Run completed. Output: {output_dir}", flush=True)


def show_model_args(
    model_type: ModelTypeOpt,
) -> None:
    """Show the MODEL_ARGS for a given model type (debug helper)."""
    output: str | None = exec_command(
        f'source "{resolve_model_script(model_type)}" && echo "${{MODEL_ARGS[@]}}"',
        capture_output=True,
    )
    if output:
        print(output.strip())
