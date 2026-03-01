"""Shared Annotated type aliases for typer CLI options.

Metadata is defined once here and reused across all commands.
"""

from pathlib import Path
from typing import Annotated

import typer

ModelTypeOpt = Annotated[str, typer.Option(help="Model type matching scripts/models/{model_type}.sh")]
HfCheckpointOpt = Annotated[Path, typer.Option(help="HuggingFace checkpoint path")]
RefLoadOpt = Annotated[Path | None, typer.Option(help="Megatron checkpoint path")]
TpOpt = Annotated[int, typer.Option(help="Tensor parallel size")]
PpOpt = Annotated[int, typer.Option(help="Pipeline parallel size")]
CpOpt = Annotated[int, typer.Option(help="Context parallel size")]
EpOpt = Annotated[int | None, typer.Option(help="Expert parallel size (default=tp)")]
EtpOpt = Annotated[int, typer.Option(help="Expert tensor parallel size")]
SpOpt = Annotated[bool, typer.Option("--sp", help="Enable sequence parallelism")]
RunBackwardOpt = Annotated[bool, typer.Option("--run-backward", help="Run backward pass")]
PromptModeOpt = Annotated[str, typer.Option(help="Prompt mode: math / file / text")]
PromptTextOpt = Annotated[str | None, typer.Option(help="Prompt text (for text mode)")]
PromptFileOpt = Annotated[Path | None, typer.Option(help="Prompt file (for file mode)")]
SeqLengthOpt = Annotated[int, typer.Option(help="Sequence length")]
BatchSizeOpt = Annotated[int, typer.Option(help="Micro batch size")]
ApplyChatTemplateOpt = Annotated[bool, typer.Option("--apply-chat-template", help="Apply chat template")]
RoleOpt = Annotated[str, typer.Option(help="Model role: actor / critic")]
SourcePatcherConfigOpt = Annotated[Path | None, typer.Option(help="Source patcher YAML config path")]
RoutingReplayDumpOpt = Annotated[Path | None, typer.Option(help="Routing replay dump path")]
RoutingReplayLoadOpt = Annotated[Path | None, typer.Option(help="Routing replay load path")]
DumperFilterOpt = Annotated[str, typer.Option(help="Dumper filter expression")]
MegatronPathOpt = Annotated[Path | None, typer.Option(help="Path to Megatron-LM")]
ExtraArgsOpt = Annotated[str, typer.Option(help="Extra args passed to worker")]
