from __future__ import annotations

import dataclasses
from pathlib import Path


def _field(
    default: object = dataclasses.MISSING,
    *,
    help: str,
    flag: str | None = None,
) -> dataclasses.Field:  # type: ignore[type-arg]
    """Shorthand to create a dataclass field with typer metadata."""
    metadata: dict[str, str] = {"help": help}
    if flag is not None:
        metadata["flag"] = flag

    return dataclasses.field(default=default, metadata=metadata)  # type: ignore[arg-type]


@dataclasses.dataclass
class CommonRunArgs:
    model_type: str = _field(help="Model type matching scripts/models/{model_type}.sh")
    hf_checkpoint: Path = _field(help="HuggingFace checkpoint path")
    ref_load: Path | None = _field(default=None, help="Megatron checkpoint path")
    sp: bool = _field(default=False, help="Enable sequence parallelism", flag="--sp")
    run_backward: bool = _field(default=False, help="Run backward pass", flag="--run-backward")
    prompt_mode: str = _field(default="math", help="Prompt mode: math / file / text")
    prompt_text: str | None = _field(default=None, help="Prompt text (for text mode)")
    prompt_file: Path | None = _field(default=None, help="Prompt file (for file mode)")
    # odd + somewhat large; also the fine-structure constant
    seq_length: int = _field(default=137, help="Sequence length")
    batch_size: int = _field(default=1, help="Micro batch size")
    apply_chat_template: bool = _field(default=False, help="Apply chat template", flag="--apply-chat-template")
    role: str = _field(default="actor", help="Model role: actor / critic")
    source_patcher_config: Path | None = _field(default=None, help="Source patcher YAML config path")
    top_k: int = _field(default=0, help="Print top-k predictions per position (0=disabled)", flag="--top-k")
    dumper_filter: str = _field(default="", help="Dumper filter expression")
    megatron_path: Path | None = _field(default=None, help="Path to Megatron-LM")
    extra_args: str = _field(default="", help="Extra args passed to worker")


@dataclasses.dataclass
class RunArgs(CommonRunArgs):
    output_dir: Path = _field(default=Path("/tmp/run_megatron_dump"), help="Dump output directory")
    tp: int = _field(default=1, help="Tensor parallel size")
    pp: int = _field(default=1, help="Pipeline parallel size")
    cp: int = _field(default=1, help="Context parallel size")
    ep: int | None = _field(default=None, help="Expert parallel size (default=tp)")
    etp: int = _field(default=1, help="Expert tensor parallel size")
    routing_replay_dump_path: Path | None = _field(default=None, help="Routing replay dump path")
    routing_replay_load_path: Path | None = _field(default=None, help="Routing replay load path")


@dataclasses.dataclass
class RunAndCompareArgs(CommonRunArgs):
    output_base_dir: Path = _field(help="Base output directory for dumps")
    baseline: str = _field(help='Baseline parallel config, e.g. "--tp 1 --cp 1"')
    target: str = _field(help='Target parallel config, e.g. "--tp 2 --cp 2"')
    routing_replay: bool = _field(
        default=False,
        help="Enable routing replay (record on baseline, replay on target)",
        flag="--routing-replay",
    )


@dataclasses.dataclass
class CompareArgs:
    baseline_dir: Path = _field(help="Baseline dump directory")
    target_dir: Path = _field(help="Target dump directory")
    output_format: str = _field(default="text", help="Output format: text / json")
    grouping: str = _field(default="logical", help="Grouping: logical / raw")
    override_baseline_dims: str | None = _field(default=None, help="Override baseline dims")
    override_target_dims: str | None = _field(default=None, help="Override target dims")
    patch_config: Path | None = _field(default=None, help="Patch config YAML path")
    diff_threshold: float | None = _field(default=None, help="Pass/fail threshold")
