"""Shared helpers for run_megatron CLI."""

import os
import subprocess
import tempfile
from pathlib import Path

import typer

_DEFAULT_MEGATRON_PATH: str = "/root/Megatron-LM"


def exec_command(cmd: str, *, capture_output: bool = False) -> str | None:
    print(f"EXEC: {cmd}", flush=True)
    try:
        result: subprocess.CompletedProcess[str] = subprocess.run(
            ["bash", "-c", cmd],
            check=True,
            capture_output=capture_output,
            **(dict(text=True) if capture_output else {}),
        )
    except subprocess.CalledProcessError as e:
        if capture_output:
            print(f"FAILED: stdout={e.stdout} stderr={e.stderr}")
        raise
    if capture_output:
        return result.stdout + result.stderr
    return None


def resolve_megatron_path(megatron_path: Path | None) -> str:
    if megatron_path is not None:
        return str(megatron_path)
    env_path: str | None = os.environ.get("MEGATRON_PATH")
    if env_path:
        return env_path
    return _DEFAULT_MEGATRON_PATH


def resolve_repo_base() -> Path:
    return Path(os.path.abspath(__file__)).resolve().parents[4]


def resolve_model_script(model_type: str) -> Path:
    repo_base: Path = resolve_repo_base()
    script: Path = repo_base / "scripts" / "models" / f"{model_type}.sh"
    if not script.exists():
        raise typer.BadParameter(f"Model script not found: {script}")
    return script


def build_parallel_dir_name(
    *,
    tp: int,
    pp: int,
    cp: int,
    ep: int | None,
    etp: int,
) -> str:
    """Build directory name from parallel config, e.g. 'tp2_cp2_ep2'."""
    parts: list[str] = [f"tp{tp}"]
    if pp > 1:
        parts.append(f"pp{pp}")
    if cp > 1:
        parts.append(f"cp{cp}")
    if ep is not None and ep != tp:
        parts.append(f"ep{ep}")
    if etp > 1:
        parts.append(f"etp{etp}")
    return "_".join(parts)


def parse_parallel_args(args_str: str) -> dict[str, int]:
    """Parse a parallel config string like '--tp 2 --cp 2' into a dict."""
    tokens: list[str] = args_str.split()
    result: dict[str, int] = {}
    arg_map: dict[str, str] = {
        "--tp": "tp",
        "--pp": "pp",
        "--cp": "cp",
        "--ep": "ep",
        "--etp": "etp",
    }

    i: int = 0
    while i < len(tokens):
        if tokens[i] in arg_map and i + 1 < len(tokens):
            result[arg_map[tokens[i]]] = int(tokens[i + 1])
            i += 2
        else:
            i += 1
    return result


def nproc(*, tp: int, pp: int, cp: int) -> int:
    return tp * pp * cp


def write_prompt_to_tmpfile(prompt_text: str) -> Path:
    tmp: tempfile.NamedTemporaryFile = tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, prefix="run_megatron_prompt_"
    )
    tmp.write(prompt_text)
    tmp.close()
    return Path(tmp.name)
