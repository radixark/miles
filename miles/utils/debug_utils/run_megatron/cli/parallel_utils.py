"""Parallel configuration utilities for run_megatron CLI."""


def nproc(*, tp: int, pp: int, cp: int) -> int:
    return tp * pp * cp


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
