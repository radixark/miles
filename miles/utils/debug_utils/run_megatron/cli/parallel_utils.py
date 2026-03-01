"""Parallel configuration utilities for run_megatron CLI."""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class ParallelConfig:
    tp: int = 1
    pp: int = 1
    cp: int = 1
    ep: int | None = None
    etp: int = 1

    @classmethod
    def from_parsed_args(cls, parsed: dict[str, int]) -> ParallelConfig:
        return cls(
            tp=parsed.get("tp", 1),
            pp=parsed.get("pp", 1),
            cp=parsed.get("cp", 1),
            ep=parsed.get("ep"),
            etp=parsed.get("etp", 1),
        )

    @property
    def nproc(self) -> int:
        return self.tp * self.pp * self.cp

    def dir_name(self) -> str:
        """Build directory name from parallel config, e.g. 'tp2_cp2_ep2'."""
        parts: list[str] = [f"tp{self.tp}"]
        if self.pp > 1:
            parts.append(f"pp{self.pp}")
        if self.cp > 1:
            parts.append(f"cp{self.cp}")
        if self.ep is not None and self.ep != self.tp:
            parts.append(f"ep{self.ep}")
        if self.etp > 1:
            parts.append(f"etp{self.etp}")
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
