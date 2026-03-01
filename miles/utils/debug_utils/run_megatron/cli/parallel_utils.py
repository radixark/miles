"""Parallel configuration utilities for run_megatron CLI."""

from __future__ import annotations

import argparse
import dataclasses


@dataclasses.dataclass(frozen=True)
class ParallelConfig:
    tp: int = 1
    pp: int = 1
    cp: int = 1
    ep: int | None = None
    etp: int = 1

    def __post_init__(self) -> None:
        effective_ep: int = self.ep if self.ep is not None else self.tp
        if self.nproc % effective_ep != 0:
            raise ValueError(
                f"nproc ({self.nproc} = tp*pp*cp = {self.tp}*{self.pp}*{self.cp}) "
                f"is not divisible by effective EP ({effective_ep})"
            )

    @property
    def effective_ep(self) -> int:
        return self.ep if self.ep is not None else self.tp

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

    def __str__(self) -> str:
        return f"tp={self.tp}, pp={self.pp}, cp={self.cp}, ep={self.ep}, etp={self.etp}, nproc={self.nproc}"

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
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    for flag in ("tp", "pp", "cp", "ep", "etp"):
        parser.add_argument(f"--{flag}", type=int)
    namespace: argparse.Namespace = parser.parse_args(args_str.split())
    return {k: v for k, v in vars(namespace).items() if v is not None}
