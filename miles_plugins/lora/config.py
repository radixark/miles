"""Run-level configuration for the Miles-native LoRA plugin."""

from __future__ import annotations

from dataclasses import dataclass

from miles.utils.lora.utils import matches_lora_target


@dataclass(frozen=True)
class LoRAConfig:
    """User-selected LoRA behavior, independent of model architecture and parallelism.

    ``target_modules`` holds the resolved HF module selectors (``args.hf_lora_targets``);
    ``None`` selects every projection the architecture spec implements.
    """

    rank: int
    alpha: float
    dropout: float
    target_modules: tuple[str, ...] | None
    a_init_method: str = "xavier"

    @property
    def scale(self) -> float:
        return self.alpha / self.rank

    def selects(self, hf_module: str) -> bool:
        return self.target_modules is None or any(
            matches_lora_target(hf_module, target) for target in self.target_modules
        )

    @classmethod
    def from_args(cls, args, *, select_all: bool) -> LoRAConfig:
        return cls(
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            target_modules=None if select_all else tuple(args.hf_lora_targets),
            a_init_method=getattr(args, "lora_A_init_method", "xavier"),
        )
