"""Run-level configuration for the Miles-native LoRA plugin."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LoRAConfig:
    """User-selected LoRA behavior, independent of model architecture and parallelism."""

    rank: int
    alpha: float
    dropout: float
    target_modules: frozenset[str]
    a_init_method: str = "xavier"
    expanded_from_all_linear: bool = False

    @property
    def scale(self) -> float:
        return self.alpha / self.rank

    @classmethod
    def from_args(cls, args) -> LoRAConfig:
        """Build an immutable config from Miles arguments.

        Target spelling is normalized here because both raw and bridge launchers
        accept Megatron names. Architecture-specific target validation happens
        after the registry resolves a ``LoRAArchSpec``.
        """
        from miles_plugins.lora.hf_adapter import convert_target_modules_to_hf

        rank = int(args.lora_rank)
        assert rank > 0, "native LoRA requires --lora-rank > 0"
        raw_targets = args.target_modules or ()
        if isinstance(raw_targets, str):
            raw_targets = [target.strip() for target in raw_targets.split(",")]
        non_leaf_selectors = {
            str(target) for target in raw_targets if any(token in str(target) for token in (".", "*", "?", "[", "]"))
        }
        non_leaf_selectors.update(getattr(args, "_lora_non_leaf_target_selectors", ()))
        assert not non_leaf_selectors, (
            "the built-in native LoRA provider accepts only HF or Megatron projection leaf names; "
            f"scoped/wildcard --target-modules selectors {sorted(non_leaf_selectors)} would broaden "
            "silently when converted to HF names. Use explicit leaf names, --megatron-to-hf-mode "
            "bridge, or a model-specific --lora-provider-path that implements scoped matching."
        )
        targets = frozenset(convert_target_modules_to_hf(list(raw_targets)))
        assert targets, (
            "the built-in native LoRA provider has no target modules after parsing/exclusion; "
            "select at least one supported projection or disable LoRA"
        )
        return cls(
            rank=rank,
            alpha=float(args.lora_alpha),
            dropout=float(getattr(args, "lora_dropout", 0.0) or 0.0),
            target_modules=targets,
            a_init_method=getattr(args, "lora_A_init_method", "xavier") or "xavier",
            expanded_from_all_linear=bool(getattr(args, "_target_modules_expanded_from_all_linear", False)),
        )
