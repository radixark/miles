"""What an adapter is, independent of where and how it was trained."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

PARAMETRIZATION_FILENAME = "adapter_parametrization.json"


@dataclass(frozen=True)
class AdapterParametrization:
    """Keys are HF module paths; a fused HF module's components carry a suffix (``in_proj_qkv`` -> ``in_proj_q``)."""

    base_model: str
    lora_rank: int
    # None only in a Reader's output for sources that do not record it (torchtitan)
    lora_alpha: float | None
    lora_type: str
    target_modules: tuple[str, ...]
    experts_shared_outer_loras: bool
    # members of each group share one lora_A; adapter_state/ stores it under the first member
    shared_lora_a_groups: tuple[tuple[str, ...], ...] = ()

    def __post_init__(self):
        assert self.lora_rank > 0, f"lora_rank must be positive, got {self.lora_rank}"
        # backends list members in their own order; one order makes equal groups equal and fixes which member holds A
        object.__setattr__(
            self, "shared_lora_a_groups", tuple(sorted(tuple(sorted(group)) for group in self.shared_lora_a_groups))
        )
        grouped = [module for group in self.shared_lora_a_groups for module in group]
        assert len(grouped) == len(set(grouped)), f"a module is in two shared-A groups: {self.shared_lora_a_groups}"
        unknown = set(grouped) - set(self.target_modules)
        assert not unknown, f"shared-A group members are not targets: {sorted(unknown)}"

    def adapter_state_keys(self) -> set[str]:
        """The HF keys ``adapter_state/`` holds: every B, and each A once, under its group's first member."""
        shared_copies = {module for group in self.shared_lora_a_groups for module in group[1:]}
        return {f"{m}.lora_B" for m in self.target_modules} | {
            f"{m}.lora_A" for m in self.target_modules if m not in shared_copies
        }


def write_parametrization(directory: Path, parametrization: AdapterParametrization) -> None:
    assert parametrization.lora_alpha is not None, "lora_alpha must be resolved before it is written"
    (directory / PARAMETRIZATION_FILENAME).write_text(json.dumps(asdict(parametrization), indent=2))


def read_parametrization(directory: Path) -> AdapterParametrization:
    fields = json.loads((directory / PARAMETRIZATION_FILENAME).read_text())
    fields["target_modules"] = tuple(fields["target_modules"])
    fields["shared_lora_a_groups"] = tuple(tuple(group) for group in fields["shared_lora_a_groups"])
    return AdapterParametrization(**fields)
