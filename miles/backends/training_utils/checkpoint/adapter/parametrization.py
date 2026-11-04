"""What an adapter is, independent of where and how it was trained."""

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

PARAMETRIZATION_FILENAME = "adapter_parametrization.json"
# bumped when a field's meaning changes or a field without a default is added; files from before it are 1
PARAMETRIZATION_FORMAT_VERSION = 1

# the HF config fields that fix every adapted module's name and shape
_BASE_IDENTITY_FIELDS = (
    "model_type",
    "vocab_size",
    "hidden_size",
    "intermediate_size",
    "moe_intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "num_experts",
    "num_local_experts",
    "n_routed_experts",
    "tie_word_embeddings",
)


def base_model_fingerprint(hf_config: dict) -> str:
    """The base's identity for adapter compatibility, from its HF ``config.json`` (a VL model's ``text_config``).

    It identifies the architecture and its shapes, not the weights: two fine-tunes of one architecture share it.
    """
    # a VL wrapper keeps the language model's fields under text_config; a text-only config has no such key
    text_config = hf_config.get("text_config") or hf_config
    fields = {field: text_config[field] for field in _BASE_IDENTITY_FIELDS if field in text_config}
    return hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()[:16]


@dataclass(frozen=True)
class AdapterParametrization:
    """Keys are HF module paths; a fused HF module's components carry a suffix (``in_proj_qkv`` -> ``in_proj_q``)."""

    # where the base was read from: a local path or a hub id, recorded for people; base_model_fingerprint identifies it
    base_model: str
    lora_rank: int
    # None only in a Reader's output for sources that do not record it (torchtitan)
    lora_alpha: float | None
    lora_type: str
    target_modules: tuple[str, ...]
    experts_shared_outer_loras: bool
    # members of each group share one lora_A; adapter_state/ stores it under the first member
    shared_lora_a_groups: tuple[tuple[str, ...], ...] = ()
    # None when written before the fingerprint, or by a writer that does not record it
    base_model_fingerprint: str | None = None

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
    fields = {"format_version": PARAMETRIZATION_FORMAT_VERSION, **asdict(parametrization)}
    (directory / PARAMETRIZATION_FILENAME).write_text(json.dumps(fields, indent=2))


def read_parametrization(directory: Path) -> AdapterParametrization:
    fields = json.loads((directory / PARAMETRIZATION_FILENAME).read_text())
    version = fields.pop("format_version", 1)
    if version != PARAMETRIZATION_FORMAT_VERSION:
        raise ValueError(
            f"{directory / PARAMETRIZATION_FILENAME} has format version {version}; this miles reads version "
            f"{PARAMETRIZATION_FORMAT_VERSION}: load it with the miles version that wrote it"
        )
    fields["target_modules"] = tuple(fields["target_modules"])
    fields["shared_lora_a_groups"] = tuple(tuple(group) for group in fields["shared_lora_a_groups"])
    return AdapterParametrization(**fields)
