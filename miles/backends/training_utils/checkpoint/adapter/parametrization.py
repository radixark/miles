"""What an adapter is, independent of where and how it was trained."""

import hashlib
import json
from dataclasses import MISSING, asdict, dataclass
from dataclasses import fields as dataclass_fields
from pathlib import Path

import transformers

from miles.backends.training_utils.checkpoint.adapter.source_error import AdapterSourceError, require_fields

PARAMETRIZATION_FILENAME = "adapter_parametrization.json"
# bumped when a field's meaning changes or a field without a default is added; a file without it is 1
PARAMETRIZATION_FORMAT_VERSION = 1

# HF config fields that describe the base's architecture and shapes
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
    "tie_word_embeddings",
)
# names configs use for the routed expert count
_EXPERT_COUNT_ALIASES = ("num_experts", "num_local_experts", "n_routed_experts")


def base_model_fingerprint(hf_config: dict) -> str:
    """The base's identity for adapter compatibility, from its HF config as a dict (a VL model's ``text_config``):
    ``config.json`` as read or ``PretrainedConfig.to_dict()`` give the same value.

    It summarizes the architecture's sizes, not the weights: two fine-tunes of one architecture share it.
    """
    # a VL wrapper keeps the language model's fields under text_config; a text-only config has no such key
    text_config = _resolved(hf_config.get("text_config") or hf_config)
    fields = {field: text_config[field] for field in _BASE_IDENTITY_FIELDS if field in text_config}
    return hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()[:16]


def _resolved(config: dict) -> dict:
    """The config as its transformers class resolves it: attribute aliases and derived defaults (e.g. head_dim)
    filled in the same way whether the dict is config.json or a to_dict()."""
    try:
        config = transformers.AutoConfig.for_model(**config).to_dict()
    except ValueError:
        # a model_type transformers does not know (remote code): only the expert count aliases are resolved
        config = dict(config)
    counts = {config[alias] for alias in _EXPERT_COUNT_ALIASES if config.get(alias) is not None}
    assert len(counts) <= 1, f"conflicting expert counts {counts} in {config.get('model_type')!r}'s config"
    for alias in _EXPERT_COUNT_ALIASES:
        config.pop(alias, None)
    if counts:
        config["num_experts"] = counts.pop()
    return config


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
    # None from a writer that does not record it
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
    version = fields.pop("format_version")
    if version != PARAMETRIZATION_FORMAT_VERSION:
        raise AdapterSourceError(
            f"{directory / PARAMETRIZATION_FILENAME} has format version {version}; this miles reads version "
            f"{PARAMETRIZATION_FORMAT_VERSION}: load it with the miles version that wrote it"
        )
    # a newer miles adds a field with a default without bumping the version
    unknown = sorted(fields.keys() - {field.name for field in dataclass_fields(AdapterParametrization)})
    if unknown:
        raise AdapterSourceError(
            f"{directory / PARAMETRIZATION_FILENAME} has unknown fields {unknown}: "
            "load it with the miles version that wrote it"
        )
    required = [field.name for field in dataclass_fields(AdapterParametrization) if field.default is MISSING]
    require_fields(fields, required, directory / PARAMETRIZATION_FILENAME)
    fields["target_modules"] = tuple(fields["target_modules"])
    fields["shared_lora_a_groups"] = tuple(tuple(group) for group in fields["shared_lora_a_groups"])
    return AdapterParametrization(**fields)
