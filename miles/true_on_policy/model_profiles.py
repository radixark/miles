from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


ModelFamily = Literal["qwen3_dense", "qwen3_moe", "qwen_next"]


@dataclass(frozen=True)
class TrueOnPolicyModelProfile:
    """Model-specific true-on-policy capabilities and launch defaults."""

    family: ModelFamily
    model_names: tuple[str, ...]
    megatron_model_types: dict[str, str]
    supports_megatron: bool = True
    supports_fsdp: bool = True
    supports_ulysses_cp: bool = True
    supports_tp_invariant: bool = True
    sglang_attention_backend: str = "fa3"
    fsdp_attention_implementation: str = "flash_attention_3"
    disable_megatron_sequence_parallel: bool = True

    def megatron_model_type_for(self, model_name: str) -> str:
        try:
            return self.megatron_model_types[model_name]
        except KeyError as exc:
            supported = ", ".join(sorted(self.megatron_model_types))
            raise ValueError(
                f"{model_name!r} does not have a Megatron model type in profile "
                f"{self.family!r}; supported names: {supported}"
            ) from exc


QWEN3_DENSE_PROFILE = TrueOnPolicyModelProfile(
    family="qwen3_dense",
    model_names=(
        "Qwen3-0.6B",
        "Qwen3-4B",
        "Qwen3-4B-Base",
        "Qwen3-4B-Instruct-2507",
    ),
    megatron_model_types={
        "Qwen3-0.6B": "qwen3-0.6B",
        "Qwen3-4B": "qwen3-4B",
        "Qwen3-4B-Base": "qwen3-4B",
        "Qwen3-4B-Instruct-2507": "qwen3-4B-Instruct-2507",
    },
)


_MODEL_PROFILES = (QWEN3_DENSE_PROFILE,)
_PROFILE_BY_MODEL_NAME = {
    model_name: profile
    for profile in _MODEL_PROFILES
    for model_name in profile.model_names
}


def get_true_on_policy_model_profile(model_name: str) -> TrueOnPolicyModelProfile:
    try:
        return _PROFILE_BY_MODEL_NAME[model_name]
    except KeyError as exc:
        supported = ", ".join(sorted(_PROFILE_BY_MODEL_NAME))
        raise ValueError(
            f"true-on-policy does not have a model profile for {model_name!r}. "
            f"Supported models: {supported}"
        ) from exc


def get_megatron_model_type(model_name: str) -> str:
    return get_true_on_policy_model_profile(model_name).megatron_model_type_for(model_name)
