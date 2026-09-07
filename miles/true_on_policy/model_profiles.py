from __future__ import annotations

from dataclasses import dataclass

from typing import Literal

from .contracts import TRUE_ON_POLICY_V1
from .schema import TrueOnPolicyContractSchema

ModelFamily = Literal["qwen3_dense"]

ParallelLayout = str


@dataclass(frozen=True)
class TrueOnPolicyModelProfile:
    """Model-specific true-on-policy capabilities and launch defaults."""

    family: ModelFamily
    model_names: tuple[str, ...]
    megatron_model_types: dict[str, str]
    sglang_attention_backend: str
    supported_train_layouts: tuple[ParallelLayout, ...]
    supported_rollout_layouts: tuple[ParallelLayout, ...]
    contract: TrueOnPolicyContractSchema

    @property
    def disable_megatron_sequence_parallel(self) -> bool:
        return self.contract.disable_megatron_sequence_parallel

    @property
    def supports_ulysses_cp(self) -> bool:
        return "ulysses_cp" in self.supported_train_layouts

    @property
    def requires_allgather_cp(self) -> bool:
        """Does this family's CP need miles' contiguous-chunk sequence layout?

        The SECOND CP axis, orthogonal to the attention comm type. Families whose attention shares
        index/kv across the CP group (DSA) gather in the contiguous layout and break under zigzag;
        Ulysses instead requires per-sequence zigzag shards, enforced by config validation.
        """
        return "allgather_cp" in self.supported_train_layouts

    @property
    def supports_train_tensor_parallel(self) -> bool:
        return "tp" in self.supported_train_layouts

    @property
    def supports_rollout_tensor_parallel(self) -> bool:
        return "tp" in self.supported_rollout_layouts

    @property
    def supports_expert_parallel(self) -> bool:
        return "ep" in self.supported_train_layouts or "ep" in self.supported_rollout_layouts

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
    sglang_attention_backend="fa3",
    megatron_model_types={
        "Qwen3-0.6B": "qwen3-0.6B",
        "Qwen3-4B": "qwen3-4B",
        "Qwen3-4B-Base": "qwen3-4B",
        "Qwen3-4B-Instruct-2507": "qwen3-4B-Instruct-2507",
    },
    supported_train_layouts=("dp", "tp", "pp", "ulysses_cp"),
    supported_rollout_layouts=("dp", "tp"),
    contract=TRUE_ON_POLICY_V1,
)


_MODEL_PROFILES = (QWEN3_DENSE_PROFILE,)
_PROFILE_BY_MODEL_NAME = {model_name: profile for profile in _MODEL_PROFILES for model_name in profile.model_names}


def get_true_on_policy_model_profile(model_name: str) -> TrueOnPolicyModelProfile:
    try:
        return _PROFILE_BY_MODEL_NAME[model_name]
    except KeyError as exc:
        supported = ", ".join(sorted(_PROFILE_BY_MODEL_NAME))
        raise ValueError(
            f"true-on-policy does not have a model profile for {model_name!r}. " f"Supported models: {supported}"
        ) from exc


def get_megatron_model_type(model_name: str) -> str:
    return get_true_on_policy_model_profile(model_name).megatron_model_type_for(model_name)
