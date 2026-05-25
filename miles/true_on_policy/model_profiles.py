from __future__ import annotations

from dataclasses import dataclass

from .contracts import (
    QWEN3_DENSE_TRUE_ON_POLICY_V1,
    QWEN3_MOE_TRUE_ON_POLICY_V1,
    LogprobContract,
    ModelFamily,
    TrueOnPolicyContract,
)

ParallelLayout = str


@dataclass(frozen=True)
class TrueOnPolicyModelProfile:
    """Model-specific true-on-policy capabilities and launch defaults."""

    family: ModelFamily
    model_names: tuple[str, ...]
    megatron_model_types: dict[str, str]
    supported_train_layouts: tuple[ParallelLayout, ...]
    supported_rollout_layouts: tuple[ParallelLayout, ...]
    contract: TrueOnPolicyContract
    supports_megatron: bool = True
    supports_fsdp: bool = True

    @property
    def required_kernel_contracts(self):
        return self.contract.required_kernel_contracts

    @property
    def logprob_contract(self) -> LogprobContract:
        return self.contract.logprob_contract

    @property
    def sglang_attention_backend(self) -> str:
        return self.contract.sglang_attention_backend

    @property
    def fsdp_attention_implementation(self) -> str:
        return self.contract.fsdp_attention_implementation

    @property
    def disable_megatron_sequence_parallel(self) -> bool:
        return self.contract.disable_megatron_sequence_parallel

    @property
    def supports_ulysses_cp(self) -> bool:
        return "ulysses_cp" in self.supported_train_layouts

    @property
    def supports_tp_invariant(self) -> bool:
        return "tp" in self.supported_train_layouts or "tp" in self.supported_rollout_layouts

    @property
    def supports_ep_invariant(self) -> bool:
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
    megatron_model_types={
        "Qwen3-0.6B": "qwen3-0.6B",
        "Qwen3-4B": "qwen3-4B",
        "Qwen3-4B-Base": "qwen3-4B",
        "Qwen3-4B-Instruct-2507": "qwen3-4B-Instruct-2507",
    },
    supported_train_layouts=("dp", "tp", "sp", "pp", "ulysses_cp"),
    supported_rollout_layouts=("dp", "tp"),
    contract=QWEN3_DENSE_TRUE_ON_POLICY_V1,
)

QWEN3_MOE_PROFILE = TrueOnPolicyModelProfile(
    family="qwen3_moe",
    model_names=("Qwen3-30B-A3B",),
    megatron_model_types={"Qwen3-30B-A3B": "qwen3-30B-A3B"},
    supported_train_layouts=("dp", "tp", "sp", "expert_tp", "ep", "pp", "ulysses_cp"),
    supported_rollout_layouts=("dp", "tp", "ep"),
    contract=QWEN3_MOE_TRUE_ON_POLICY_V1,
)


_MODEL_PROFILES = (QWEN3_DENSE_PROFILE, QWEN3_MOE_PROFILE)
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
