from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Literal

from .model_profiles import TrueOnPolicyModelProfile, get_true_on_policy_model_profile


OnPolicyTarget = Literal["fsdp", "fsdp_tp"]
TrainBackend = Literal["fsdp", "megatron"]


@dataclass(frozen=True)
class TrueOnPolicyLaunchPlan:
    """Derived cross-repo launch contract for one true-on-policy run."""

    enabled: bool
    model_profile: TrueOnPolicyModelProfile | None = None
    train_backend: TrainBackend | None = None
    sglang_target: OnPolicyTarget | None = None
    train_args: str = ""
    env_vars: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class TrueOnPolicyConfig:
    """Typed contract derived from the single public true-on-policy switch."""

    enabled: bool
    model_profile: TrueOnPolicyModelProfile
    train_backend: TrainBackend
    tensor_model_parallel_size: int
    context_parallel_size: int
    pipeline_model_parallel_size: int
    rollout_num_gpus_per_engine: int
    sglang_target_override: OnPolicyTarget | None = None

    @property
    def requires_tp_invariant_rollout(self) -> bool:
        return self.tensor_model_parallel_size > 1 or self.rollout_num_gpus_per_engine > 1

    @property
    def sglang_target(self) -> OnPolicyTarget:
        if self.sglang_target_override is not None:
            return self.sglang_target_override
        return "fsdp_tp" if self.requires_tp_invariant_rollout else "fsdp"

    def validate(self) -> None:
        if not self.enabled:
            return
        if self.train_backend == "megatron" and not self.model_profile.supports_megatron:
            raise ValueError(f"{self.model_profile.family} does not support Megatron true-on-policy")
        if self.train_backend == "fsdp" and not self.model_profile.supports_fsdp:
            raise ValueError(f"{self.model_profile.family} does not support FSDP true-on-policy")
        if self.context_parallel_size > 1 and not self.model_profile.supports_ulysses_cp:
            raise ValueError(f"{self.model_profile.family} does not support Ulysses CP true-on-policy")
        if self.sglang_target == "fsdp_tp" and not self.model_profile.supports_tp_invariant:
            raise ValueError(f"{self.model_profile.family} does not support TP-invariant true-on-policy")

    def build_launch_plan(self) -> TrueOnPolicyLaunchPlan:
        self.validate()
        train_args = (
            "--sglang-enable-deterministic-inference "
            f"--sglang-rl-on-policy-target {self.sglang_target} "
            f"--sglang-attention-backend {self.model_profile.sglang_attention_backend} "
            "--deterministic-mode "
            "--true-on-policy-mode "
        )

        if self.train_backend == "megatron":
            train_args += (
                "--use-sglang "
                "--transformer-impl local "
                "--use-cpu-initialization "
                "--batch-invariant-mode "
                "--no-bias-swiglu-fusion "
                "--no-rope-fusion "
            )
        elif self.train_backend == "fsdp":
            train_args += f"--attn-implementation {self.model_profile.fsdp_attention_implementation} "
        else:
            raise NotImplementedError(f"Unsupported true-on-policy train backend: {self.train_backend}")

        env_vars = {
            "NCCL_ALGO": os.environ.get("NCCL_ALGO", "Ring"),
            "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        }
        if self.sglang_target == "fsdp_tp":
            env_vars.update(
                {
                    "ROW_LINEAR_ENABLE_INV": "1",
                    "MEGATRON_USE_DETERMINISTIC_ALLREDUCE": "1",
                }
            )

        return TrueOnPolicyLaunchPlan(
            enabled=True,
            model_profile=self.model_profile,
            train_backend=self.train_backend,
            sglang_target=self.sglang_target,
            train_args=train_args,
            env_vars=env_vars,
        )


def _get_required_int(args: Any, name: str) -> int:
    value = getattr(args, name)
    if value is None:
        raise ValueError(f"{name} must be initialized before deriving true-on-policy config")
    return int(value)


def build_true_on_policy_config(args: Any) -> TrueOnPolicyConfig | None:
    if not getattr(args, "true_on_policy", False):
        return None

    profile = get_true_on_policy_model_profile(args.model_name)
    return TrueOnPolicyConfig(
        enabled=True,
        model_profile=profile,
        train_backend=args.train_backend,
        tensor_model_parallel_size=_get_required_int(args, "tensor_model_parallel_size"),
        context_parallel_size=_get_required_int(args, "context_parallel_size"),
        pipeline_model_parallel_size=_get_required_int(args, "pipeline_model_parallel_size"),
        rollout_num_gpus_per_engine=_get_required_int(args, "rollout_num_gpus_per_engine"),
        sglang_target_override=getattr(args, "sglang_rl_on_policy_target", None),
    )


def build_true_on_policy_launch_plan(args: Any) -> TrueOnPolicyLaunchPlan:
    config = build_true_on_policy_config(args)
    if config is None:
        return TrueOnPolicyLaunchPlan(enabled=False)
    return config.build_launch_plan()


def apply_true_on_policy_script_defaults(args: Any) -> None:
    """Apply derived defaults that must be visible before command assembly."""
    config = build_true_on_policy_config(args)
    if config is None:
        return

    config.validate()
    args.sglang_rl_on_policy_target = config.sglang_target
    if (
        args.train_backend == "megatron"
        and config.model_profile.disable_megatron_sequence_parallel
    ):
        args.use_sequence_parallel = False
