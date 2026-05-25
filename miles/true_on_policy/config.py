from __future__ import annotations

import os
import shlex
from dataclasses import dataclass, field
from typing import Any, Literal

from .contracts import TrueOnPolicyContract, get_true_on_policy_contract
from .model_profiles import TrueOnPolicyModelProfile, get_true_on_policy_model_profile


OnPolicyTarget = Literal["fsdp", "fsdp_tp"]
TrainBackend = Literal["fsdp", "megatron"]


@dataclass(frozen=True)
class TrueOnPolicyArgList:
    """Structured command-line args that stringify only at launch boundaries."""

    values: tuple[str, ...] = ()

    def as_cli_string(self) -> str:
        if not self.values:
            return ""
        return " ".join(shlex.quote(value) for value in self.values) + " "

    def contains(self, flag: str) -> bool:
        return flag in self.values


@dataclass(frozen=True)
class TrueOnPolicyParallelLayout:
    """Training and rollout topology relevant to true-on-policy parity."""

    train_tensor_parallel_size: int
    train_sequence_parallel: bool
    train_context_parallel_size: int
    train_pipeline_parallel_size: int
    train_expert_model_parallel_size: int
    train_expert_tensor_parallel_size: int
    rollout_num_gpus_per_engine: int
    rollout_expert_parallel_size: int = 1

    @property
    def uses_train_tp(self) -> bool:
        return self.train_tensor_parallel_size > 1

    @property
    def uses_train_sp(self) -> bool:
        return self.train_sequence_parallel and self.uses_train_tp

    @property
    def uses_ulysses_cp(self) -> bool:
        return self.train_context_parallel_size > 1

    @property
    def uses_train_pp(self) -> bool:
        return self.train_pipeline_parallel_size > 1

    @property
    def uses_train_ep(self) -> bool:
        return self.train_expert_model_parallel_size > 1

    @property
    def uses_train_expert_tp(self) -> bool:
        return self.train_expert_tensor_parallel_size > 1

    @property
    def uses_rollout_tp(self) -> bool:
        return self.rollout_num_gpus_per_engine > 1

    @property
    def uses_rollout_ep(self) -> bool:
        return self.rollout_expert_parallel_size > 1


@dataclass(frozen=True)
class TrueOnPolicyKernelPolicy:
    """Kernel/runtime switches required to keep SGLang and Megatron aligned."""

    contract: TrueOnPolicyContract
    deterministic_inference: bool
    deterministic_training: bool
    sglang_attention_backend: str
    megatron_uses_sglang_backend: bool
    disable_rope_fusion: bool
    disable_bias_swiglu_fusion: bool
    batch_invariant_mode: bool
    tp_invariant_row_linear: bool
    deterministic_tp_allreduce: bool
    deterministic_moe_routing: bool
    moe_topk_tiebreak: str | None
    deterministic_moe_dispatch: bool
    deterministic_moe_combine: bool
    ep_invariant_moe: bool
    sglang_attention_data_parallel_size: int
    disable_sglang_cuda_graph: bool
    rollout_tp_size: int

    def build_sglang_args(self) -> TrueOnPolicyArgList:
        values = [
            "--sglang-true-on-policy-contract",
            self.contract.name,
            "--sglang-attention-backend",
            self.sglang_attention_backend,
        ]
        if self.disable_sglang_cuda_graph:
            values.insert(0, "--sglang-disable-cuda-graph")
        if self.deterministic_inference:
            values.insert(0, "--sglang-enable-deterministic-inference")
        if self.sglang_attention_data_parallel_size > 1:
            values.extend(
                [
                    "--sglang-data-parallel-size",
                    str(self.sglang_attention_data_parallel_size),
                    "--sglang-enable-dp-attention",
                ]
            )
        return TrueOnPolicyArgList(tuple(values))

    def build_megatron_args(self) -> TrueOnPolicyArgList:
        values: list[str] = []
        if self.megatron_uses_sglang_backend:
            values.extend(
                [
                    "--true-on-policy-contract",
                    self.contract.name,
                    "--transformer-impl",
                    "local",
                    "--use-cpu-initialization",
                ]
            )
        if (
            self.batch_invariant_mode
            and os.environ.get("MILES_TRUE_ON_POLICY_DISABLE_BATCH_INVARIANT", "0") != "1"
        ):
            values.append("--batch-invariant-mode")
        if self.disable_bias_swiglu_fusion:
            values.append("--no-bias-swiglu-fusion")
        if self.disable_rope_fusion:
            values.append("--no-rope-fusion")
        return TrueOnPolicyArgList(tuple(values))

    def build_env_vars(self) -> dict[str, str]:
        env = {
            "NCCL_ALGO": os.environ.get("NCCL_ALGO", "Ring"),
            "NCCL_NVLS_ENABLE": "0",
            "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "MILES_TRUE_ON_POLICY_ROLLOUT_TP_SIZE": str(self.rollout_tp_size),
            "SGLANG_TRUE_ON_POLICY_FUSED_RMSNORM": "1",
        }
        if self.batch_invariant_mode:
            env.update(
                {
                    "SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_DEEPGEMM": "0",
                    "SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_FALLBACK_VARIANT": "0",
                }
            )
        if self.deterministic_moe_combine:
            env["MODEL_ARGS_DISABLE_MOE_PERMUTE_FUSION"] = "1"
        return env


@dataclass(frozen=True)
class TrueOnPolicyLaunchPlan:
    """Derived cross-repo launch contract for one true-on-policy run."""

    enabled: bool
    model_profile: TrueOnPolicyModelProfile | None = None
    contract: TrueOnPolicyContract | None = None
    train_backend: TrainBackend | None = None
    sglang_target: OnPolicyTarget | None = None
    parallel_layout: TrueOnPolicyParallelLayout | None = None
    kernel_policy: TrueOnPolicyKernelPolicy | None = None
    sglang_args: TrueOnPolicyArgList = field(default_factory=TrueOnPolicyArgList)
    megatron_args: TrueOnPolicyArgList = field(default_factory=TrueOnPolicyArgList)
    fsdp_args: TrueOnPolicyArgList = field(default_factory=TrueOnPolicyArgList)
    miles_args: TrueOnPolicyArgList = field(default_factory=TrueOnPolicyArgList)
    env_vars: dict[str, str] = field(default_factory=dict)

    @property
    def train_args(self) -> str:
        return (
            self.sglang_args.as_cli_string()
            + self.megatron_args.as_cli_string()
            + self.fsdp_args.as_cli_string()
            + self.miles_args.as_cli_string()
        )


@dataclass(frozen=True)
class TrueOnPolicyConfig:
    """Typed contract derived from the single public true-on-policy switch."""

    enabled: bool
    model_profile: TrueOnPolicyModelProfile
    train_backend: TrainBackend
    tensor_model_parallel_size: int
    sequence_parallel: bool
    context_parallel_size: int
    pipeline_model_parallel_size: int
    rollout_num_gpus_per_engine: int
    expert_model_parallel_size: int = 1
    expert_tensor_parallel_size: int = 1
    rollout_expert_parallel_size: int = 1
    train_world_size: int | None = None
    contract_override: str | None = None

    @property
    def parallel_layout(self) -> TrueOnPolicyParallelLayout:
        return TrueOnPolicyParallelLayout(
            train_tensor_parallel_size=self.tensor_model_parallel_size,
            train_sequence_parallel=self.sequence_parallel,
            train_context_parallel_size=self.context_parallel_size,
            train_pipeline_parallel_size=self.pipeline_model_parallel_size,
            train_expert_model_parallel_size=self.expert_model_parallel_size,
            train_expert_tensor_parallel_size=self.expert_tensor_parallel_size,
            rollout_num_gpus_per_engine=self.rollout_num_gpus_per_engine,
            rollout_expert_parallel_size=self.rollout_expert_parallel_size,
        )

    @property
    def requires_tp_invariant_rollout(self) -> bool:
        layout = self.parallel_layout
        return layout.uses_train_tp or layout.uses_train_expert_tp or layout.uses_rollout_tp

    @property
    def sglang_target(self) -> OnPolicyTarget:
        return "fsdp_tp" if self.requires_tp_invariant_rollout else "fsdp"

    @property
    def contract(self) -> TrueOnPolicyContract:
        if self.contract_override is not None:
            return get_true_on_policy_contract(self.contract_override)
        return self.model_profile.contract

    def validate(self) -> None:
        if not self.enabled:
            return
        layout = self.parallel_layout
        if self.contract.model_family != self.model_profile.family:
            raise ValueError(
                f"Contract {self.contract.name!r} is for {self.contract.model_family}, "
                f"but model profile is {self.model_profile.family}"
            )
        if self.train_backend == "megatron" and not self.model_profile.supports_megatron:
            raise ValueError(f"{self.model_profile.family} does not support Megatron true-on-policy")
        if self.train_backend == "fsdp" and not self.model_profile.supports_fsdp:
            raise ValueError(f"{self.model_profile.family} does not support FSDP true-on-policy")
        if layout.uses_ulysses_cp and not self.model_profile.supports_ulysses_cp:
            raise ValueError(f"{self.model_profile.family} does not support Ulysses CP true-on-policy")
        if layout.uses_train_pp and "pp" not in self.model_profile.supported_train_layouts:
            raise ValueError(f"{self.model_profile.family} does not support PP true-on-policy")
        if layout.uses_train_sp and "sp" not in self.model_profile.supported_train_layouts:
            raise ValueError(f"{self.model_profile.family} does not support SP true-on-policy")
        if layout.uses_train_ep and "ep" not in self.model_profile.supported_train_layouts:
            raise ValueError(f"{self.model_profile.family} does not support EP true-on-policy")
        if layout.uses_train_expert_tp and "expert_tp" not in self.model_profile.supported_train_layouts:
            raise ValueError(f"{self.model_profile.family} does not support expert TP true-on-policy")
        if layout.uses_rollout_ep and "ep" not in self.model_profile.supported_rollout_layouts:
            raise ValueError(f"{self.model_profile.family} does not support rollout EP true-on-policy")
        if self.sglang_target == "fsdp_tp" and not self.model_profile.supports_tp_invariant:
            raise ValueError(f"{self.model_profile.family} does not support TP-invariant true-on-policy")
        self._validate_megatron_moe_rollout_topology()
        if (
            self.train_backend == "megatron"
            and layout.uses_train_tp
            and layout.uses_train_ep
            and not layout.uses_train_sp
        ):
            raise ValueError(
                "Megatron MoE true-on-policy requires sequence parallel when train TP and EP "
                "are both enabled."
            )
        self._validate_megatron_train_topology()

    def _validate_megatron_moe_rollout_topology(self) -> None:
        if self.train_backend != "megatron" or self.model_profile.family != "qwen3_moe":
            return

        if self.rollout_expert_parallel_size < 1:
            raise ValueError("SGLang rollout EP must be at least 1 for MoE true-on-policy.")
        if self.rollout_num_gpus_per_engine % self.rollout_expert_parallel_size != 0:
            raise ValueError(
                "Qwen3 MoE true-on-policy requires rollout_num_gpus_per_engine to be "
                "divisible by sglang_expert_parallel_size "
                f"({self.rollout_num_gpus_per_engine} % {self.rollout_expert_parallel_size} != 0)."
            )

        # TODO(true-on-policy): factor in sglang_moe_data_parallel_size when that
        # flag is wired through the qwen3_moe true-on-policy launch path. The
        # SGLang MoE TP is rollout_num_gpus_per_engine / (sglang_ep * sglang_moe_dp);
        # today no script sets sglang_moe_data_parallel_size > 1, so dividing by
        # rollout_expert_parallel_size alone is correct, but enabling MoE DP later
        # will need this validation to multiply rollout_expert_parallel_size by
        # sglang_moe_data_parallel_size before computing rollout_moe_tp_size.
        rollout_moe_tp_size = self.rollout_num_gpus_per_engine // self.rollout_expert_parallel_size
        if rollout_moe_tp_size != self.expert_tensor_parallel_size:
            raise ValueError(
                "Qwen3 MoE true-on-policy requires SGLang MoE TP to match Megatron "
                "expert tensor parallelism: "
                "rollout_num_gpus_per_engine / sglang_expert_parallel_size "
                f"= {rollout_moe_tp_size}, but expert_tensor_parallel_size "
                f"= {self.expert_tensor_parallel_size}."
            )

    def _validate_megatron_train_topology(self) -> None:
        if self.train_backend != "megatron" or self.train_world_size is None:
            return

        model_parallel_size = (
            self.tensor_model_parallel_size
            * self.pipeline_model_parallel_size
            * self.context_parallel_size
        )
        if self.train_world_size % model_parallel_size != 0:
            raise ValueError(
                "Megatron true-on-policy train world size must be divisible by "
                "TP * PP * CP "
                f"({self.train_world_size} % {model_parallel_size} != 0)."
            )

        train_data_parallel_size = self.train_world_size // model_parallel_size
        if self.expert_model_parallel_size > train_data_parallel_size:
            min_world_size = model_parallel_size * self.expert_model_parallel_size
            raise ValueError(
                "Megatron MoE true-on-policy requires EP to fit inside the train "
                "data-parallel dimension: "
                f"DP={train_data_parallel_size}, EP={self.expert_model_parallel_size}. "
                f"This topology needs at least {min_world_size} train GPUs "
                f"for TP={self.tensor_model_parallel_size}, "
                f"PP={self.pipeline_model_parallel_size}, "
                f"CP={self.context_parallel_size}, "
                f"EP={self.expert_model_parallel_size}."
            )
        if train_data_parallel_size % self.expert_model_parallel_size != 0:
            raise ValueError(
                "Megatron MoE true-on-policy requires train DP to be divisible by EP "
                f"({train_data_parallel_size} % {self.expert_model_parallel_size} != 0)."
            )

    def build_kernel_policy(self) -> TrueOnPolicyKernelPolicy:
        policy_kwargs = self.contract.kernel_policy_kwargs_for(
            train_backend=self.train_backend,
            parallel_layout=self.parallel_layout,
        )
        policy_kwargs["rollout_tp_size"] = self.rollout_num_gpus_per_engine
        return TrueOnPolicyKernelPolicy(
            contract=self.contract,
            **policy_kwargs,
        )

    def build_launch_plan(self) -> TrueOnPolicyLaunchPlan:
        self.validate()
        kernel_policy = self.build_kernel_policy()
        miles_args = TrueOnPolicyArgList(("--deterministic-mode", "--true-on-policy-mode"))

        if self.train_backend == "megatron":
            megatron_args = kernel_policy.build_megatron_args()
            fsdp_args = TrueOnPolicyArgList()
        elif self.train_backend == "fsdp":
            megatron_args = TrueOnPolicyArgList()
            fsdp_args = TrueOnPolicyArgList(("--attn-implementation", self.contract.fsdp_attention_implementation))
        else:
            raise NotImplementedError(f"Unsupported true-on-policy train backend: {self.train_backend}")

        return TrueOnPolicyLaunchPlan(
            enabled=True,
            model_profile=self.model_profile,
            contract=self.contract,
            train_backend=self.train_backend,
            sglang_target=self.sglang_target,
            parallel_layout=self.parallel_layout,
            kernel_policy=kernel_policy,
            sglang_args=kernel_policy.build_sglang_args(),
            megatron_args=megatron_args,
            fsdp_args=fsdp_args,
            miles_args=miles_args,
            env_vars=kernel_policy.build_env_vars(),
        )


def _get_required_int(args: Any, name: str) -> int:
    value = getattr(args, name)
    if value is None:
        raise ValueError(f"{name} must be initialized before deriving true-on-policy config")
    return int(value)


def _get_optional_int(args: Any, name: str, default: int) -> int:
    value = getattr(args, name, default)
    if value is None:
        return default
    return int(value)


def build_true_on_policy_config(args: Any) -> TrueOnPolicyConfig | None:
    if not getattr(args, "true_on_policy", False):
        return None

    profile = get_true_on_policy_model_profile(args.model_name)
    num_nodes = getattr(args, "num_nodes", None)
    num_gpus_per_node = getattr(args, "num_gpus_per_node", None)
    train_world_size = (
        None
        if num_nodes is None or num_gpus_per_node is None
        else int(num_nodes) * int(num_gpus_per_node)
    )
    return TrueOnPolicyConfig(
        enabled=True,
        model_profile=profile,
        train_backend=args.train_backend,
        tensor_model_parallel_size=_get_required_int(args, "tensor_model_parallel_size"),
        sequence_parallel=bool(getattr(args, "use_sequence_parallel", False)),
        context_parallel_size=_get_required_int(args, "context_parallel_size"),
        pipeline_model_parallel_size=_get_required_int(args, "pipeline_model_parallel_size"),
        expert_model_parallel_size=_get_optional_int(args, "expert_model_parallel_size", 1),
        expert_tensor_parallel_size=_get_optional_int(args, "expert_tensor_parallel_size", 1),
        rollout_num_gpus_per_engine=_get_required_int(args, "rollout_num_gpus_per_engine"),
        rollout_expert_parallel_size=_get_optional_int(args, "sglang_expert_parallel_size", 1),
        train_world_size=train_world_size,
        contract_override=getattr(args, "true_on_policy_contract", None),
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
    if args.train_backend == "megatron" and config.model_profile.disable_megatron_sequence_parallel:
        args.use_sequence_parallel = False
    if (
        args.train_backend == "megatron"
        and config.context_parallel_size > 1
        and getattr(args, "cp_comm_type", None) is None
    ):
        args.cp_comm_type = "a2a"
