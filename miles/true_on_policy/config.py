from __future__ import annotations

import os
import shlex
from dataclasses import dataclass, field
from typing import Any, Literal

from .contracts import get_true_on_policy_contract
from .schema import TrueOnPolicyContractSchema
from .model_profiles import TrueOnPolicyModelProfile, get_true_on_policy_model_profile


TrainBackend = Literal["megatron"]


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
    train_context_parallel_size: int
    train_pipeline_parallel_size: int
    rollout_num_gpus_per_engine: int
    # CP IS TWO ORTHOGONAL AXES, and conflating them made CP unreachable for DSA families once.
    #   train_cp_comm_type  -> how ATTENTION communicates: "a2a" is Ulysses, "all_gather" is the
    #                          DSA gather-per-chunk scheme, and "p2p" is ring. None means
    #                          megatron's own default, ["p2p"] -- the one scheme that can never
    #                          be bitwise against a cp=1 rollout.
    #   train_allgather_cp  -> miles' SEQUENCE LAYOUT: contiguous chunks per rank (True) vs the
    #                          zigzag ring layout (False). Independent of the comm type above.
    train_cp_comm_type: str | None = None
    train_allgather_cp: bool = False
    # EP is stated on BOTH sides because the degrees may legitimately differ: the combine is
    # per-slot at the token owner, so it references no rank and no contributor count. A
    # per-CONTRIBUTOR fold would not be degree-invariant -- that distinction is the reason an
    # earlier ep=4-vs-ep=2 measurement read 0.03125 and the current one reads 0.0.
    train_expert_parallel_size: int = 1
    rollout_expert_parallel_size: int = 1
    rollout_data_parallel_size: int = 1

    @property
    def uses_train_tp(self) -> bool:
        return self.train_tensor_parallel_size > 1

    @property
    def uses_train_cp(self) -> bool:
        return self.train_context_parallel_size > 1

    @property
    def train_cp_attention_scheme(self) -> str:
        """Derived from the comm type, never from the degree. THREE schemes, not two:

        * "ulysses"  (a2a)          -- transport by heads, no reduction. The only exact one.
        * "allgather"               -- megatron's LOCAL attention implements this and only this, but
                                       with EAGER attention (a full materialized softmax), which is
                                       a different kernel from the rollout's FA3.
        * "ring"     (p2p/a2a+p2p)  -- merges chunks with an online softmax: a different reduction
                                       from the rollout's single call.

        "all_gather" is NOT a flavour of "ring" and must not be folded into it: it adds no
        reduction at all, it runs a different kernel on this code path.
        """
        if self.train_cp_comm_type == "a2a":
            return "ulysses"
        if self.train_cp_comm_type == "all_gather":
            return "allgather"
        return "ring"

    @property
    def train_cp_sequence_layout(self) -> str:
        return "allgather" if self.train_allgather_cp else "zigzag"

    @property
    def required_cp_layout(self) -> str:
        """The `supported_train_layouts` key this CP scheme needs.

        Derived from the scheme rather than fixed at "ulysses_cp" for any cp>1, so a profile can
        say "supports CP, but not that flavour". Fixing it would make CP unreachable for every
        family whose supported flavour is allgather rather than Ulysses.
        """
        return f"{self.train_cp_attention_scheme}_cp"

    @property
    def uses_ulysses_cp(self) -> bool:
        """Kept for callers that only ask "is this the Ulysses program"; not a CP-degree test."""
        return self.uses_train_cp and self.train_cp_attention_scheme == "ulysses"

    @property
    def uses_train_pp(self) -> bool:
        return self.train_pipeline_parallel_size > 1

    @property
    def uses_rollout_tp(self) -> bool:
        return self.rollout_num_gpus_per_engine > 1

    @property
    def uses_train_ep(self) -> bool:
        return self.train_expert_parallel_size > 1

    @property
    def uses_rollout_ep(self) -> bool:
        return self.rollout_expert_parallel_size > 1

    @property
    def rollout_moe_tensor_parallel_size(self) -> int:
        """What sglang gives the EXPERTS once EP and DP have taken their share of the engine.

        sglang derives `moe_tp = gpus_per_engine // ep // dp`, so an engine wider than its EP
        degree silently puts the remainder into MoE tensor parallelism. Invisible while every
        rung had gpus_per_engine == ep; CP is the first axis to consume world size WITHOUT
        consuming EP degree, which is what made it reachable.
        """
        consumed = self.rollout_expert_parallel_size * self.rollout_data_parallel_size
        if consumed <= 0:
            return 1
        return max(1, self.rollout_num_gpus_per_engine // consumed)

    @property
    def uses_rollout_moe_tp(self) -> bool:
        return self.rollout_moe_tensor_parallel_size > 1


@dataclass(frozen=True)
class TrueOnPolicyKernelPolicy:
    """Kernel/runtime switches required to keep SGLang and Megatron aligned."""

    contract: TrueOnPolicyContractSchema
    sglang_attention_backend: str

    def build_sglang_args(self) -> TrueOnPolicyArgList:
        return TrueOnPolicyArgList(
            (
                "--sglang-enable-deterministic-inference",
                "--sglang-true-on-policy-contract",
                self.contract.name,
                "--sglang-attention-backend",
                self.sglang_attention_backend,
            )
        )

    def build_megatron_args(self) -> TrueOnPolicyArgList:
        return TrueOnPolicyArgList(
            (
                "--true-on-policy-contract",
                self.contract.name,
                "--spec",
                "miles_plugins.top.spec",
                "get_top_spec",
                "--transformer-impl",
                "local",
                "--use-cpu-initialization",
                "--batch-invariant-mode",
                "--no-bias-swiglu-fusion",
            )
        )

    def build_env_vars(self) -> dict[str, str]:
        return {
            "NCCL_ALGO": os.environ.get("NCCL_ALGO", "Ring"),
            "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        }


@dataclass(frozen=True)
class TrueOnPolicyLaunchPlan:
    """Derived cross-repo launch contract for one true-on-policy run."""

    enabled: bool
    model_profile: TrueOnPolicyModelProfile | None = None
    contract: TrueOnPolicyContractSchema | None = None
    train_backend: TrainBackend | None = None
    parallel_layout: TrueOnPolicyParallelLayout | None = None
    kernel_policy: TrueOnPolicyKernelPolicy | None = None
    sglang_args: TrueOnPolicyArgList = field(default_factory=TrueOnPolicyArgList)
    megatron_args: TrueOnPolicyArgList = field(default_factory=TrueOnPolicyArgList)
    miles_args: TrueOnPolicyArgList = field(default_factory=TrueOnPolicyArgList)
    env_vars: dict[str, str] = field(default_factory=dict)

    @property
    def train_args(self) -> str:
        return (
            self.sglang_args.as_cli_string()
            + self.megatron_args.as_cli_string()
            + self.miles_args.as_cli_string()
        )


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
    cp_comm_type: str | None = None
    allgather_cp: bool = False
    expert_model_parallel_size: int = 1
    sglang_ep_size: int = 1
    sglang_dp_size: int = 1
    contract_override: str | None = None

    @property
    def parallel_layout(self) -> TrueOnPolicyParallelLayout:
        return TrueOnPolicyParallelLayout(
            train_tensor_parallel_size=self.tensor_model_parallel_size,
            train_context_parallel_size=self.context_parallel_size,
            train_pipeline_parallel_size=self.pipeline_model_parallel_size,
            rollout_num_gpus_per_engine=self.rollout_num_gpus_per_engine,
            train_cp_comm_type=self.cp_comm_type,
            train_allgather_cp=self.allgather_cp,
            train_expert_parallel_size=self.expert_model_parallel_size,
            rollout_expert_parallel_size=self.sglang_ep_size,
            rollout_data_parallel_size=self.sglang_dp_size,
        )

    @property
    def contract(self) -> TrueOnPolicyContractSchema:
        if self.contract_override is not None:
            return get_true_on_policy_contract(self.contract_override)
        return self.model_profile.contract

    def validate(self) -> None:
        if not self.enabled:
            return
        layout = self.parallel_layout
        if self.train_backend != "megatron":
            raise ValueError(
                f"true-on-policy supports the megatron backend only, got {self.train_backend!r}"
            )
        if layout.uses_train_cp:
            # AXIS 1 -- the attention comm. Ring is refused for EVERY family: it merges attention
            # chunks with an online softmax, a different number of rescales than the rollout's one
            # call, and each rescale rounds -- no declaration can make that bitwise. Refused by
            # NAME rather than left to read as unsupported-model.
            scheme = layout.train_cp_attention_scheme
            if scheme == "ring":
                got = layout.train_cp_comm_type or 'megatron default ["p2p"]'
                raise ValueError(
                    f"true-on-policy cannot run ring CP; got {got}. Ring merges attention chunks "
                    "with an online softmax, a different reduction from the rollout's single call, "
                    "so it cannot be bitwise at any degree. Declare a2a (Ulysses) or all_gather, "
                    "whichever scheme the family supports. Note the scheme lives in THIS contract, "
                    "not in megatron's --cp-comm-type alone: under --transformer-impl local "
                    "megatron's own attention has its own comm-type restrictions, and TOP "
                    "substitutes the attention module."
                )
            # Ulysses and allgather are both exact schemes -- WHICH is exact depends on what the
            # family's attention actually runs, so the answer is the profile's, not a blanket
            # rule. Ulysses moves heads with a2a and adds no arithmetic; it is the scheme for a
            # family whose delegated kernel is FA3 over the full sequence. Allgather replicates
            # KV across the group and adds no arithmetic either; it is the scheme for a family
            # whose spec-substituted attention gathers its own KV and runs the delegated kernel
            # per chunk (glm_moe_dsa). What allgather is NOT exact for is megatron's own local
            # DotProductAttention, which implements it with EAGER attention -- which is why the
            # qwen families do not declare it, not why nobody may.
            required = layout.required_cp_layout
            if required not in self.model_profile.supported_train_layouts:
                raise ValueError(
                    f"{self.model_profile.family} does not support "
                    f"{layout.train_cp_attention_scheme}-CP true-on-policy (needs {required!r} in "
                    f"supported_train_layouts; it has "
                    f"{list(self.model_profile.supported_train_layouts)})."
                )
            # AXIS 2 -- the sequence layout, independent of the comm type. A family whose
            # index/kv sharing gathers over the CP group needs contiguous chunks; the zigzag ring
            # layout breaks those gathers.
            if scheme == "ulysses" and layout.train_allgather_cp:
                raise ValueError(
                    "true-on-policy Ulysses CP requires per-sequence zigzag shards; "
                    "--allgather-cp produces contiguous shards that TopAttention cannot restore."
                )
            if self.model_profile.requires_allgather_cp and not layout.train_allgather_cp:
                raise ValueError(
                    f"{self.model_profile.family} requires --allgather-cp at cp>1: it gathers over "
                    "the CP group in the contiguous-chunk layout and the zigzag ring layout is not "
                    f"supported. Got the {layout.train_cp_sequence_layout} layout."
                )
        if layout.uses_train_pp and "pp" not in self.model_profile.supported_train_layouts:
            raise ValueError(f"{self.model_profile.family} does not support PP true-on-policy")
        if layout.uses_train_tp and not self.model_profile.supports_train_tensor_parallel:
            raise ValueError(
                f"{self.model_profile.family} does not support trainer tensor parallelism "
                "under true-on-policy"
            )
        if layout.uses_rollout_tp and not self.model_profile.supports_rollout_tensor_parallel:
            raise ValueError(
                f"{self.model_profile.family} does not support rollout tensor parallelism "
                "under true-on-policy"
            )
        # EP CLAUSE 1 -- the degree itself. A dense family has no expert path to make invariant.
        if (layout.uses_train_ep or layout.uses_rollout_ep) and not self.model_profile.supports_expert_parallel:
            raise ValueError(
                f"{self.model_profile.family} does not support expert-parallel true-on-policy "
                f"(needs 'ep' in supported_train_layouts; it has "
                f"{list(self.model_profile.supported_train_layouts)})."
            )
        # EP CLAUSE 2 -- what EP LEAVES OVER, and it applies ONLY to families that have experts.
        # On a dense model the same leftover IS ordinary rollout tensor parallelism, which is
        # supported and measured to zero at tp=4; reading it as MoE-TP refused every dense engine
        # wider than one GPU. Refused here rather than as TopMoELayer's raise deep in the forward,
        # because the fix is a launch-time one: size the engine to ep*dp. The e2e harness works
        # around this by hand; a production launcher never goes through that harness.
        if self.model_profile.supports_expert_parallel and layout.uses_rollout_moe_tp:
            raise ValueError(
                f"true-on-policy refuses MoE tensor parallelism on the rollout: an engine of "
                f"{layout.rollout_num_gpus_per_engine} GPUs at ep="
                f"{layout.rollout_expert_parallel_size} dp={layout.rollout_data_parallel_size} "
                f"leaves moe_tp={layout.rollout_moe_tensor_parallel_size}, which splits the "
                f"experts' K dimension across ranks so the trainer cannot call one fused kernel. "
                f"Size the engine to ep*dp."
            )

    def build_kernel_policy(self) -> TrueOnPolicyKernelPolicy:
        return TrueOnPolicyKernelPolicy(
            contract=self.contract,
            sglang_attention_backend=self.model_profile.sglang_attention_backend,
        )

    def build_launch_plan(self) -> TrueOnPolicyLaunchPlan:
        self.validate()
        kernel_policy = self.build_kernel_policy()
        # NEVER emit --recompute-logprobs-via-prefill here. The gate must score the logprobs the
        # rollout's DECODE actually produced — the distribution RL samples from. A prefill
        # recompute substitutes a different execution (batch shape, kernels, cuda-graph path) and
        # masks every decode-only gap, so a zero under it certifies the wrong program. The flag
        # stays a manual, per-investigation opt-in; the FSDP plan already refuses it.
        miles_values = [
            "--deterministic-mode",
            "--true-on-policy-mode",
        ]
        layout = self.parallel_layout
        if layout.uses_train_cp:
            # Emit it: megatron's parser DEFAULTS --cp-comm-type to ["p2p"]
            # (megatron/training/arguments.py) and forwards it to the config, so an unemitted
            # declaration silently selects ring -- the one scheme that can never be bitwise.
            # TopAttention asserts it at the seam, so a wrong value fails loud rather than running
            # a different program.
            #
            # This requires the megatron-side change that moved the local-attention CP restriction
            # out of TransformerConfig.__post_init__ and into DotProductAttention.__init__. Config
            # validation cannot know that `--spec` substituted the core attention module, so it
            # rejected a2a for an implementation TOP does not use. Without that change this emission
            # fails at config validation -- which is the correct loud failure, not a silent one.
            #
            # The VALIDATED comm type, not a constant. validate() has already refused ring and
            # checked the scheme against the profile by the time a plan exists, so this emission is
            # the accepted declaration: a2a for a ulysses family, all_gather for an allgather one.
            # A constant here overrides the recipe's flag silently -- megatron's parser keeps the
            # LAST occurrence -- which is how an allgather family ended up building its attention
            # submodule with a2a and dying on DotProductAttention's own assert.
            miles_values += ["--cp-comm-type", layout.train_cp_comm_type]
            if layout.train_allgather_cp:
                miles_values.append("--allgather-cp")
        miles_args = TrueOnPolicyArgList(tuple(miles_values))

        return TrueOnPolicyLaunchPlan(
            enabled=True,
            model_profile=self.model_profile,
            contract=self.contract,
            train_backend=self.train_backend,
            parallel_layout=self.parallel_layout,
            kernel_policy=kernel_policy,
            sglang_args=kernel_policy.build_sglang_args(),
            megatron_args=kernel_policy.build_megatron_args(),
            miles_args=miles_args,
            env_vars=kernel_policy.build_env_vars(),
        )


def _normalize_cp_comm_type(value: Any) -> str | None:
    """Megatron accepts a str or a per-layer list; the contract only reasons about one scheme."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        distinct = {str(v) for v in value}
        if len(distinct) != 1:
            raise ValueError(
                f"true-on-policy needs ONE cp_comm_type for the whole model; got per-layer "
                f"{list(value)!r}. A per-layer mix is a per-layer contract, which the schema "
                f"cannot express yet."
            )
        return distinct.pop()
    return str(value)


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
        cp_comm_type=_normalize_cp_comm_type(getattr(args, "cp_comm_type", None)),
        allgather_cp=bool(getattr(args, "allgather_cp", False)),
        # Defaulted, not required: these reach validate() from callers that predate the axis.
        expert_model_parallel_size=int(getattr(args, "expert_model_parallel_size", None) or 1),
        sglang_ep_size=int(getattr(args, "sglang_ep_size", None) or 1),
        sglang_dp_size=int(getattr(args, "sglang_dp_size", None) or 1),
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
    if config.model_profile.disable_megatron_sequence_parallel:
        args.use_sequence_parallel = False
