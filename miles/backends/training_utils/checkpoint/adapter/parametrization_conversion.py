"""Steps from a source parametrization to the run's; weights convert exactly, optimizer state does not convert."""

import enum
from dataclasses import dataclass

import torch

from miles.backends.training_utils.checkpoint.adapter.parametrization import AdapterParametrization

_EXPERT_INPUT_SIDE = (".experts.gate_proj.", ".experts.up_proj.")
_EXPERT_OUTPUT_SIDE = (".experts.down_proj.",)


class ConversionOp(str, enum.Enum):
    UPCAST_TO_FP32 = "upcast_to_fp32"
    SPLIT_SHARED_LORA_A = "split_shared_lora_a"
    MERGE_SHARED_LORA_A = "merge_shared_lora_a"
    BROADCAST_SHARED_EXPERT_SIDE = "broadcast_shared_expert_side"
    COLLAPSE_SHARED_EXPERT_SIDE = "collapse_shared_expert_side"


@dataclass(frozen=True)
class ConversionStep:
    op: ConversionOp
    # the shared-A group, or empty for whole-adapter steps
    modules: tuple[str, ...] = ()


def plan_parametrization_conversion(
    source: AdapterParametrization, target: AdapterParametrization, *, source_dtype: torch.dtype
) -> list[ConversionStep]:
    steps = []
    if source_dtype != torch.float32:
        steps.append(ConversionStep(ConversionOp.UPCAST_TO_FP32))
    source_groups, target_groups = set(source.shared_lora_a_groups), set(target.shared_lora_a_groups)
    steps += [ConversionStep(ConversionOp.SPLIT_SHARED_LORA_A, g) for g in sorted(source_groups - target_groups)]
    steps += [ConversionStep(ConversionOp.MERGE_SHARED_LORA_A, g) for g in sorted(target_groups - source_groups)]
    if source.experts_shared_outer_loras and not target.experts_shared_outer_loras:
        steps.append(ConversionStep(ConversionOp.BROADCAST_SHARED_EXPERT_SIDE))
    if target.experts_shared_outer_loras and not source.experts_shared_outer_loras:
        steps.append(ConversionStep(ConversionOp.COLLAPSE_SHARED_EXPERT_SIDE))
    return steps


def assert_parametrization_matches_run(
    source: AdapterParametrization,
    target: AdapterParametrization,
    steps: list[ConversionStep],
    *,
    load_optimizer: bool,
    has_optimizer_state: bool,
) -> None:
    """Alpha is not compared: the caller owns it (Tinker adopts the source's, RL compares it to ``--lora-alpha``)."""
    assert (
        source.base_model == target.base_model
    ), f"adapter base {source.base_model!r} != run base {target.base_model!r}"
    assert source.lora_rank == target.lora_rank, f"adapter rank {source.lora_rank} != run rank {target.lora_rank}"
    missing, extra = set(target.target_modules) - set(source.target_modules), set(source.target_modules) - set(
        target.target_modules
    )
    assert not missing and not extra, f"target modules differ: missing {sorted(missing)}, unexpected {sorted(extra)}"
    if load_optimizer:
        assert has_optimizer_state, "the source has no optimizer state; load weights only (--no-load-optim)"
        # every step changes what the optimizer state describes, even when the weights convert exactly
        assert not steps, f"loading optimizer state needs the source's own parametrization, but it needs {steps}"


def _side_of_shared_expert_factor(key: str) -> bool:
    """True when this expert factor is the side shared-outer keeps as ``[1, ...]``."""
    return (key.endswith(".lora_A") and any(s in key + "." for s in _EXPERT_INPUT_SIDE)) or (
        key.endswith(".lora_B") and any(s in key + "." for s in _EXPERT_OUTPUT_SIDE)
    )


def apply_parametrization_conversion(
    weights: dict[str, torch.Tensor], steps: list[ConversionStep]
) -> dict[str, torch.Tensor]:
    """Whole-tensor weights keyed by HF key; shared-A groups store A under the first member only."""
    weights = dict(weights)
    for step in steps:
        if step.op == ConversionOp.UPCAST_TO_FP32:
            weights = {k: v.float() for k, v in weights.items()}
        elif step.op == ConversionOp.SPLIT_SHARED_LORA_A:
            shared = weights[f"{step.modules[0]}.lora_A"]
            for member in step.modules[1:]:
                weights[f"{member}.lora_A"] = shared.clone()
        elif step.op == ConversionOp.MERGE_SHARED_LORA_A:
            first = weights[f"{step.modules[0]}.lora_A"]
            for member in step.modules[1:]:
                copy = weights.pop(f"{member}.lora_A")
                assert torch.equal(
                    copy, first
                ), f"{member}.lora_A differs from {step.modules[0]}.lora_A; cannot share A"
        elif step.op == ConversionOp.BROADCAST_SHARED_EXPERT_SIDE:
            # the per-expert side carries the expert count
            num_experts = max(v.shape[0] for k, v in weights.items() if ".experts." in k and v.dim() == 3)
            for key, value in weights.items():
                if _side_of_shared_expert_factor(key):
                    assert value.shape[0] == 1, f"{key}: shared side must have dim0 = 1, got {tuple(value.shape)}"
                    weights[key] = value.expand(num_experts, *value.shape[1:]).clone()
        elif step.op == ConversionOp.COLLAPSE_SHARED_EXPERT_SIDE:
            for key, value in weights.items():
                if _side_of_shared_expert_factor(key):
                    assert bool((value == value[:1]).all()), f"{key}: experts differ; cannot share the outer factor"
                    weights[key] = value[:1].clone()
    return weights
