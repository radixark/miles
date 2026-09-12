"""Adaptive entropy regularization (Skywork-OR1).

With ``--use-adaptive-entropy`` the entropy bonus only acts while the policy's
entropy is at or below ``--entropy-target``, and its coefficient moves by
``--entropy-coef-delta`` per optimizer step, up when entropy is at or below the
target and down otherwise, clamped to ``[--entropy-coef-min, --entropy-coef-max]``.
The coefficient and the gate use the entropy of the previous step, so every
micro-batch of a step applies the same bonus; ``update_adaptive_entropy`` runs
once per step on the reduced ``entropy_loss`` (FSDP, and the last pipeline stage in
Megatron). ``aggregate_train_losses`` reduces over ``effective_dp_cp`` including the
outer ``indep_dp`` group, so every loss rank sees the same entropy and stays in step.

Runtime state lives on ``args`` as ``adaptive_entropy_coef`` and ``adaptive_entropy_last``;
``--entropy-coef`` itself is never mutated. Megatron pickles ``args`` into checkpoints but
does not compare or restore these fields, so a resumed run restarts the controller from
``--entropy-coef``. Reported as ``entropy_coef`` in the train metrics.
"""

from typing import Any

ENTROPY_KEY = "entropy_loss"


def entropy_coef_to_apply(args) -> float:
    """The coefficient for this step's policy loss."""
    if not getattr(args, "use_adaptive_entropy", False):
        return args.entropy_coef
    last = getattr(args, "adaptive_entropy_last", None)
    if last is not None and last > args.entropy_target:
        return 0.0
    return getattr(args, "adaptive_entropy_coef", args.entropy_coef)


def update_adaptive_entropy(args, loss_dict: dict[str, Any]) -> None:
    """Update runtime state and log the gated coefficient for the next step in loss_dict."""
    if not getattr(args, "use_adaptive_entropy", False) or ENTROPY_KEY not in loss_dict:
        return
    entropy = float(loss_dict[ENTROPY_KEY])
    coefficient = getattr(args, "adaptive_entropy_coef", args.entropy_coef)
    if entropy <= args.entropy_target:
        args.adaptive_entropy_coef = min(coefficient + args.entropy_coef_delta, args.entropy_coef_max)
    else:
        args.adaptive_entropy_coef = max(coefficient - args.entropy_coef_delta, args.entropy_coef_min)
    args.adaptive_entropy_last = entropy
    loss_dict["entropy_coef"] = entropy_coef_to_apply(args)
