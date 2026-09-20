"""Conservative, single-turn GRPO smoke filters (not benchmark accuracy gates).

Use either public function as ``--dynamic-sampling-filter-path``. Both retain
all samples, rewards, statuses and existing token masks. On acceptance only,
TRUNCATED samples get ``remove_sample=True``; the standard train-data converter
then zeros their response loss masks *after* computing group reward statistics.
No additional ``--rollout-sample-filter-path`` is needed.

The general policy allows masked/truncated rewards to supply the group baseline.
The stricter policy also requires reward diversity among trainable COMPLETED
samples. With an independently audited binary grader this supplies completed
correct/incorrect evidence, rather than diversity caused only by truncation.
Neither policy verifies answer parsing, guarantees nonzero parameter gradients,
implements a sampling budget, or establishes benchmark quality. A later sample
filter must not remove all surviving signal or unmask a truncated response;
recheck final advantages at the training boundary when composing other hooks.

Scope: one prompt group with one Sample per rollout, including the usual four
samples. Segmented/multi-turn groups are deliberately unsupported. Additional
advantage whitening, OPD and custom reward/conversion hooks are rejected because
a group-local filter cannot certify their final training advantages. PPO loss
ratios, reduction denominators and normalization settings are never changed.
"""

import math
from argparse import Namespace
from numbers import Real

import torch

from miles.rollout.filter_hub.base_types import FilterOutput
from miles.utils.types import Sample

__all__ = [
    "mask_truncated_and_check_advantages",
    "mask_truncated_and_require_completed_reward_diversity",
]

_REWARD_STD_EPS = 1e-8


def mask_truncated_and_check_advantages(
    args: Namespace, samples: list[Sample | list[Sample]], **kwargs
) -> FilterOutput:
    """Keep a diverse group only if its final loss mask supports nonzero GRPO advantages.

    Existing removals are respected, including on COMPLETED samples. A response
    with ``loss_mask=None`` contributes ``response_length`` tokens unless removed
    or truncated. Masked rewards still participate in the unchanged group mean/std.
    Rejected groups are left untouched. Calls are stateless and idempotent.
    """
    return _filter_group(args, samples, require_completed_reward_diversity=False)


def mask_truncated_and_require_completed_reward_diversity(
    args: Namespace, samples: list[Sample | list[Sample]], **kwargs
) -> FilterOutput:
    """Stricter smoke policy: also require diverse rewards on trainable COMPLETED samples.

    This can reject useful groups accepted by the general policy, so it is an
    explicit opt-in proof gate, not the universal truncation policy. For DAPO,
    audited -1/+1 scores on completed answers provide the intended evidence.
    """
    return _filter_group(args, samples, require_completed_reward_diversity=True)


def _validate_configuration(args: Namespace) -> None:
    if args.advantage_estimator != "grpo":
        raise ValueError("Conservative truncation smoke filters require advantage_estimator=grpo.")
    if args.normalize_advantages:
        raise ValueError(
            "Conservative truncation smoke filters cannot certify --normalize-advantages: "
            "DP/CP token whitening must be checked at the training boundary. No setting was changed."
        )
    if args.n_samples_per_prompt < 2:
        raise ValueError("Conservative truncation smoke filters require at least two samples per prompt.")
    for option in ("custom_reward_post_process_path", "custom_convert_samples_to_train_data_path", "use_opd"):
        if getattr(args, option, None):
            raise ValueError(f"Conservative truncation smoke filters do not support {option}.")


def _validate_group(args: Namespace, samples: list[Sample | list[Sample]]) -> str | None:
    if any(not isinstance(sample, Sample) for sample in samples):
        raise ValueError("Conservative truncation smoke filters require a flat, single-turn Sample group.")
    if len(samples) != args.n_samples_per_prompt:
        return "incomplete_prompt_group"
    if len({sample.group_index for sample in samples}) != 1:
        return "mixed_prompt_groups"
    # Match the train converter's rollout identity, without accepting segmented
    # siblings as independent reward samples. Missing IDs use distinct row keys.
    keys = [
        (
            sample.rollout_id
            if sample.rollout_id is not None
            else (sample.index if sample.index is not None else ("row", position))
        )
        for position, sample in enumerate(samples)
    ]
    if len(set(keys)) != len(keys):
        raise ValueError("Conservative truncation smoke filters require one Sample per distinct rollout.")
    if any(sample.status not in (Sample.Status.COMPLETED, Sample.Status.TRUNCATED) for sample in samples):
        return "unfinished_or_failed_sample"
    for sample in samples:
        try:
            sample.validate()
        except AssertionError:
            return "invalid_sample_lengths"
        if sample.loss_mask is not None and any(value not in (0, 1) for value in sample.loss_mask):
            return "invalid_loss_mask"
        if sample.rollout_log_probs is not None and not all(math.isfinite(v) for v in sample.rollout_log_probs):
            return "nonfinite_rollout_log_probs"
    return None


def _group_advantages(args: Namespace, rewards: torch.Tensor) -> torch.Tensor:
    # Same float32 arithmetic as _normalize_rewards_by_rollout for one sample
    # per rollout. Keep this prediction read-only; the converter remains the
    # authority that computes the actual rewards. Conversion-parity tests pin it.
    advantages = rewards.clone()
    if args.rewards_normalization:
        advantages = rewards - rewards.mean()
        if args.grpo_std_normalization:
            std = rewards.std()
            if std > 0:
                advantages = advantages / (std + 1e-6)
    return advantages


def _filter_group(
    args: Namespace, samples: list[Sample | list[Sample]], *, require_completed_reward_diversity: bool
) -> FilterOutput:
    _validate_configuration(args)
    if reason := _validate_group(args, samples):
        return FilterOutput(keep=False, reason=reason)

    try:
        selected_rewards = [sample.get_reward_value(args) for sample in samples]
        if any(not isinstance(reward, Real) for reward in selected_rewards):
            return FilterOutput(keep=False, reason="missing_or_invalid_reward")
        raw_rewards = [float(reward) for reward in selected_rewards]
    except (KeyError, TypeError, ValueError, OverflowError):
        return FilterOutput(keep=False, reason="missing_or_invalid_reward")
    if not all(math.isfinite(reward) for reward in raw_rewards):
        return FilterOutput(keep=False, reason="nonfinite_reward")
    rewards = torch.tensor(raw_rewards, dtype=torch.float32, device="cpu")
    if not bool(torch.isfinite(rewards).all()):
        return FilterOutput(keep=False, reason="nonfinite_float32_reward")
    reward_std = torch.tensor(raw_rewards, dtype=torch.float64, device="cpu").std()
    if not bool(reward_std > _REWARD_STD_EPS):
        return FilterOutput(keep=False, reason="zero_reward_std")

    active = torch.tensor(
        [
            sample.status == Sample.Status.COMPLETED
            and not sample.remove_sample
            and sample.effective_response_length > 0
            # A strict grader's invalid/format penalty is not proof of a
            # completed mathematical error. Keep its reward in the baseline,
            # but do not let it manufacture completed-answer diversity.
            and (not isinstance(sample.reward, dict) or sample.reward.get("valid", True))
            for sample in samples
        ],
        dtype=torch.bool,
        device="cpu",
    )
    if not bool(active.any()):
        return FilterOutput(keep=False, reason="no_effective_response_tokens")

    advantages = _group_advantages(args, rewards)
    if not bool(torch.isfinite(advantages).all()):
        return FilterOutput(keep=False, reason="nonfinite_group_advantages")
    if not bool((advantages[active] != 0).any()):
        return FilterOutput(keep=False, reason="zero_effective_advantages")

    if require_completed_reward_diversity:
        # Check the dtype the trainer will actually use: float64-only reward
        # differences that disappear in float32 do not supply completed proof.
        completed_rewards = rewards[active]
        if completed_rewards.numel() < 2 or not bool(completed_rewards.std() > _REWARD_STD_EPS):
            return FilterOutput(keep=False, reason="no_completed_reward_diversity")

    # Mutation is the hook's explicit policy: leave all other fields, objects
    # and group membership unchanged, including previously removed samples.
    for sample in samples:
        if sample.status == Sample.Status.TRUNCATED:
            sample.remove_sample = True
    return FilterOutput(keep=True)
