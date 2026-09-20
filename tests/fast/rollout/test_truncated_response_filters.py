from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="stage-a-cpu", labels=[])

from argparse import Namespace
from copy import deepcopy

import pytest
import torch

from miles.ray.rollout.train_data_conversion import convert_samples_to_train_data
from miles.rollout.filter_hub.base_types import FilterOutput
from miles.rollout.filter_hub.common_filters import apply_preput_filters
from miles.rollout.filter_hub.truncated_response_filters import (
    _group_advantages,
    mask_truncated_and_check_advantages,
    mask_truncated_and_require_completed_reward_diversity,
)
from miles.utils.function_registry import load_function
from miles.utils.types import Sample

POLICIES = [mask_truncated_and_check_advantages, mask_truncated_and_require_completed_reward_diversity]


def make_args(**overrides):
    return Namespace(
        **{
            "advantage_estimator": "grpo",
            "rewards_normalization": True,
            "grpo_std_normalization": True,
            "normalize_advantages": False,
            "rollout_batch_size": 1,
            "n_samples_per_prompt": 4,
            "reward_key": None,
            "use_dynamic_global_batch_size": False,
            **overrides,
        }
    )


def make_group(rewards=(-1.0, 1.0, -1.0, 1.0), *, truncated=()):
    return [
        Sample(
            group_index=12,
            index=index,
            tokens=[10, 11, 12, 13],
            response_length=3,
            response=f"Answer: {index}",
            label="1",
            reward=reward,
            rollout_log_probs=[-0.2, -0.3, -0.4],
            status=Sample.Status.TRUNCATED if index in truncated else Sample.Status.COMPLETED,
        )
        for index, reward in enumerate(rewards)
    ]


def test_strict_grader_format_penalty_is_not_completed_math_diversity():
    args = make_args(reward_key="score")
    samples = make_group(
        rewards=(
            {"score": -1.0, "valid": False},
            {"score": 1.0, "valid": True},
            {"score": 1.0, "valid": True},
            {"score": 1.0, "valid": True},
        )
    )
    result = mask_truncated_and_require_completed_reward_diversity(args, samples)
    assert not result.keep
    assert result.reason == "no_completed_reward_diversity"
    samples[0].reward["valid"] = True
    assert mask_truncated_and_require_completed_reward_diversity(args, samples).keep


def convert(args, samples):
    return convert_samples_to_train_data(
        args,
        samples,
        metadata={},
        custom_convert_samples_to_train_data_func=None,
        custom_reward_post_process_func=None,
    )


@pytest.mark.parametrize("policy", POLICIES)
def test_launcher_paths_resolve(policy):
    path = f"miles.rollout.filter_hub.truncated_response_filters.{policy.__name__}"
    assert load_function(path, sync_required=True) is policy


@pytest.mark.parametrize("policy", POLICIES)
def test_completed_diversity_with_truncation_preserves_group_and_reward_contract(policy):
    args = make_args(reward_key="score")
    group = make_group(truncated=(2,))
    for sample in group:
        sample.reward = {"score": sample.reward, "acc": sample.reward == 1, "pred": sample.response}
    group[0].loss_mask = [1, 0, 1]
    group[2].loss_mask = [1, 1, 0]
    group[3].remove_sample = True
    original = deepcopy(group)
    baseline = convert(args, deepcopy(group))
    sample_ids = [id(s) for s in group]
    reward_ids = [id(s.reward) for s in group]
    mask_ids = [id(s.loss_mask) for s in group]
    args_before = deepcopy(args)

    assert apply_preput_filters(args, policy, group, ignored=True) == FilterOutput(keep=True)
    assert len(group) == 4
    assert [id(s) for s in group] == sample_ids
    assert [id(s.reward) for s in group] == reward_ids
    assert [id(s.loss_mask) for s in group] == mask_ids
    assert args == args_before
    expected = deepcopy(original)
    expected[2].remove_sample = True
    assert group == expected
    assert policy(args, group) == FilterOutput(keep=True)  # idempotent
    assert group == expected

    data = convert(args, group)
    assert data["raw_reward"] == baseline["raw_reward"] == [-1.0, 1.0, -1.0, 1.0]
    assert data["rewards"] == baseline["rewards"]
    assert data["rewards"] == pytest.approx([-0.866025, 0.866025, -0.866025, 0.866025], abs=1e-6)
    assert data["loss_masks"] == [[1, 0, 1], [1, 1, 1], [0, 0, 0], [0, 0, 0]]
    assert data["truncated"] == [0, 0, 1, 0]
    assert data["sample_indices"] == baseline["sample_indices"]
    assert data["rollout_mask_sums"] == [2, 3, 0, 0]
    assert any(reward != 0 and sum(mask) > 0 for reward, mask in zip(data["rewards"], data["loss_masks"], strict=True))


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize(
    ("rewards", "truncated", "reason"),
    [
        ((-1, 1, -1, 1), (0, 1, 2, 3), "no_effective_response_tokens"),
        ((1, 1, 1, 1), (3,), "zero_reward_std"),
        ((-1, -1, -1, -1), (), "zero_reward_std"),
        ((-1, 1, 0, 0), (0, 1), "zero_effective_advantages"),
        ((100000000, 100000001, 100000000, 100000001), (), "zero_effective_advantages"),
        ((2e38, 3e38, 2e38, 3e38), (), "nonfinite_group_advantages"),
    ],
)
def test_rejects_unusable_signal_without_mutation(policy, rewards, truncated, reason):
    group = make_group(rewards, truncated=truncated)
    before = deepcopy(group)
    assert policy(make_args(), group) == FilterOutput(keep=False, reason=reason)
    assert group == before


def test_general_policy_allows_truncated_baseline_but_proof_policy_does_not():
    args = make_args()
    group = make_group((-1, 1, 1, 1), truncated=(0,))
    before = deepcopy(group)
    assert mask_truncated_and_require_completed_reward_diversity(args, group) == FilterOutput(
        keep=False, reason="no_completed_reward_diversity"
    )
    assert group == before
    assert mask_truncated_and_check_advantages(args, group).keep
    data = convert(args, group)
    assert data["rewards"] == pytest.approx([-1.5, 0.5, 0.5, 0.5], abs=2e-6)
    assert data["loss_masks"][0] == [0, 0, 0]
    assert all(sum(mask) == 3 for mask in data["loss_masks"][1:])


def test_proof_rejects_completed_diversity_that_disappears_in_float32():
    group = make_group((100000000, 100000001, 0, 0), truncated=(2, 3))
    assert mask_truncated_and_require_completed_reward_diversity(make_args(), group) == FilterOutput(
        keep=False, reason="no_completed_reward_diversity"
    )
    assert mask_truncated_and_check_advantages(make_args(), group).keep


@pytest.mark.parametrize("inactive", ["removed", "zero-mask", "empty"])
def test_proof_diversity_counts_only_final_active_completed_samples(inactive):
    group = make_group((-1, 1, 1, -1), truncated=(3,))
    if inactive == "removed":
        group[0].remove_sample = True
    elif inactive == "zero-mask":
        group[0].loss_mask = [0, 0, 0]
    else:
        group[0].response_length = 0
        group[0].rollout_log_probs = []
    assert mask_truncated_and_require_completed_reward_diversity(make_args(), group) == FilterOutput(
        keep=False, reason="no_completed_reward_diversity"
    )
    assert mask_truncated_and_check_advantages(make_args(), group).keep


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("inactive", ["removed", "zero-mask", "empty"])
def test_no_active_completed_tokens(policy, inactive):
    group = make_group(truncated=(3,))
    for sample in group[:3]:
        if inactive == "removed":
            sample.remove_sample = True
        elif inactive == "zero-mask":
            sample.loss_mask = [0, 0, 0]
        else:
            sample.response_length = 0
            sample.rollout_log_probs = []
    assert policy(make_args(), group) == FilterOutput(keep=False, reason="no_effective_response_tokens")


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("grpo_std_normalization", [False, True])
@pytest.mark.parametrize("rewards_normalization", [False, True])
def test_standard_conversion_parity_for_normalization_options(policy, grpo_std_normalization, rewards_normalization):
    args = make_args(grpo_std_normalization=grpo_std_normalization, rewards_normalization=rewards_normalization)
    group = make_group((-1, 1, 1, 1), truncated=(3,))
    baseline = convert(args, deepcopy(group))
    predicted = _group_advantages(args, torch.tensor(baseline["raw_reward"], dtype=torch.float32, device="cpu"))
    assert predicted.tolist() == baseline["rewards"]
    assert policy(args, group).keep
    data = convert(args, group)
    assert data["rewards"] == baseline["rewards"]
    assert data["raw_reward"] == baseline["raw_reward"]
    assert data["loss_masks"] == [[1, 1, 1], [1, 1, 1], [1, 1, 1], [0, 0, 0]]


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("reward", [float("nan"), float("inf"), -float("inf"), 1e100])
def test_nonfinite_rewards_rejected_even_on_masked_truncated_samples(policy, reward):
    group = make_group(truncated=(3,))
    group[3].reward = reward
    group[3].remove_sample = True
    result = policy(make_args(), group)
    assert not result.keep
    assert result.reason in {"nonfinite_reward", "nonfinite_float32_reward"}
    assert group[3].reward is reward


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("reward", [None, "not-a-number", "1", {}, {"score": None}])
def test_missing_or_invalid_reward(policy, reward):
    group = make_group()
    group[0].reward = reward
    args = make_args(reward_key="score" if isinstance(reward, dict) else None)
    assert policy(args, group) == FilterOutput(keep=False, reason="missing_or_invalid_reward")


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_rollout_logprob_rejected_even_when_masked(policy, value):
    group = make_group(truncated=(3,))
    group[3].rollout_log_probs[0] = value
    assert policy(make_args(), group) == FilterOutput(keep=False, reason="nonfinite_rollout_log_probs")


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("mask", [[1, 1], [1, -1, 1], [1, float("nan"), 1], [1, float("inf"), 1], [0.5, 1, 1]])
def test_invalid_masks(policy, mask):
    group = make_group()
    group[0].loss_mask = mask
    result = policy(make_args(), group)
    assert not result.keep
    assert result.reason == ("invalid_sample_lengths" if len(mask) != 3 else "invalid_loss_mask")


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("status", [Sample.Status.PENDING, Sample.Status.ABORTED, Sample.Status.FAILED])
def test_unfinished_and_failed_statuses(policy, status):
    group = make_group()
    group[0].status = status
    assert policy(make_args(), group) == FilterOutput(keep=False, reason="unfinished_or_failed_sample")


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"normalize_advantages": True}, "normalize-advantages"),
        ({"advantage_estimator": "ppo"}, "advantage_estimator=grpo"),
        ({"n_samples_per_prompt": 1}, "at least two"),
        ({"custom_reward_post_process_path": "custom.reward"}, "custom_reward_post_process_path"),
        ({"custom_convert_samples_to_train_data_path": "custom.convert"}, "custom_convert_samples_to_train_data_path"),
        ({"use_opd": True}, "use_opd"),
    ],
)
def test_unsupported_configuration_fails_explicitly_without_changing_inputs(policy, overrides, message):
    args = make_args(**overrides)
    before_args = deepcopy(args)
    group = make_group(truncated=(3,))
    before_group = deepcopy(group)
    with pytest.raises(ValueError, match=message):
        policy(args, group)
    assert args == before_args
    assert group == before_group


@pytest.mark.parametrize("policy", POLICIES)
def test_group_shape_and_identity_guards(policy):
    args = make_args()
    group = make_group()
    assert policy(args, []) == FilterOutput(keep=False, reason="incomplete_prompt_group")
    assert policy(args, group[:3]) == FilterOutput(keep=False, reason="incomplete_prompt_group")
    with pytest.raises(ValueError, match="flat, single-turn"):
        policy(args, [[sample] for sample in group])
    group[0].group_index = 99
    assert policy(args, group) == FilterOutput(keep=False, reason="mixed_prompt_groups")
    group[0].group_index = group[1].group_index
    group[0].rollout_id = group[1].rollout_id = 20
    with pytest.raises(ValueError, match="distinct rollout"):
        policy(args, group)


@pytest.mark.parametrize("policy", POLICIES)
def test_missing_optional_ids_and_logprobs_are_supported(policy):
    group = make_group(truncated=(3,))
    for sample in group:
        sample.group_index = None
        sample.index = None
        sample.rollout_log_probs = None
    assert policy(make_args(), group).keep


@pytest.mark.parametrize("policy", POLICIES)
def test_all_completed_group_remains_unchanged(policy):
    group = make_group()
    before = deepcopy(group)
    assert policy(make_args(), group).keep
    assert group == before


@pytest.mark.parametrize("policy", POLICIES)
def test_empty_truncated_sample_does_not_disqualify_completed_evidence(policy):
    group = make_group(truncated=(3,))
    group[3].response_length = 0
    group[3].response = ""
    group[3].rollout_log_probs = []
    assert policy(make_args(), group).keep
    assert group[3].remove_sample
    assert len(group) == 4
    assert convert(make_args(), group)["loss_masks"][3] == []


@pytest.mark.parametrize("policy", POLICIES)
def test_boolean_binary_reward_is_supported(policy):
    group = make_group((False, True, False, True), truncated=(3,))
    assert policy(make_args(), group).keep
    assert group[0].reward is False
    assert group[1].reward is True


@pytest.mark.parametrize("policy", POLICIES)
def test_no_gpu_calls(policy, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("CPU filter must not use CUDA")

    monkeypatch.setattr(torch.cuda, "current_device", forbidden)
    monkeypatch.setattr(torch.cuda, "is_available", forbidden)
    monkeypatch.setattr(torch.Tensor, "cuda", forbidden)
    assert policy(make_args(), make_group(truncated=(3,))).keep
