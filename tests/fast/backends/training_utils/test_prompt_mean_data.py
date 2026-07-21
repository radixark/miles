from types import SimpleNamespace

import pytest


def _sample(miles, group_index, index, response_length, loss_mask):
    sample = miles.Sample(group_index=group_index, index=index, response_length=response_length, reward=1.0)
    sample.tokens = [0] * response_length
    sample.loss_mask = loss_mask
    sample.status = miles.Sample.Status.COMPLETED
    return sample


def _args(**overrides):
    values = dict(
        advantage_estimator="grpo",
        balance_data=False,
        global_batch_size=4,
        rewards_normalization=False,
        reward_key=None,
        use_dynamic_global_batch_size=False,
        loss_aggregation="prompt_mean",
        n_samples_per_prompt=2,
        rollout_batch_size=2,
        grpo_std_normalization=False,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _convert(miles, samples, args):
    return miles.convert_samples_to_train_data(
        args,
        samples,
        metadata={},
        custom_convert_samples_to_train_data_func=None,
        custom_reward_post_process_func=None,
    )


def test_conversion_computes_prompt_group_denominators(miles):
    samples = [
        _sample(miles, 0, 0, 3, [1, 1, 0]),
        _sample(miles, 0, 1, 3, [1, 1, 1]),
        _sample(miles, 1, 2, 4, [1, 0, 0, 0]),
        _sample(miles, 1, 3, 4, [1, 1, 1, 1]),
    ]

    train_data = _convert(miles, samples, _args())

    assert train_data["prompt_group_indices"] == [0, 0, 1, 1]
    assert train_data["prompt_mask_sums"] == [5, 5, 5, 5]


def test_conversion_rejects_incomplete_prompt_groups(miles):
    samples = [
        _sample(miles, 0, 0, 2, [1, 1]),
        _sample(miles, 1, 1, 2, [1, 1]),
    ]

    with pytest.raises(ValueError, match="complete prompt groups"):
        _convert(miles, samples, _args())


def test_dp_split_keeps_prompt_groups_whole(miles, monkeypatch):
    monkeypatch.setattr(miles.train_data_conversion.ray, "put", lambda value: value)
    samples = [
        _sample(miles, 0, 0, 3, [1, 1, 0]),
        _sample(miles, 0, 1, 3, [1, 1, 1]),
        _sample(miles, 1, 2, 4, [1, 0, 0, 0]),
        _sample(miles, 1, 3, 4, [1, 1, 1, 1]),
    ]
    args = _args()

    refs = miles.split_train_data_by_dp(args, _convert(miles, samples, args), dp_size=2)
    parts = [ref.inner for ref in refs]

    assert parts[0]["partition"] == [0, 1]
    assert parts[1]["partition"] == [2, 3]
    assert parts[0]["prompt_group_indices"] == [0, 0]
    assert parts[1]["prompt_group_indices"] == [1, 1]
    assert parts[0]["prompt_mask_sums"] == [5, 5]
    assert parts[1]["prompt_mask_sums"] == [5, 5]


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("prompt_mask_sums", [4, 4, 4], "one prompt_mask_sums entry per sample"),
        ("prompt_mask_sums", [99, 99, 4, 4], "inconsistent prompt_mask_sums"),
        ("prompt_group_indices", [0, 1, 1, 1], "complete prompt groups"),
    ],
)
def test_dp_split_rejects_malformed_prompt_metadata(miles, field, value, match):
    samples = [
        _sample(miles, 0, 0, 2, [1, 1]),
        _sample(miles, 0, 1, 2, [1, 1]),
        _sample(miles, 1, 2, 2, [1, 1]),
        _sample(miles, 1, 3, 2, [1, 1]),
    ]
    args = _args()
    train_data = _convert(miles, samples, args)
    train_data[field] = value

    with pytest.raises(ValueError, match=match):
        miles.train_data_conversion.split_train_data_by_dp_raw(args, train_data, dp_size=2)


def test_dp_split_rejects_undistributable_prompt_groups(miles, monkeypatch):
    monkeypatch.setattr(miles.train_data_conversion.ray, "put", lambda value: value)
    samples = [_sample(miles, group, 2 * group + sample, 2, [1, 1]) for group in range(3) for sample in range(2)]
    args = _args()

    with pytest.raises(ValueError, match="divisible by dp_size"):
        miles.split_train_data_by_dp(args, _convert(miles, samples, args), dp_size=2)


def test_dp_split_rejects_groups_that_straddle_steps(miles, monkeypatch):
    monkeypatch.setattr(miles.train_data_conversion.ray, "put", lambda value: value)
    samples = [_sample(miles, group, 4 * group + sample, 2, [1, 1]) for group in range(2) for sample in range(4)]
    args = _args(n_samples_per_prompt=4, global_batch_size=4)

    with pytest.raises(ValueError, match="straddle an optimizer-step boundary"):
        miles.split_train_data_by_dp(args, _convert(miles, samples, args), dp_size=2)


def test_dp_split_keeps_groups_aligned_across_multiple_steps(miles, monkeypatch):
    monkeypatch.setattr(miles.train_data_conversion.ray, "put", lambda value: value)
    samples = [_sample(miles, group, 4 * group + sample, 2, [1, 1]) for group in range(4) for sample in range(4)]
    args = _args(n_samples_per_prompt=4, global_batch_size=8)

    refs = miles.split_train_data_by_dp(args, _convert(miles, samples, args), dp_size=2)

    for part in (ref.inner for ref in refs):
        assert len(part["prompt_group_indices"]) == 8
        for step_slice in (part["prompt_group_indices"][:4], part["prompt_group_indices"][4:]):
            assert len(set(step_slice)) == 1


def test_dp_split_requires_both_prompt_fields(miles):
    with pytest.raises(ValueError, match="prompt_mask_sums"):
        miles.train_data_conversion.split_train_data_by_dp_raw(
            _args(),
            {"tokens": [[0, 0]] * 4},
            dp_size=2,
        )


def test_conversion_rejects_missing_group_index(miles):
    samples = [
        _sample(miles, None, 0, 2, [1, 1]),
        _sample(miles, None, 1, 2, [1, 0]),
    ]

    with pytest.raises(ValueError, match="group_index"):
        _convert(miles, samples, _args(rollout_batch_size=1))


def test_default_mode_omits_prompt_metadata(miles):
    samples = [
        _sample(miles, 0, 0, 2, [1, 1]),
        _sample(miles, 0, 1, 2, [1, 0]),
    ]

    train_data = _convert(miles, samples, _args(loss_aggregation="sample_mean", rollout_batch_size=1))

    assert "prompt_group_indices" not in train_data
    assert "prompt_mask_sums" not in train_data


def test_token_mean_rejects_an_optimizer_step_without_active_tokens(miles):
    data = {
        "tokens": [[0, 0]] * 4,
        "loss_masks": [[0, 0], [0, 0], [1, 0], [0, 1]],
    }

    with pytest.raises(ValueError, match="at least one active token"):
        miles.train_data_conversion.split_train_data_by_dp_raw(
            _args(loss_aggregation="token_mean", global_batch_size=2),
            data,
            dp_size=1,
        )


def test_token_mean_checks_steps_after_length_balancing(miles):
    data = {
        "tokens": [[0], [0, 0], [0], [0]],
        "loss_masks": [[0], [1, 0], [0], [1]],
    }

    with pytest.raises(ValueError, match="at least one active token"):
        miles.train_data_conversion.split_train_data_by_dp_raw(
            _args(loss_aggregation="token_mean", global_batch_size=2, balance_data=True),
            data,
            dp_size=2,
        )
