from miles.utils.pipeline_rl_utils import apply_pipeline_rl_lag_mask


def test_pipeline_rl_lag_mask_zeroes_stale_samples():
    rollout_data = {
        "weight_version_first": ["1", "5"],
        "loss_masks": [[1, 1, 1], [1, 1]],
    }
    exceeds_count, exceeds_frac = apply_pipeline_rl_lag_mask(
        rollout_data,
        current_version=10,
        max_weight_lag=4,
    )

    assert exceeds_count == 2
    assert exceeds_frac == 1.0
    assert rollout_data["loss_masks"] == [[0, 0, 0], [0, 0]]


def test_pipeline_rl_lag_mask_falls_back_to_weight_version_last():
    rollout_data = {
        "weight_version_last": ["7"],
        "loss_masks": [[1, 1]],
    }
    exceeds_count, exceeds_frac = apply_pipeline_rl_lag_mask(
        rollout_data,
        current_version=10,
        max_weight_lag=4,
    )

    assert exceeds_count == 0
    assert exceeds_frac == 0.0
    assert rollout_data["loss_masks"] == [[1, 1]]


if __name__ == "__main__":
    test_pipeline_rl_lag_mask_zeroes_stale_samples()
    test_pipeline_rl_lag_mask_falls_back_to_weight_version_last()

