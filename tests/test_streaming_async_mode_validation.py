from types import SimpleNamespace

from miles.utils.streaming_async_mode import validate_streaming_async_args


def _args(**overrides):
    base = dict(
        streaming_async=True,
        pipeline_rl=True,
        pipeline_weight_update_interval=1,
        pipeline_max_weight_lag=4,
        balance_data=False,
        advantage_estimator="ppo",
        prefill_num_servers=None,
        use_miles_router=False,
        use_tis=True,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_streaming_allows_prefill_num_servers():
    args = _args(prefill_num_servers=1)
    validate_streaming_async_args(args)


def test_streaming_forbids_miles_router():
    args = _args(use_miles_router=True)
    try:
        validate_streaming_async_args(args)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for streaming_async + use_miles_router")


def test_streaming_forbids_balance_data_for_group_estimators():
    args = _args(advantage_estimator="grpo", balance_data=True)
    try:
        validate_streaming_async_args(args)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for group estimator + balance_data under streaming")


def test_streaming_requires_pipeline_rl():
    args = _args(pipeline_rl=False)
    try:
        validate_streaming_async_args(args)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for streaming_async without pipeline_rl")


def test_pipeline_rl_requires_streaming():
    args = _args(streaming_async=False, pipeline_rl=True)
    try:
        validate_streaming_async_args(args)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for pipeline_rl without streaming_async")


def test_invalid_pipeline_weight_update_interval_rejected():
    args = _args(pipeline_weight_update_interval=0)
    try:
        validate_streaming_async_args(args)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for pipeline_weight_update_interval < 1")


def test_negative_pipeline_max_weight_lag_rejected():
    args = _args(pipeline_max_weight_lag=-1)
    try:
        validate_streaming_async_args(args)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for pipeline_max_weight_lag < 0")


if __name__ == "__main__":
    test_streaming_allows_prefill_num_servers()
    test_streaming_forbids_miles_router()
    test_streaming_forbids_balance_data_for_group_estimators()
    test_streaming_requires_pipeline_rl()
    test_pipeline_rl_requires_streaming()
    test_invalid_pipeline_weight_update_interval_rejected()
    test_negative_pipeline_max_weight_lag_rejected()
