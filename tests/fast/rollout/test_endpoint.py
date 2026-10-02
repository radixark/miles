from types import SimpleNamespace

from miles.rollout.endpoint import compute_rollout_concurrency, get_rollout_url


def _args(**overrides):
    values = dict(
        rollout_endpoint_url=None,
        sglang_server_concurrency=32,
        rollout_num_gpus=8,
        rollout_num_gpus_per_engine=2,
        eval_num_gpus=0,
        eval_num_gpus_per_engine=1,
        eval_uses_snapshots=False,
        async_max_concurrent_samples=None,
        rollout_batch_size=8,
        n_samples_per_prompt=4,
    )
    return SimpleNamespace(**(values | overrides))


def test_managed_fleet_concurrency_scales_with_engines():
    assert compute_rollout_concurrency(_args()) == 128


def test_snapshot_eval_reserves_connections_without_rollout_gpus():
    assert compute_rollout_concurrency(_args(rollout_num_gpus=None, eval_uses_snapshots=True)) == 32


def test_external_endpoint_uses_async_sample_bound():
    args = _args(rollout_endpoint_url="https://rollout.example", rollout_num_gpus=0, async_max_concurrent_samples=96)
    assert compute_rollout_concurrency(args) == 96


def test_external_endpoint_uses_sync_rollout_size():
    args = _args(rollout_endpoint_url="https://rollout.example", rollout_num_gpus=0)
    assert compute_rollout_concurrency(args) == 32


def test_rollout_url_uses_external_endpoint():
    args = _args(
        rollout_endpoint_url="https://rollout.example/", sglang_router_ip="127.0.0.1", sglang_router_port=30000
    )
    assert get_rollout_url(args, "/generate") == "https://rollout.example/generate"


def test_rollout_url_uses_miles_router_by_default():
    args = _args(sglang_router_ip="127.0.0.1", sglang_router_port=30000)
    assert get_rollout_url(args, "/generate") == "http://127.0.0.1:30000/generate"
