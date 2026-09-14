from types import SimpleNamespace

from miles.rollout.endpoint import compute_rollout_concurrency, get_rollout_url


class TestRolloutConcurrency:
    def test_local_fleet_scales_per_engine_concurrency_by_engine_count(self):
        args = SimpleNamespace(
            rollout_endpoint_url=None,
            sglang_server_concurrency=32,
            rollout_num_gpus=8,
            rollout_num_gpus_per_engine=2,
            eval_num_gpus=0,
            eval_num_gpus_per_engine=1,
            async_max_concurrent_samples=None,
            rollout_batch_size=8,
            n_samples_per_prompt=4,
        )

        assert compute_rollout_concurrency(args) == 128

    def test_external_endpoint_uses_the_async_sample_bound(self):
        args = SimpleNamespace(
            rollout_endpoint_url="https://rollout.example",
            sglang_server_concurrency=32,
            rollout_num_gpus=0,
            rollout_num_gpus_per_engine=1,
            eval_num_gpus=0,
            eval_num_gpus_per_engine=1,
            async_max_concurrent_samples=96,
            rollout_batch_size=8,
            n_samples_per_prompt=4,
        )

        assert compute_rollout_concurrency(args) == 96

    def test_external_endpoint_uses_the_sync_rollout_size(self):
        args = SimpleNamespace(
            rollout_endpoint_url="https://rollout.example",
            sglang_server_concurrency=32,
            rollout_num_gpus=0,
            rollout_num_gpus_per_engine=1,
            eval_num_gpus=0,
            eval_num_gpus_per_engine=1,
            async_max_concurrent_samples=None,
            rollout_batch_size=8,
            n_samples_per_prompt=4,
        )

        assert compute_rollout_concurrency(args) == 32


class TestRolloutUrl:
    def test_uses_external_endpoint_when_configured(self):
        args = SimpleNamespace(
            rollout_endpoint_url="https://rollout.example/",
            sglang_router_ip="127.0.0.1",
            sglang_router_port=30000,
        )

        assert get_rollout_url(args, "/generate") == "https://rollout.example/generate"

    def test_uses_miles_router_by_default(self):
        args = SimpleNamespace(
            rollout_endpoint_url=None,
            sglang_router_ip="127.0.0.1",
            sglang_router_port=30000,
        )

        assert get_rollout_url(args, "/generate") == "http://127.0.0.1:30000/generate"
