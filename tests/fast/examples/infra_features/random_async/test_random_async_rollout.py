from argparse import Namespace
from types import ModuleType

import numpy as np
import pybase64
import pytest

from miles.backends.sglang_utils.sglang_config import SglangConfig


class TestTypedSettings:
    @pytest.mark.parametrize("topk", [1, 3])
    def test_routing_shape_uses_the_backend_owned_topk(self, random_rollout: ModuleType, topk: int) -> None:
        """Routing replay decodes expert IDs using backend configuration without a flat top-k copy."""
        args = Namespace(num_layers=2, raw_megatron=Namespace(base_args={"moe_router_topk": topk}))
        expected = np.arange(4 * topk, dtype=np.int32).reshape(2, 2, topk)
        encoded = pybase64.b64encode(expected.tobytes()).decode("ascii")

        actual = random_rollout._decode_routed_experts(args, encoded, token_count=5, start_len=2)

        np.testing.assert_array_equal(actual, expected)

    @pytest.mark.parametrize("enabled", [False, True])
    def test_metrics_registration_uses_the_sglang_owner(self, random_rollout: ModuleType, enabled: bool) -> None:
        """Metrics enablement uses typed SGLang settings without requiring a duplicate flat flag."""
        args = Namespace(
            sglang=SglangConfig(models=[], base_args={"enable_metrics": enabled}),
            sglang_router_ip="127.0.0.1",
            sglang_router_port=1234,
            rollout_num_gpus_per_engine=2,
        )

        worker = random_rollout.AsyncRandomRolloutWorker(args)

        assert (worker.metrics_reporter is not None) is enabled
        if enabled:
            assert worker.metrics_reporter.router_url == "http://127.0.0.1:1234"
            assert worker.metrics_reporter.prefill_num_gpus == 2
            assert worker.metrics_reporter.decode_num_gpus == 2
