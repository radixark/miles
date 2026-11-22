import pytest
from tests.fast.backends.sglang_utils.test_arguments import _parse_sglang_args

from miles.backends.sglang_utils.arguments import validate_args


class TestValidationConfigConsumers:
    @pytest.mark.parametrize("enabled", [False, True])
    def test_prefill_recomputation_controls_both_deterministic_inference_modes(self, enabled: bool) -> None:
        """Prefill recomputation enables both serving determinism modes together."""
        args = _parse_sglang_args([])
        args.loss_type = "policy_loss"
        args.rollout_num_gpus_per_engine = 1
        args.true_on_policy_mode = False
        args.recompute_logprobs_via_prefill = enabled
        args.use_session_server = False
        args.sglang_router_policy = None
        args.sglang_router_ip = None
        args.sglang_enable_prefill_only_deterministic_inference = False
        args.sglang_enable_deterministic_inference = False

        validate_args(args)

        assert args.sglang_enable_prefill_only_deterministic_inference is enabled
        assert args.sglang_enable_deterministic_inference is enabled

    @pytest.mark.parametrize("address,expected", [(None, None), ("127.0.0.1", "127.0.0.1"), ("::1", "[::1]")])
    def test_router_validation_brackets_ipv6_without_changing_ipv4_or_absent_addresses(
        self, address: str | None, expected: str | None
    ) -> None:
        """Router configuration produces an address usable in HTTP endpoint construction."""
        args = _parse_sglang_args([])
        args.loss_type = "policy_loss"
        args.rollout_num_gpus_per_engine = 1
        args.true_on_policy_mode = False
        args.recompute_logprobs_via_prefill = False
        args.use_session_server = False
        args.sglang_router_policy = None
        args.sglang_router_ip = address

        validate_args(args)

        assert args.sglang_router_ip == expected
