from types import ModuleType
from typing import Any

import pytest

from miles.utils.args.custom_view import ImmutableNamespace
from miles.utils.types import Sample


class TestRoutingReplayConstraint:
    @pytest.mark.parametrize("enabled", [False, True])
    async def test_launch_setting_controls_the_request_and_rejects_missing_required_replay_data(
        self, random_rollout: ModuleType, monkeypatch: pytest.MonkeyPatch, enabled: bool
    ) -> None:
        """Disabled replay accepts ordinary output while enabled replay requires metadata from SGLang."""
        payloads: list[dict[str, Any]] = []

        async def post(url: str, payload: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
            payloads.append(dict(payload))
            return {
                "meta_info": {
                    "output_token_logprobs": [[-0.2, 5]],
                    "prompt_tokens": 1,
                    "cached_tokens": 0,
                    "finish_reason": {"type": "length"},
                }
            }

        monkeypatch.setattr(random_rollout, "PROMPT_TOKEN_RANGE", (1, 1))
        monkeypatch.setattr(random_rollout, "MAX_TOKENS_RANGE", (1, 1))
        monkeypatch.setattr(random_rollout, "MAX_CONTEXT_TOKENS", 2)
        monkeypatch.setattr(random_rollout, "http_post", post)
        args = ImmutableNamespace(
            sglang_router_ip="router",
            sglang_router_port=8000,
            rollout_num_gpus_per_engine=2,
            use_rollout_routing_replay=enabled,
        )
        sample = Sample(index=3)

        if enabled:
            with pytest.raises(RuntimeError, match="missing required routed_experts"):
                await random_rollout._generate_one_random_sample(args, sample)
        else:
            assert await random_rollout._generate_one_random_sample(args, sample) is sample
            assert sample.status is Sample.Status.COMPLETED
            assert sample.response_length == 1

        [payload] = payloads
        assert payload["routed_dp_rank"] == payload["disagg_prefill_dp_rank"] == 1
        assert payload.get("return_routed_experts", False) is enabled
        assert ("routed_experts_start_len" in payload) is enabled
