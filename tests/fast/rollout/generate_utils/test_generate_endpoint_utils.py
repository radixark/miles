import asyncio
from types import SimpleNamespace

import numpy as np
import pybase64
import pytest
from tests.fast.fixtures.sglang_config_fixtures import make_sglang_config

from miles.rollout.generate_utils import generate_endpoint_utils
from miles.rollout.generate_utils.generate_endpoint_utils import compute_request_payload, update_sample_from_response
from miles.utils.output_store.plan import OUTPUT_STORE_REQUEST_KEY, StorePlanError
from miles.utils.types import Sample


def _make_args() -> SimpleNamespace:
    return SimpleNamespace(
        rollout_max_response_len=16,
        max_seq_len=None,
        use_rollout_routing_replay=False,
        use_rollout_indexer_replay=False,
        use_sampling_support_replay=False,
        rollout_top_logprobs_num=0,
        rollout_temperature=1.0,
        lora_rank=0,
        lora_adapter_path=None,
        use_output_store=False,
    )


class TestComputeRequestPayloadExtraKey:
    def test_no_extra_key_leaves_the_payload_without_one(self):
        """Without an extra key the payload has no extra_key field at all."""
        payload, halt_status = compute_request_payload(_make_args(), input_ids=[1, 2], sampling_params={})

        assert halt_status is None
        assert "extra_key" not in payload

    def test_an_extra_key_reaches_the_payload_verbatim(self):
        """The extra key is forwarded unchanged to the engine request."""
        payload, halt_status = compute_request_payload(
            _make_args(), input_ids=[1, 2], sampling_params={}, kv_cache_namespace="train:-:7"
        )

        assert halt_status is None
        assert payload["extra_key"] == "train:-:7"


class TestComputeRequestPayloadOutputStore:
    def make_args(self, **overrides) -> SimpleNamespace:
        return SimpleNamespace(**{**vars(_make_args()), "use_output_store": True, **overrides})

    def test_a_training_request_for_replay_asks_for_the_store(self):
        payload, _ = compute_request_payload(
            self.make_args(use_rollout_routing_replay=True), input_ids=[1, 2], sampling_params={}
        )

        assert payload[OUTPUT_STORE_REQUEST_KEY] is True

    @pytest.mark.parametrize(
        ("overrides", "evaluation"),
        [({}, True), ({"use_rollout_routing_replay": False}, False), ({"use_output_store": False}, False)],
        ids=["evaluation", "no-replay-field", "switch-off"],
    )
    def test_no_store_for_evaluation_without_replay_or_without_the_switch(self, overrides, evaluation):
        args = self.make_args(**{"use_rollout_routing_replay": True, **overrides})

        payload, _ = compute_request_payload(args, input_ids=[1, 2], sampling_params={}, evaluation=evaluation)

        assert OUTPUT_STORE_REQUEST_KEY not in payload

    def test_a_multimodal_prompt_is_refused(self):
        with pytest.raises(StorePlanError, match="multimodal"):
            compute_request_payload(
                self.make_args(use_rollout_routing_replay=True),
                input_ids=[1, 2],
                sampling_params={},
                multimodal_inputs={"images": ["/shared/media/image.png"]},
            )


def test_routed_experts_cover_the_prompt_the_engine_served():
    layers, topk, response = 2, 3, [7, 8]
    sample = Sample(prompt="", tokens=[1, -101, 2])
    served_prompt_tokens = 6  # the placeholder became a 4-token image run
    rows = np.arange((served_prompt_tokens + len(response) - 1) * layers * topk, dtype=np.int32)
    output = {
        "text": "",
        "meta_info": {
            "output_token_logprobs": [(-0.5, t, None) for t in response],
            "finish_reason": {"type": "stop"},
            "prompt_tokens": served_prompt_tokens,
            "completion_tokens": len(response),
            "routed_experts": pybase64.b64encode(rows.tobytes()).decode(),
        },
    }
    args = SimpleNamespace(
        rollout_top_logprobs_num=0, sglang=make_sglang_config(speculative_algorithm=None), num_layers=layers
    )

    asyncio.run(update_sample_from_response(args, sample, {"input_ids": sample.tokens}, output))

    assert sample.rollout_routed_experts.shape == (served_prompt_tokens + len(response) - 1, layers, topk)
