from types import SimpleNamespace

import pytest
from tests.fast.fixtures.args_fixtures import sglang_router_args

from miles.rollout.generate_utils import generate_endpoint_utils
from miles.rollout.generate_utils.generate_endpoint_utils import compute_request_payload, stamp_prompt_routing_key
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

    def test_a_multimodal_prompt_is_refused(self, monkeypatch):
        monkeypatch.setattr(generate_endpoint_utils, "encode_image_for_rollout_engine", lambda image: "image")
        with pytest.raises(StorePlanError, match="multimodal"):
            compute_request_payload(
                self.make_args(use_rollout_routing_replay=True),
                input_ids=[1, 2],
                sampling_params={},
                multimodal_inputs={"images": [object()]},
            )


def _make_router_args(policy: str, assignment_mode: str = "random") -> SimpleNamespace:
    router_args = sglang_router_args(policy)
    router_args["router_args"]["assignment_mode"] = assignment_mode
    return SimpleNamespace(
        **router_args, sglang=SimpleNamespace(models=[SimpleNamespace(has_pd_disaggregation=False)])
    )


class TestStampPromptRoutingKey:
    def test_a_load_placing_router_keys_a_prompts_samples_together(self):
        """`manual min_load` lands the new key on the least-loaded engine, so sharing it prefills the prompt once."""
        samples = [Sample(prompt="p") for _ in range(3)]

        stamp_prompt_routing_key(_make_router_args("manual", "min_load"), samples)

        keys = {sample.routing_key for sample in samples}
        assert len(keys) == 1 and None not in keys

    def test_a_hashing_router_keys_every_sample_on_its_own(self):
        """A shared key under consistent hashing would pin a prompt's samples to one rank and idle the rest."""
        samples = [Sample(prompt="p") for _ in range(3)]

        stamp_prompt_routing_key(_make_router_args("consistent_hashing"), samples)

        keys = {sample.routing_key for sample in samples}
        assert len(keys) == 3 and None not in keys

    def test_each_prompt_gets_its_own_key(self):
        """Different prompts stay free to spread across engines."""
        first, second = [Sample(prompt="p")], [Sample(prompt="q")]
        args = _make_router_args("manual", "min_load")

        stamp_prompt_routing_key(args, first)
        stamp_prompt_routing_key(args, second)

        assert first[0].routing_key != second[0].routing_key

    def test_a_sample_that_already_carries_a_key_keeps_it(self):
        """A re-sampled or derived sample keeps the engine it is bound to; only keyless siblings join the new key."""
        kept, fresh = Sample(prompt="p", routing_key="bound"), Sample(prompt="p")

        stamp_prompt_routing_key(_make_router_args("manual", "min_load"), [kept, fresh])

        assert kept.routing_key == "bound"
        assert fresh.routing_key not in (None, "bound")

    def test_a_router_that_ignores_keys_leaves_the_samples_unkeyed(self):
        """Under a load or cache policy no request carries a routing header."""
        samples = [Sample(prompt="p") for _ in range(2)]

        stamp_prompt_routing_key(_make_router_args("round_robin"), samples)

        assert [sample.routing_key for sample in samples] == [None, None]
