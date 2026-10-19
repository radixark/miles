from types import SimpleNamespace

import pytest

from miles.rollout.generate_utils.generate_endpoint_utils import prepare_generate_request


def _make_args() -> SimpleNamespace:
    return SimpleNamespace(
        rollout_max_response_len=16,
        rollout_max_context_len=None,
        use_rollout_routing_replay=False,
        use_rollout_indexer_replay=False,
        use_sampling_support_replay=False,
        rollout_top_logprobs_num=0,
        lora_rank=0,
        lora_adapter_path=None,
    )


class TestPrepareGenerateRequestExtraKey:
    async def test_no_extra_key_leaves_the_payload_without_one(self):
        """Without an extra key the payload has no extra_key field at all."""
        payload, halt_status = await prepare_generate_request(_make_args(), input_ids=[1, 2], sampling_params={})

        assert halt_status is None
        assert "extra_key" not in payload

    async def test_an_extra_key_reaches_the_payload_verbatim(self):
        """The extra key is forwarded unchanged to the engine request."""
        payload, halt_status = await prepare_generate_request(
            _make_args(), input_ids=[1, 2], sampling_params={}, kv_cache_namespace="train:-:7"
        )

        assert halt_status is None
        assert payload["extra_key"] == "train:-:7"


@pytest.mark.parametrize("evaluation", [False, True])
async def test_request_hook_runs_before_sampling_validation(evaluation):
    def hook(payload, headers, *, evaluation):
        payload["sampling_params"]["top_k"] = 8

    if evaluation:
        payload, _ = await prepare_generate_request(
            _make_args(), [1, 2], {}, request_hook=hook, headers={}, evaluation=True
        )
        assert payload["sampling_params"]["top_k"] == 8
        assert "return_sampling_mask" not in payload
    else:
        with pytest.raises(ValueError, match="bounded training-request sampling"):
            await prepare_generate_request(_make_args(), [1, 2], {}, request_hook=hook, headers={})


async def test_request_hook_failure_propagates():
    async def hook(payload, headers, *, evaluation):
        raise RuntimeError("Cannot resolve request metadata")

    with pytest.raises(RuntimeError, match="Cannot resolve request metadata"):
        await prepare_generate_request(_make_args(), [1, 2], {}, request_hook=hook, headers={})


@pytest.mark.parametrize("field,value", [("stream", True), ("logprob_start_len", 2), ("lora_path", "other")])
async def test_request_hook_cannot_change_response_or_adapter_contract(field, value):
    def hook(payload, headers, *, evaluation):
        payload[field] = value

    with pytest.raises(ValueError, match=field):
        await prepare_generate_request(_make_args(), [1, 2], {}, request_hook=hook, headers={})
