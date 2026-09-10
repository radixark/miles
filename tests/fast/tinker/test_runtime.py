"""Runtime translation preserves datum order, sampling parameters, and token logprobs."""

import pytest
import torch

from miles.tinker.core.types import UserInputError
from miles.tinker.runtime import (
    MilesBackend,
    _build_train_data,
    _pad_to_dp_multiple,
    _prompt_logprobs,
    _to_sequence,
    _topk_prompt_logprobs,
)


def _datum(tokens: list[int], **extra) -> dict:
    return {"tokens": tokens, "target_len": len(tokens) - 1, **extra}


class TestBuildTrainData:
    def test_datums_become_rollout_batch_keys(self):
        train_data = _build_train_data([(3, _datum([1, 2, 3])), (5, _datum([4, 5]))])
        assert train_data["tokens"] == [[1, 2, 3], [4, 5]]
        assert train_data["response_lengths"] == [2, 1]
        assert train_data["total_lengths"] == [3, 2]
        assert train_data["loss_masks"] == [[1, 1], [1]]
        assert train_data["adapter_slots"] == [3, 5]
        assert train_data["sample_indices"] == [0, 1]
        assert train_data["dynamic_global_batch_size"] == 2

    def test_optional_datum_keys_map_to_batch_names(self):
        train_data = _build_train_data(
            [(0, _datum([1, 2], weights=[1.0], advantages=[2.0], sampling_logprobs=[-0.5]))]
        )
        assert train_data["loss_weights"] == [[1.0]]
        assert train_data["advantages"] == [[2.0]]
        assert train_data["rollout_log_probs"] == [[-0.5]]


async def test_forward_backward_merges_worker_replicas():
    backend = MilesBackend(trainer=None, router_url="http://router")
    per_datum = [
        {"sample_index": 1, "loss": 2.0, "logprobs": torch.tensor([-0.2])},
        {"sample_index": 0, "loss": 1.0, "logprobs": torch.tensor([-0.1])},
    ]

    async def fake_run_batch(method, batch_id, train_data):
        return [{"per_datum": per_datum}, {"per_datum": per_datum}]  # two ranks report the same datums

    backend._run_batch = fake_run_batch
    outputs = await backend.forward_backward(1, [(0, _datum([1, 2])), (0, _datum([3, 4]))], "cross_entropy", {})
    assert outputs == [
        {"loss": 1.0, "logprobs": [pytest.approx(-0.1)]},
        {"loss": 2.0, "logprobs": [pytest.approx(-0.2)]},
    ]


class TestGenerateRequest:
    def _request(self, params: dict, **payload_extra) -> dict:
        payload = {
            "prompt_tokens": [1, 2],
            "sampling_params": params,
            "num_samples": 1,
            "prompt_logprobs": False,
            "topk_prompt_logprobs": 0,
            **payload_extra,
        }
        return MilesBackend(None, "http://router")._generate_request(payload, lora_name="m@1")

    def test_max_tokens_is_required(self):
        with pytest.raises(UserInputError, match="max_tokens"):
            self._request({})

    def test_params_map_to_sglang_names(self):
        request = self._request({"max_tokens": 8, "temperature": 0.0, "seed": 7})
        assert request["sampling_params"]["max_new_tokens"] == 8
        assert request["sampling_params"]["sampling_seed"] == 7
        assert request["lora_path"] == "m@1"
        assert request["return_logprob"] is True

    def test_stop_splits_token_ids_from_strings(self):
        by_ids = self._request({"max_tokens": 1, "stop": [7, 8]})
        by_text = self._request({"max_tokens": 1, "stop": ["\n"]})
        assert by_ids["sampling_params"]["stop_token_ids"] == [7, 8]
        assert by_text["sampling_params"]["stop"] == ["\n"]

    def test_topk_requests_prompt_logprobs(self):
        request = self._request({"max_tokens": 1}, topk_prompt_logprobs=3)
        assert request["logprob_start_len"] == 0
        assert request["top_logprobs_num"] == 3


class TestEngineResponseParsing:
    def test_to_sequence_reads_tokens_and_stop_reason(self):
        response = {
            "meta_info": {
                "output_token_logprobs": [(-0.1, 11), (-0.2, 12)],
                "finish_reason": {"type": "length"},
            }
        }
        sequence = _to_sequence(response)
        assert (sequence["tokens"], sequence["logprobs"], sequence["stop_reason"]) == (
            [11, 12],
            [-0.1, -0.2],
            "length",
        )

    def test_prompt_logprobs_pad_the_unscored_first_token(self):
        response = {"meta_info": {"input_token_logprobs": [(None, 1), (-0.5, 2)]}}
        first, second = _prompt_logprobs(response)
        assert first != first, "the unscored position must be NaN"
        assert second == -0.5

    def test_topk_pads_ragged_positions(self):
        response = {"meta_info": {"input_top_logprobs": [None, [(-0.1, 5)]]}}
        topk = _topk_prompt_logprobs(response, k=2)
        assert topk["token_ids"] == [[0, 0], [5, 0]]
        assert topk["logprobs"][1][0] == -0.1
        assert topk["logprobs"][0][0] != topk["logprobs"][0][0]


async def test_forward_only_runs_the_requested_loss():
    backend = MilesBackend(trainer=None, router_url="http://router")
    captured = {}

    async def fake_run_batch(method, batch_id, train_data):
        captured["method"] = method
        captured["loss_fn"] = train_data["loss_fn"]
        return [{"per_datum": [{"sample_index": 0, "loss": 3.0, "logprobs": torch.tensor([-0.3])}]}]

    backend._run_batch = fake_run_batch
    outputs = await backend.forward_only(1, [(0, _datum([1, 2]))], "importance_sampling", {})
    assert captured == {"method": "forward_only", "loss_fn": "importance_sampling"}
    assert outputs == [{"loss": 3.0, "logprobs": [pytest.approx(-0.3)]}]


def test_a_pinned_seed_still_gets_one_stream_per_sample():
    from miles.tinker.runtime import _with_sample_seed

    request = {"sampling_params": {"sampling_seed": 7, "temperature": 0.0}}
    assert [_with_sample_seed(request, i)["sampling_params"]["sampling_seed"] for i in range(3)] == [7, 8, 9]
    assert request["sampling_params"]["sampling_seed"] == 7
    assert "sampling_seed" not in _with_sample_seed({"sampling_params": {}}, 2)["sampling_params"]


class TestPadToDpMultiple:
    def test_a_multiple_passes_through_unchanged(self):
        slot_datums = [(0, _datum([1, 2])), (1, _datum([3, 4]))]
        assert _pad_to_dp_multiple(slot_datums, 2) is slot_datums

    def test_padding_replicates_the_last_datum_with_zero_loss_weight(self):
        slot_datums = [(0, _datum([1, 2, 3], weights=[1.0, 1.0], advantages=[2.0, 2.0]))]
        padded = _pad_to_dp_multiple(slot_datums, 4)
        assert len(padded) == 4
        for slot, filler in padded[1:]:
            assert slot == 0 and filler["tokens"] == [1, 2, 3]
            assert filler["weights"] == [0.0, 0.0] and filler["advantages"] == [0.0, 0.0]
        assert slot_datums[0][1]["weights"] == [1.0, 1.0], "the original datum must not be mutated"


async def test_forward_backward_pads_the_batch_and_drops_padding_outputs():
    backend = MilesBackend(trainer=None, router_url="http://router", dp_size=2)
    seen = {}

    async def fake_run_batch(method, batch_id, train_data):
        seen.update(train_data)
        per_datum = [
            {"sample_index": index, "loss": float(index), "logprobs": torch.tensor([-0.1])}
            for index in range(len(train_data["tokens"]))
        ]
        return [{"per_datum": per_datum}]

    backend._run_batch = fake_run_batch
    outputs = await backend.forward_backward(1, [(0, _datum([1, 2], weights=[1.0]))], "cross_entropy", {})
    assert seen["dynamic_global_batch_size"] == 2, "a singleton batch must be padded to the DP size"
    assert seen["loss_weights"] == [[1.0], [0.0]]
    assert len(outputs) == 1, "padding outputs are dropped"
