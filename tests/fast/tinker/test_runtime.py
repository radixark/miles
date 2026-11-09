"""Runtime translation preserves datum order, sampling parameters, and token logprobs."""

import httpx
import pytest
import torch

from miles.backends.training_utils.checkpoint.adapter.commit import tenant_digest as library_digest
from miles.tinker.core.types import UserInputError
from miles.tinker.core.utils import tenant_digest as gateway_digest
from miles.tinker.runtime import (
    MilesBackend,
    _build_train_data,
    _pad_to_dp_multiple,
    _prompt_logprobs,
    _slot_failure,
    _to_sequence,
    _topk_prompt_logprobs,
    classify_lora_groups,
)
from miles.tinker.server.proto_codec import encode_sample_response
from miles.utils.lora.hf_lora_targets import get_hf_lora_targets
from tinker import types as tinker_types
from tinker.proto.response_conv import deserialize_sample_response


async def test_sampling_preserves_engine_client_error_details(monkeypatch):
    async def rejected(*args):
        response = httpx.Response(
            400, text="prompt is too long", request=httpx.Request("POST", "http://router/generate")
        )
        response.raise_for_status()

    monkeypatch.setattr("miles.tinker.runtime.post", rejected)
    payload = {
        "prompt_tokens": [1],
        "sampling_params": tinker_types.SamplingParams(max_tokens=1).model_dump(),
        "num_samples": 1,
        "prompt_logprobs": False,
        "topk_prompt_logprobs": 0,
    }
    with pytest.raises(UserInputError, match="prompt is too long"):
        await MilesBackend(None, "http://router", lora_layout=None).sample(payload, None)


def _datum(tokens: list[int], **extra) -> dict:
    return {"tokens": tokens, "target_len": len(tokens) - 1, "target_tokens": tokens[1:], **extra}


class TestBuildTrainData:
    def test_datums_become_rollout_batch_keys(self):
        train_data = _build_train_data([(3, _datum([1, 2, 3])), (5, _datum([4, 5]))])
        assert train_data["tokens"] == [[1, 2, 3], [4, 5]]
        assert train_data["target_tokens"] == [[2, 3], [5]]
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
    backend = MilesBackend(trainer=None, router_url="http://router", lora_layout=None)
    per_datum = [
        {"sample_index": 1, "loss": 2.0, "logprobs": torch.tensor([-0.2])},
        {"sample_index": 0, "loss": 1.0, "logprobs": torch.tensor([-0.1])},
    ]

    async def fake_call_trainer(method, batch_id, train_data):
        return [{"per_datum": per_datum}, {"per_datum": per_datum}]  # two ranks report the same datums

    backend._call_trainer = fake_call_trainer
    outputs = await backend.forward_backward(1, [(0, _datum([1, 2])), (0, _datum([3, 4]))], "cross_entropy", {})
    assert outputs == [
        {"loss": 1.0, "logprobs": [pytest.approx(-0.1)]},
        {"loss": 2.0, "logprobs": [pytest.approx(-0.2)]},
    ]


class TestGenerateRequest:
    def _request(self, params: dict, **payload_extra) -> dict:
        payload = {
            "prompt_tokens": [1, 2],
            # the gateway forwards every SDK field (decode_sample_request)
            "sampling_params": tinker_types.SamplingParams(**params).model_dump(),
            "num_samples": 1,
            "prompt_logprobs": False,
            "topk_prompt_logprobs": 0,
            **payload_extra,
        }
        return MilesBackend(None, "http://router", lora_layout=None)._generate_request(payload, lora_name="m@1")

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

    def test_an_empty_stop_list_disables_eos(self):
        request = self._request({"max_tokens": 1, "stop": []})
        assert request["sampling_params"]["ignore_eos"] is True
        assert "stop" not in request["sampling_params"] and "stop_token_ids" not in request["sampling_params"]
        default = self._request({"max_tokens": 1})
        assert "ignore_eos" not in default["sampling_params"], "stop=None keeps the default EOS behavior"

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
        result = {"sequences": [], "topk_prompt_logprobs": topk}
        decoded = deserialize_sample_response(encode_sample_response(result)).topk_prompt_logprobs
        assert decoded[0] is None
        assert len(decoded[1]) == 1
        token_id, logprob = decoded[1][0]
        assert token_id == 5
        assert logprob == pytest.approx(-0.1)


async def test_forward_only_runs_the_requested_loss():
    backend = MilesBackend(trainer=None, router_url="http://router", lora_layout=None)
    captured = {}

    async def fake_call_trainer(method, batch_id, train_data):
        captured["method"] = method
        captured["loss_fn"] = train_data["loss_fn"]
        return [{"per_datum": [{"sample_index": 0, "loss": 3.0, "logprobs": torch.tensor([-0.3])}]}]

    backend._call_trainer = fake_call_trainer
    outputs = await backend.forward_only(1, [(0, _datum([1, 2]))], "importance_sampling", {})
    assert captured == {"method": "forward_only", "loss_fn": "importance_sampling"}
    assert outputs == [{"loss": 3.0, "logprobs": [pytest.approx(-0.3)]}]


def test_a_pinned_seed_still_gets_one_model_queue_per_sample():
    from miles.tinker.runtime import _with_sample_seed

    request = {"sampling_params": {"sampling_seed": 7, "temperature": 0.0}}
    assert [_with_sample_seed(request, i)["sampling_params"]["sampling_seed"] for i in range(3)] == [7, 8, 9]
    assert request["sampling_params"]["sampling_seed"] == 7
    assert "sampling_seed" not in _with_sample_seed({"sampling_params": {}}, 2)["sampling_params"]


class TestPadToDpMultiple:
    def test_padding_replicates_the_last_datum_with_a_padding_marker(self):
        slot_datums = [(0, _datum([1, 2, 3], weights=[1.0, 1.0], advantages=[2.0, 2.0]))]
        padded = _pad_to_dp_multiple(slot_datums, 4)
        assert len(padded) == 4
        for slot, filler in padded[1:]:
            assert slot == 0 and filler["tokens"] == [1, 2, 3] and filler["padding"]
        assert "padding" not in slot_datums[0][1], "the original datum must not be mutated"


async def test_forward_backward_pads_the_batch_and_drops_padding_outputs():
    backend = MilesBackend(trainer=None, router_url="http://router", lora_layout=None, dp_size=2)
    seen = {}

    async def fake_call_trainer(method, batch_id, train_data):
        seen.update(train_data)
        per_datum = [
            {"sample_index": index, "loss": float(index), "logprobs": torch.tensor([-0.1])}
            for index in range(len(train_data["tokens"]))
        ]
        return [{"per_datum": per_datum}]

    backend._call_trainer = fake_call_trainer
    outputs = await backend.forward_backward(1, [(0, _datum([1, 2], weights=[1.0]))], "cross_entropy", {})
    assert seen["dynamic_global_batch_size"] == 2, "a singleton batch must be padded to the DP size"
    assert seen["loss_masks"] == [[1], [0]], "the zero mask removes padding from every loss term"
    assert len(outputs) == 1, "padding outputs are dropped"


def test_an_actor_error_verdict_survives_runtime_translation():
    failure = {"error": "bad shard"}
    assert _slot_failure([None, failure]) is failure


def test_an_aborted_sample_fails_instead_of_passing_as_a_stop():
    """A truncated sequence fed to RL as a completed sample corrupts training data silently."""
    response = {"meta_info": {"output_token_logprobs": [(-0.1, 11)], "finish_reason": {"type": "abort"}}}
    assert "abort" in _to_sequence(response)["error"]


@pytest.mark.parametrize("tenant", ["tenant-a", "", "ключ/with spaces"])
def test_the_gateway_digests_a_tenant_as_the_library_checks_it(tenant):
    """Core (stdlib-only) writes the digest the library's ownership check recomputes; the two must agree."""
    assert gateway_digest(tenant) == library_digest(tenant)


def test_lora_groups_come_from_the_recorded_targets():
    qwen3 = get_hf_lora_targets({"model_type": "qwen3", "num_hidden_layers": 1})
    assert classify_lora_groups(("model.layers.0.self_attn.q_proj", "lm_head"), qwen3) == {
        "train_attn": True,
        "train_mlp": False,
        "train_unembed": True,
    }
    with pytest.raises(UserInputError, match="in no LoRA group"):
        classify_lora_groups(("model.layers.0.input_layernorm",), qwen3)


@pytest.mark.parametrize(
    "config",
    [
        {"model_type": "mimo_v2", "num_hidden_layers": 2, "moe_layer_freq": [0, 1]},
        {
            "model_type": "kimi_k2",
            "num_hidden_layers": 2,
            "n_routed_experts": 4,
            "first_k_dense_replace": 1,
            "n_shared_experts": 0,
            "q_lora_rank": None,
        },
    ],
    ids=["mimo_v2", "kimi_k2"],
)
def test_a_per_expert_familys_stacked_expert_targets_count_as_mlp(config):
    # the layout names routed experts per expert (experts.*.down_proj); a checkpoint records them stacked
    layout = get_hf_lora_targets(config)
    flags = classify_lora_groups(("model.layers.1.mlp.experts.down_proj",), layout)
    assert flags == {"train_attn": False, "train_mlp": True, "train_unembed": False}


def test_a_fused_modules_components_count_as_its_group():
    gdn = get_hf_lora_targets(
        {
            "model_type": "qwen3_5_text",
            "num_hidden_layers": 1,
            "layer_types": ["linear_attention"],
            "linear_num_key_heads": 1,
            "linear_key_head_dim": 2,
            "linear_num_value_heads": 1,
            "linear_value_head_dim": 3,
        }
    )
    flags = classify_lora_groups(("model.layers.0.linear_attn.in_proj_q",), gdn)
    assert flags == {"train_attn": True, "train_mlp": False, "train_unembed": False}
