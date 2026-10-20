"""--use-output-store: launch checks, the engines' store config, and which requests use the store."""

import json
from argparse import Namespace

import pytest

from miles.backends.sglang_utils.arguments import _SKIPPED_SERVER_ARGS
from miles.rollout.generate_utils.generate_endpoint_utils import compute_request_payload
from miles.utils.output_store_config import (
    SGLANG_OUTPUT_STORE_FIELDS,
    compute_sglang_output_store_args,
    validate_output_store_args,
)
from miles.utils.replay_index import set_outputs_via_store


def _args(**overrides) -> Namespace:
    values = dict(
        use_output_store=True,
        object_store_backend="mooncake",
        pause_generation_mode="in_place",
        ft_components=["rollout"],
        rollout_external=False,
        cluster_backend="ray",
        mooncake_store_init_kwargs={"master_server_address": "10.0.0.1:50051", "protocol": "tcp"},
        mooncake_replica_num=1,
        output_store_local_buffer_size="512mb",
        run_uuid="run-abc",
    )
    return Namespace(**(values | overrides))


def test_valid_launch_passes():
    validate_output_store_args(_args(), multi_policy=False, pd_disaggregation=False)
    validate_output_store_args(
        _args(use_output_store=False, object_store_backend="ray"), multi_policy=True, pd_disaggregation=True
    )


@pytest.mark.parametrize("use_output_store", [False, True])
def test_the_store_turns_on_delayed_assembly(use_output_store):
    # Only the train ranks can read store objects, so they assemble the replay data.
    args = _args(use_output_store=use_output_store, enable_delay_token_replay_assembly=False)
    validate_output_store_args(args, multi_policy=False, pd_disaggregation=False)
    assert args.enable_delay_token_replay_assembly is use_output_store


@pytest.mark.parametrize(
    "overrides,flags,match",
    [
        (dict(object_store_backend="ray"), {}, "--object-store-backend must be mooncake"),
        ({}, dict(pd_disaggregation=True), "PD disaggregation is not supported"),
        (dict(pause_generation_mode="retract"), {}, "--pause-generation-mode must be in_place: .*got retract"),
        (dict(pause_generation_mode="abort"), {}, "--pause-generation-mode must be in_place"),
        (dict(ft_components=["train"]), {}, "train fault tolerance"),
        ({}, dict(multi_policy=True), "multi-policy"),
        (dict(rollout_external=True), {}, "external rollout engines"),
        (dict(cluster_backend="kubernetes"), {}, "--cluster-backend ray"),
    ],
)
def test_unsupported_launch_fails_with_the_reason(overrides, flags, match):
    with pytest.raises(ValueError, match=match):
        validate_output_store_args(_args(**overrides), **({"multi_policy": False, "pd_disaggregation": False} | flags))


def test_all_violations_are_reported_at_once():
    with pytest.raises(ValueError, match="mooncake.*in_place"):
        validate_output_store_args(
            _args(object_store_backend="ray", pause_generation_mode="retract"),
            multi_policy=False,
            pd_disaggregation=False,
        )


def test_engine_config_comes_from_the_runs_mooncake():
    server_args = compute_sglang_output_store_args(_args(mooncake_replica_num=2), local_hostname="10.0.0.7")
    assert server_args["output_store_backend"] == "mooncake"
    assert json.loads(server_args["output_store_backend_extra_config"]) == {
        "master_server_address": "10.0.0.1:50051",
        "local_hostname": "10.0.0.7",
        "local_buffer_size": "512mb",
        "key_prefix": "miles-output-store/run-abc",
        "protocol": "tcp",
        "metadata_server": "P2PHANDSHAKE",
        "device_name": "",
        "replica_num": 2,
    }


def test_sglang_store_flags_are_owned_by_miles():
    assert set(SGLANG_OUTPUT_STORE_FIELDS) <= set(_SKIPPED_SERVER_ARGS)


@pytest.mark.parametrize(
    "request_body,use_output_store,evaluation,expected",
    [
        ({"return_routed_experts": True}, True, False, True),
        ({"return_sampling_mask": True}, True, False, True),
        ({"return_routed_experts": True}, True, True, False),
        ({"return_routed_experts": True}, False, False, False),
        ({"return_outputs_via_store": True}, True, False, False),
    ],
)
def test_only_training_requests_with_replay_outputs_use_the_store(
    request_body, use_output_store, evaluation, expected
):
    set_outputs_via_store(request_body, use_output_store=use_output_store, evaluation=evaluation)
    assert request_body.get("return_outputs_via_store", False) is expected


@pytest.mark.parametrize("evaluation", [False, True])
def test_generate_payload_requests_the_store_for_training_only(evaluation):
    args = Namespace(
        rollout_max_response_len=4,
        rollout_max_context_len=None,
        use_sampling_support_replay=False,
        rollout_temperature=1.0,
        use_rollout_routing_replay=True,
        use_rollout_indexer_replay=False,
        rollout_top_logprobs_num=0,
        use_output_store=True,
        lora_rank=0,
        lora_adapter_path=None,
    )
    payload, _ = compute_request_payload(
        args, input_ids=[1, 2], sampling_params={"temperature": 1.0}, evaluation=evaluation
    )
    # Evaluation still gets its R3 inline, as before the store existed.
    assert payload["return_routed_experts"] is True
    assert payload.get("return_outputs_via_store", False) is not evaluation
