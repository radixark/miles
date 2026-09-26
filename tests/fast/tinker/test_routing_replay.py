"""Captured MoE routes survive both SDK transports and fail before trainer dispatch."""

from dataclasses import replace

import numpy as np
import pytest

from tests.ci.ci_register import register_cpu_ci
from tests.fast.tinker.harness import datum, fb_payload, make_config

from miles.tinker.core.input_validation import validate_batch_payload
from miles.tinker.core.types import CommandOp, RoutingReplayConfig, UserInputError
from miles.tinker.runtime import _build_train_data, _pad_to_dp_multiple
from miles.tinker.server.encoding import decode_command
from miles.tinker.server.proto_codec import decode_forward_backward_request
from tinker import types
from tinker.proto import tinker_public_pb2 as public_pb
from tinker.proto.request_conv import forward_backward_request_to_proto

register_cpu_ci(est_time=30, suite="stage-a-cpu", labels=[])

ROUTES = {"shape": [3, 2, 2], "data": [3, 1, 0, 2, 2, 1, 3, 0, 1, 0, 2, 3]}


def _request():
    return types.ForwardBackwardRequest(
        model_id="model-x",
        seq_id=7,
        forward_backward_input=types.ForwardBackwardInput(
            data=[
                types.Datum(
                    model_input=types.ModelInput.from_ints([1, 2, 3]),
                    loss_fn_inputs={
                        "target_tokens": [2, 3, 4],
                        "weights": [0.0, 1.0, 1.0],
                        "routed_experts": types.TensorData(dtype="int64", **ROUTES),
                    },
                )
            ],
            loss_fn="cross_entropy",
        ),
    )


def _json_request():
    return {
        "model_id": "model-x",
        "seq_id": 7,
        "forward_backward_input": {
            "data": [
                {
                    "model_input": {"chunks": [{"type": "encoded_text", "tokens": [1, 2, 3]}]},
                    "loss_fn_inputs": {
                        "target_tokens": [2, 3, 4],
                        "weights": [0.0, 1.0, 1.0],
                        "routed_experts": {"dtype": "int64", **ROUTES},
                    },
                }
            ],
            "loss_fn": "cross_entropy",
        },
    }


def _config(tmp_path):
    return make_config(tmp_path, routing_replay=RoutingReplayConfig(num_layers=2, num_experts=4, topk=2))


@pytest.mark.parametrize("forward_only", [False, True])
def test_json_and_protobuf_routes_reach_the_trainer_batch_unchanged(tmp_path, forward_only):
    request = _request()
    json_request = _json_request() | {"forward_only": forward_only}
    proto_request = forward_backward_request_to_proto(request)
    proto_request.forward_only = forward_only
    json_op, json_payload = decode_command("forward_backward", json_request)
    proto_op, proto_payload = decode_forward_backward_request(proto_request.SerializeToString())
    assert json_op == proto_op == ("forward_only" if forward_only else "forward_backward")
    assert json_payload == proto_payload
    validate_batch_payload(CommandOp(json_op), json_payload, _config(tmp_path))
    assert json_payload["datums"][0]["routed_experts"] == ROUTES

    # DP fillers retain valid routing data but never contribute to the loss.
    slots = _pad_to_dp_multiple([(2, json_payload["datums"][0])], dp_size=2)
    batch = _build_train_data(slots)
    expected = np.array(ROUTES["data"], dtype=np.int32).reshape(ROUTES["shape"])
    for routes in batch["rollout_routed_experts"]:
        assert routes.dtype == np.int32
        np.testing.assert_array_equal(routes, expected)
    assert batch["adapter_slots"] == [2, 2]
    assert batch["loss_masks"] == [[1, 1, 1], [0, 0, 0]]


@pytest.mark.parametrize(
    "routes,error",
    [
        (None, "shape"),
        ({**ROUTES, "shape": [3.0, 2, 2]}, "shape"),
        ({**ROUTES, "shape": [2, 2, 2]}, "shape"),
        ({**ROUTES, "shape": [3, 1, 2]}, "shape"),
        ({**ROUTES, "shape": [3, 2, 1]}, "shape"),
        ({**ROUTES, "data": [0, 1]}, "data length"),
        ({**ROUTES, "data": [4, 1] * 6}, "integer expert IDs"),
        ({**ROUTES, "data": [-2, 1] * 6}, "integer expert IDs"),
        ({**ROUTES, "data": [True, 1] * 6}, "integer expert IDs"),
        ({**ROUTES, "data": [0.0, 1] * 6}, "integer expert IDs"),
        ({**ROUTES, "data": [-1, 1] * 6}, "distinct IDs"),
        ({**ROUTES, "data": [1, 1] * 6}, "distinct IDs"),
    ],
)
def test_invalid_routes_fail_at_admission(tmp_path, routes, error):
    payload = fb_payload("model", 1, [datum() | {"routed_experts": routes}])
    with pytest.raises(UserInputError, match=error):
        validate_batch_payload(CommandOp.FORWARD_BACKWARD, payload, _config(tmp_path))


def test_all_padding_rows_are_valid(tmp_path):
    routes = {**ROUTES, "data": [-1, -1] + ROUTES["data"][2:]}
    payload = fb_payload("model", 1, [datum() | {"routed_experts": routes}])
    validate_batch_payload(CommandOp.FORWARD_BACKWARD, payload, _config(tmp_path))


def test_capability_is_required_only_when_routes_are_supplied(tmp_path):
    config = replace(_config(tmp_path), routing_replay=None)
    payload = fb_payload("model", 1, [datum()])
    validate_batch_payload(CommandOp.FORWARD_BACKWARD, payload, config)
    payload["datums"][0]["routed_experts"] = ROUTES
    with pytest.raises(UserInputError, match="--use-rollout-routing-replay"):
        validate_batch_payload(CommandOp.FORWARD_BACKWARD, payload, config)


def test_one_request_cannot_mix_routes_and_ordinary_routing(tmp_path):
    payload = fb_payload("model", 1, [datum(), datum() | {"routed_experts": ROUTES}])
    with pytest.raises(UserInputError, match="every datum"):
        validate_batch_payload(CommandOp.FORWARD_BACKWARD, payload, _config(tmp_path))


@pytest.mark.parametrize("bad_dtype", ["float32", "int32"])
def test_wrong_wire_dtype_preserves_ordered_error_envelope(bad_dtype):
    request = _request()
    json_request = _json_request()
    json_request["forward_backward_input"]["data"][0]["loss_fn_inputs"]["routed_experts"]["dtype"] = bad_dtype
    proto_request = forward_backward_request_to_proto(request)
    proto_request.data[0].loss_fn_inputs["routed_experts"].dtype = (
        public_pb.DTYPE_FLOAT32 if bad_dtype == "float32" else public_pb.DTYPE_INT32
    )
    for _, payload in (
        decode_command("forward_backward", json_request),
        decode_forward_backward_request(proto_request.SerializeToString()),
    ):
        assert payload == {
            "model_id": "model-x",
            "seq_id": 7,
            "validation_error": "routed_experts must be a dense int64 TensorData",
        }


def test_sparse_routes_are_rejected_by_both_transports():
    request = _request()
    json_request = _json_request()
    json_request["forward_backward_input"]["data"][0]["loss_fn_inputs"]["routed_experts"]["sparse_crow_indices"] = [
        0,
        1,
    ]
    proto_request = forward_backward_request_to_proto(request)
    tensor = proto_request.data[0].loss_fn_inputs["routed_experts"]
    tensor.ClearField("dense")
    tensor.sparse_csr.SetInParent()
    for _, payload in (
        decode_command("forward_backward", json_request),
        decode_forward_backward_request(proto_request.SerializeToString()),
    ):
        assert "dense" in payload["validation_error"]
