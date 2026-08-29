"""Driver-side executor for operation rounds: how one round runs; the backend answers what is runnable."""

import asyncio
import logging

from pydantic import ValidationError

from miles.ray.multi_lora.operations import BadRequest
from miles.ray.multi_lora.tinker.api_data_types import ForwardBackwardRequest, OptimStepRequest
from miles.ray.multi_lora.tinker.conversion import forward_backward_samples, pad_samples_to_multiple
from miles.ray.rollout.train_data_conversion import ROLLOUT_DATA_VALUE_SPEC, convert_samples_to_train_data
from miles.utils import object_store
from miles.utils.types import AdapterRef

logger = logging.getLogger(__name__)

TINKER_HTTP_SERVER_PATH = "miles.ray.multi_lora.tinker.http_server.TinkerHTTPServer"
OPERATION_BACKEND_PATH = "miles.ray.multi_lora.operation_backend.MultiLoRAOperationBackend"


def apply_tinker_defaults(args):
    """Serve the tinker wire protocol unless the seams are explicitly overridden."""
    assert getattr(args, "tinker_mode", False), "operation rounds require --tinker-mode"
    args.multi_lora_http_server_path = args.multi_lora_http_server_path or TINKER_HTTP_SERVER_PATH
    args.multi_lora_backend_path = args.multi_lora_backend_path or OPERATION_BACKEND_PATH
    args.delay_split_train_data_by_dp = True  # actors split per dp_rank, so the driver needs no parallel config
    return args


async def run_operation_round(args, actor_model, rollout_id: int, controller=None) -> int:
    """Driver side (controller.py idiom: actor class and driver helpers cohabit): claim and execute one round."""
    if controller is None:
        from miles.ray.multi_lora.controller import get_multi_lora_controller

        controller = get_multi_lora_controller()
    claimed = await controller.collect_operation_round.remote()
    outcomes = [await _execute_control_op(actor_model, op) for op in claimed["control_ops"]]
    samples, spans, parse_failures = _parse_data_ops(claimed["data_ops"], rollout_id)
    outcomes += parse_failures
    if samples:
        outcomes += await _train_co_batch(args, actor_model, rollout_id, samples, spans)
        rollout_id += 1
    if outcomes:
        await controller.complete_operations.remote(outcomes)
    else:
        await asyncio.sleep(args.multi_lora_idle_poll_s)
    return rollout_id


def _parse_data_ops(data_ops: list[dict], rollout_id: int) -> tuple[list, list[dict], list[dict]]:
    """Parse wire payloads into adapter-stamped samples; a bad op fails alone, never the round."""
    samples: list = []
    spans: list[dict] = []
    failures: list[dict] = []
    for op in data_ops:
        if op["kind"] != "forward_backward":
            failures.append(_failure(op, f"'{op['kind']}' is not wired in this build", "server"))
            continue
        try:
            request = ForwardBackwardRequest.model_validate(op["payload"])
            op_samples = forward_backward_samples(request.forward_backward_input)
        except (ValidationError, BadRequest) as exc:
            failures.append(_failure(op, str(exc), "user"))
            continue
        for sample in op_samples:
            sample.index = len(samples)
            sample.rollout_id = rollout_id
            sample.adapter = AdapterRef(name=op["name"], slot=op["slot"])
            samples.append(sample)
        spans.append({"op": op, "lengths": [sample.response_length for sample in op_samples]})
    return samples, spans, failures


async def _execute_control_op(actor_model, op: dict) -> dict:
    if op["kind"] != "optim_step":
        return _failure(op, f"'{op['kind']}' is not wired in this build", "server")
    try:
        request = OptimStepRequest.model_validate(op["payload"])
    except ValidationError as exc:
        return _failure(op, str(exc), "user")
    try:
        metrics = await actor_model.optim_step_adapter(op["slot"], request.adam_params.model_dump(), op["ordinal"])
        return {
            "name": op["name"],
            "ordinal": op["ordinal"],
            "ok": True,
            "result": {"metrics": _first_rank_metrics(metrics)},
        }
    except Exception:  # noqa: BLE001 - a failed control op answers its future, not the driver loop
        logger.exception(f"optim_step failed for '{op['name']}'")
        return _failure(op, "optim_step failed on the trainer", "server")


async def _train_co_batch(args, actor_model, rollout_id: int, samples: list, spans: list[dict]) -> list[dict]:
    padded = pad_samples_to_multiple(samples, getattr(args, "multi_lora_dp_size", None) or 1)
    data = convert_samples_to_train_data(
        args,
        padded,
        metadata={},
        custom_convert_samples_to_train_data_func=None,
        custom_reward_post_process_func=_zero_rewards,
    )
    ref = object_store.get_instance().put(value=data, value_spec=ROLLOUT_DATA_VALUE_SPEC)
    pack = {"sample_indices": data.get("sample_indices"), "data_ref": ref}
    try:
        metrics = await actor_model.train(rollout_id, pack)
    except Exception:  # noqa: BLE001 - a failed co-batch fails its operations, not the driver loop
        logger.exception("co-batch train failed")
        return [_failure(span["op"], "forward_backward failed on the trainer", "server") for span in spans]
    finally:
        # Inline removal keeps this module off miles.utils.data's heavy import chain.
        for stale in ref if isinstance(ref, list) else [ref]:
            object_store.get_instance().remove(stale)
    return [_forward_backward_success(span, metrics) for span in spans]


def _first_rank_metrics(result) -> dict:
    """Group broadcasts return one payload per actor; rank 0 speaks for the collective."""
    if isinstance(result, list):
        result = result[0] if result else None
    return dict(result or {})


def _forward_backward_success(span: dict, metrics) -> dict:
    # Zeroed per-token logprobs until the tinker loss function emits real ones.
    outs = [{"logprobs": {"data": [0.0] * n, "dtype": "float32", "shape": [n]}} for n in span["lengths"]]
    body = {
        "loss_fn_output_type": "ArrayRecord",
        "loss_fn_outputs": outs,
        "metrics": _first_rank_metrics(metrics) | {"num_tokens:sum": float(sum(span["lengths"]))},
    }
    op = span["op"]
    return {"name": op["name"], "ordinal": op["ordinal"], "ok": True, "result": body}


def _failure(op: dict, error: str, category: str) -> dict:
    return {"name": op["name"], "ordinal": op["ordinal"], "ok": False, "error": error, "category": category}


def _zero_rewards(args, samples):
    """Tinker computes rewards client-side; packaging still wants the channel filled."""
    return [0.0] * len(samples), [0.0] * len(samples)
