"""Datum-to-Sample conversion for the tinker training plane; validation failures are client errors."""

import copy

from miles.ray.multi_lora.operations import BadRequest
from miles.ray.multi_lora.tinker.api_data_types import LOSS_FNS, Datum, ForwardBackwardInput, TensorData
from miles.utils.types import Sample

LOSS_KIND_IDS = {name: index for index, name in enumerate(LOSS_FNS)}


def tensor_values(tensor: TensorData) -> list:
    """Dense 1-D values, int-cast for int64; per-datum tensors never arrive sparse (the SDK only sparsifies 2-D)."""
    if tensor.is_sparse or (tensor.shape is not None and len(tensor.shape) != 1):
        raise BadRequest("loss_fn_inputs tensors must be dense and 1-D per datum")
    if tensor.dtype == "int64":
        return [int(value) for value in tensor.data]
    return [float(value) for value in tensor.data]


def datum_to_sample(datum: Datum) -> Sample:
    """Full token stream = model_input + final target; targets must be the inputs shifted left by one."""
    input_tokens = datum.model_input.token_ids()
    targets_tensor = datum.loss_fn_inputs.get("target_tokens")
    if targets_tensor is None:
        raise BadRequest("datum needs loss_fn_inputs.target_tokens")
    target_tokens = tensor_values(targets_tensor)
    if not input_tokens or not target_tokens:
        raise BadRequest("datum needs non-empty model_input and target_tokens")
    if len(target_tokens) > len(input_tokens):
        raise BadRequest("datum has more target_tokens than model_input positions")
    overlap = input_tokens[len(input_tokens) - len(target_tokens) + 1 :]
    if overlap != target_tokens[:-1]:
        raise BadRequest("target_tokens must be the model_input tokens shifted left by one")
    loss_weights = _per_token_values(datum, "weights", len(target_tokens), default=1.0)
    advantages = _per_token_values(datum, "advantages", len(target_tokens), default=0.0)
    return Sample(
        tokens=input_tokens + [target_tokens[-1]],
        response_length=len(target_tokens),
        loss_mask=[1] * len(target_tokens),
        loss_weights=loss_weights,
        advantages=advantages,
        status=Sample.Status.COMPLETED,
    )


def pad_samples_to_multiple(samples: list[Sample], multiple: int) -> list[Sample]:
    """Pad with zero-contribution clones so a co-batch splits evenly across DP ranks."""
    if multiple <= 1 or not samples or len(samples) % multiple == 0:
        return samples
    template = samples[-1]
    padded = list(samples)
    while len(padded) % multiple != 0:
        clone = copy.deepcopy(template)
        clone.index = len(padded)
        clone.remove_sample = True  # convert_samples_to_train_data zeroes the loss mask for these
        clone.loss_weights = [0.0] * clone.response_length
        clone.advantages = [0.0] * clone.response_length
        padded.append(clone)
    return padded


def _per_token_values(datum: Datum, key: str, length: int, default: float) -> list[float]:
    """Channels stay homogeneous across mixed-loss co-batches, so absent inputs fill with the default."""
    tensor = datum.loss_fn_inputs.get(key)
    if tensor is None:
        return [default] * length
    values = tensor_values(tensor)
    if len(values) != length:
        raise BadRequest(f"{key} length must match target_tokens length")
    return values


def forward_backward_samples(forward_backward_input: ForwardBackwardInput) -> list[Sample]:
    """One Sample per datum, indexed in submission order; unknown loss_fn fails the whole operation."""
    if forward_backward_input.loss_fn not in LOSS_FNS:
        raise BadRequest(f"unknown loss_fn '{forward_backward_input.loss_fn}'; expected one of {LOSS_FNS}")
    samples = []
    for index, datum in enumerate(forward_backward_input.data):
        sample = datum_to_sample(datum)
        sample.index = index
        sample.loss_kind = LOSS_KIND_IDS[forward_backward_input.loss_fn]
        samples.append(sample)
    return samples
