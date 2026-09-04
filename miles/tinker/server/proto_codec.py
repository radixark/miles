"""SDK wire <-> gateway internal language, binary half.

Requests: /forward_backward arrives as a ForwardBackwardRequest protobuf
(optionally zstd-compressed). Responses: forward_backward and sample results
encode as ForwardBackwardOutput / SampleResponse protos when the client asks
(Accept: application/x-protobuf), so logprobs travel as raw bytes instead of
JSON floats. The schema ships with the tinker pip package.
"""

import numpy as np

from miles.tinker.core.types import UserInputError
from miles.tinker.server.encoding import build_row
from tinker.proto import tinker_public_pb2 as public_pb

PROTO_CONTENT_TYPE = "application/x-protobuf"

_PROTO_DTYPE_TO_NUMPY = {
    public_pb.DTYPE_FLOAT32: np.dtype(np.float32),
    public_pb.DTYPE_INT64: np.dtype(np.int64),
    public_pb.DTYPE_INT32: np.dtype(np.int32),
}
_STOP_REASON_TO_PROTO = {
    "stop": public_pb.STOP_REASON_STOP,
    "length": public_pb.STOP_REASON_LENGTH,
}


def maybe_decompress(body: bytes, content_encoding: str | None) -> bytes:
    if content_encoding == "zstd":
        import zstandard

        return zstandard.ZstdDecompressor().decompress(body)
    return body


def decode_forward_backward_request(body: bytes) -> tuple[str, dict]:
    """ForwardBackwardRequest proto -> (kind, internal payload)."""
    message = public_pb.ForwardBackwardRequest()
    message.ParseFromString(body)

    rows = []
    for index, datum in enumerate(message.data):
        tokens: list[int] = []
        for chunk in datum.model_input:
            if chunk.WhichOneof("chunk") != "encoded_text":
                raise UserInputError(f"unsupported model_input chunk type: {chunk.WhichOneof('chunk')}")
            tokens.extend(np.frombuffer(chunk.encoded_text.tokens, dtype=np.int32).tolist())
        inputs = {name: _decode_tensor(name, tensor) for name, tensor in datum.loss_fn_inputs.items()}
        rows.append(build_row(tokens, inputs, index))

    decoded = {
        "model_id": message.model_id,
        "seq_id": message.seq_id,
        "rows": rows,
        "loss_fn": message.loss_fn,
        "loss_fn_config": dict(message.loss_fn_config),
    }
    return ("forward_only" if message.forward_only else "forward_backward"), decoded


def _decode_tensor(name: str, tensor) -> list:
    np_dtype = _PROTO_DTYPE_TO_NUMPY.get(tensor.dtype)
    if np_dtype is None:
        raise UserInputError(f"loss_fn_inputs[{name!r}]: unsupported tensor dtype {tensor.dtype}")
    encoding = tensor.WhichOneof("encoding")
    if encoding == "dense":
        return np.frombuffer(tensor.dense, dtype=np_dtype).tolist()
    if encoding == "sparse_csr":
        (length,) = tensor.shape
        dense = np.zeros(length, dtype=np_dtype)
        cols = np.frombuffer(tensor.sparse_csr.col_indices, dtype=np.int64)
        dense[cols] = np.frombuffer(tensor.sparse_csr.values, dtype=np_dtype)
        return dense.tolist()
    raise UserInputError(f"loss_fn_inputs[{name!r}]: tensor without data")


# -------- result encoding --------


def encode_forward_backward_output(result: dict) -> bytes:
    message = public_pb.ForwardBackwardOutput()
    message.loss_fn_output_type = "ArrayRecord"
    outputs = result["outputs"]
    message.metrics["loss:sum"] = float(sum(output["loss"] for output in outputs))

    record = message.loss_fn_outputs.add()
    record.num_datums = len(outputs)
    for field, arrays in (
        ("loss:sum", [np.asarray([output["loss"]], dtype=np.float32) for output in outputs]),
        ("logprobs", [np.asarray(output["logprobs"], dtype=np.float32) for output in outputs]),
    ):
        batched = record.fields[field]
        batched.dtype = public_pb.DTYPE_FLOAT32
        offsets = np.zeros(len(arrays) + 1, dtype=np.int64)
        np.cumsum([array.nbytes for array in arrays], out=offsets[1:])
        batched.offsets = offsets.tobytes()
        batched.data = b"".join(array.tobytes() for array in arrays)
        # flat per-datum arrays: empty trailing_shape decodes as shape [n]
    return message.SerializeToString()


def encode_sample_response(result: dict) -> bytes:
    message = public_pb.SampleResponse()
    for sequence in result["sequences"]:
        out = message.sequences.add()
        out.stop_reason = _STOP_REASON_TO_PROTO[sequence["stop_reason"]]
        out.tokens = np.asarray(sequence["tokens"], dtype=np.int32).tobytes()
        if sequence.get("logprobs") is not None:
            out.logprobs = np.asarray(sequence["logprobs"], dtype=np.float32).tobytes()
    if result.get("prompt_logprobs") is not None:
        message.prompt_logprobs = np.asarray(result["prompt_logprobs"], dtype=np.float32).tobytes()
    if result.get("topk_prompt_logprobs") is not None:
        topk = result["topk_prompt_logprobs"]
        token_ids = np.asarray(topk["token_ids"], dtype=np.int32)
        message.topk_prompt_logprobs.prompt_length, message.topk_prompt_logprobs.k = token_ids.shape
        message.topk_prompt_logprobs.token_ids = token_ids.tobytes()
        message.topk_prompt_logprobs.logprobs = np.asarray(topk["logprobs"], dtype=np.float32).tobytes()
    return message.SerializeToString()


PROTO_ENCODERS = {
    "forward_backward": encode_forward_backward_output,
    "forward_only": encode_forward_backward_output,
    "sample": encode_sample_response,
}
