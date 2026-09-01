"""Sampling wire codec: official tinker request shapes to plain dicts, and results to proto bytes."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from tinker.proto import tinker_public_pb2 as public_pb

_STOP_REASON_TO_PROTO = {
    "stop": public_pb.STOP_REASON_STOP,
    "length": public_pb.STOP_REASON_LENGTH,
}


def prompt_tokens_from_wire(prompt: dict[str, Any]) -> list[int]:
    """Flatten a wire ModelInput into token ids; MVP accepts encoded_text chunks only."""
    tokens: list[int] = []
    for chunk in prompt.get("chunks", []):
        if chunk.get("type") != "encoded_text":
            raise ValueError(f"unsupported prompt chunk type {chunk.get('type')!r}")
        tokens.extend(chunk["tokens"])
    return tokens


def sglang_sampling_params(wire: dict[str, Any]) -> dict[str, Any]:
    """Official SamplingParams wire dict to an sglang sampling_params dict."""
    converted: dict[str, Any] = {
        "max_new_tokens": wire.get("max_tokens"),
        "temperature": wire.get("temperature", 0.0),
        "top_p": wire.get("top_p", 1.0),
        "top_k": wire.get("top_k", -1),
        "skip_special_tokens": False,
        "no_stop_trim": True,
        "spaces_between_special_tokens": False,
    }
    if wire.get("stop") is not None:
        converted["stop"] = wire["stop"]
    if wire.get("seed") is not None:
        converted["sampling_seed"] = wire["seed"]
    return converted


def sample_response_proto_bytes(sequences: list[dict[str, Any]], prompt_logprobs: list[float | None] | None) -> bytes:
    """Serialize sampled sequences into the proto SampleResponse the 0.26.1 SDK requires."""
    msg = public_pb.SampleResponse()
    for sequence in sequences:
        out = msg.sequences.add()
        out.stop_reason = _STOP_REASON_TO_PROTO[sequence["stop_reason"]]
        out.tokens = np.asarray(sequence["tokens"], dtype=np.int32).tobytes()
        out.logprobs = np.asarray(sequence["logprobs"], dtype=np.float32).tobytes()
    if prompt_logprobs is not None:
        filled = [math.nan if lp is None else lp for lp in prompt_logprobs]
        msg.prompt_logprobs = np.asarray(filled, dtype=np.float32).tobytes()
    msg.prompt_cache_hit_tokens = 0
    return msg.SerializeToString()
