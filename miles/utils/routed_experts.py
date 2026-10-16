"""Decode SGLang's routed-experts payloads for rollout routing replay (R3).

SGLang owns the wire format (``sglang.srt.state_capturer.routed_experts_codec``); the
rows arrive in the capturer's host dtype, so rollout keeps them that narrow and the
trainer widens them before padding.
"""

import numpy as np
from sglang.srt.state_capturer.routed_experts_codec import decode_routed_experts as _decode_payload


def decode_routed_experts(
    meta_info: dict, num_tokens: int, num_layers: int, *, allow_extra_tail_row: bool = False
) -> np.ndarray | None:
    """``[num_tokens, num_layers, topk]`` rows, or None if the response carries none.

    ``allow_extra_tail_row``: SGLang may also forward the final token at a stop edge;
    that row feeds no training position and is dropped.
    """
    payload = meta_info.get("routed_experts")
    if payload is None:
        return None
    rows = _decode_payload(payload)
    if allow_extra_tail_row and rows.shape[0] == num_tokens + 1:
        rows = rows[:num_tokens]
    if rows.shape[:2] != (num_tokens, num_layers):
        raise ValueError(f"routed_experts rows {rows.shape} do not match ({num_tokens}, {num_layers}, topk)")
    assert rows.size == 0 or rows.any(), (
        "routed_experts payload is all zeros: the sglang engine did not capture routed experts "
        "(topk-bypassing --moe-runner-backend such as flashinfer_trtllm?)."
    )
    return rows
