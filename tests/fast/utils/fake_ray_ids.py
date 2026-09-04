from __future__ import annotations

_NODE_ID_HEX_DIGITS = 56


def fake_ray_node_id(index: int) -> str:
    """Ray validates node ids as hex, so a readable "node-0" is rejected before scheduling."""
    return f"{index:0{_NODE_ID_HEX_DIGITS}x}"
