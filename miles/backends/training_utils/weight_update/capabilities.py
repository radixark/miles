"""Side-effect-free admission for optional weight transfer capabilities."""


def supports_receiver_pools(mode: str) -> bool:
    """Only protocols with independent per-pool baselines may synchronize dedicated eval."""
    return mode == "gpu-delta"
