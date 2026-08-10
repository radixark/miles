from __future__ import annotations

from functools import cache

_PEAK_BF16_TFLOPS: dict[str, float] = {
    "A100": 312.0,
    "A800": 312.0,
    "H100": 989.0,
    "H200": 989.0,
    "H800": 989.0,
    "B200": 2250.0,
    "GB200": 2500.0,
}

_MATCH_ORDER: tuple[str, ...] = tuple(sorted(_PEAK_BF16_TFLOPS, key=len, reverse=True))


def peak_bf16_tflops(device_name: str) -> float | None:
    name = device_name.upper()
    for key in _MATCH_ORDER:
        if key in name:
            return _PEAK_BF16_TFLOPS[key]
    return None


@cache
def _current_device_name() -> str | None:
    import torch

    if not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_name()


def local_peak_bf16_tflops() -> float | None:
    device_name = _current_device_name()
    return peak_bf16_tflops(device_name) if device_name else None
