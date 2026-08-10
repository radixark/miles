from __future__ import annotations

import pytest

from miles.utils import device_flops
from miles.utils.device_flops import local_peak_bf16_tflops, peak_bf16_tflops


@pytest.mark.parametrize(
    "device_name, expected",
    [
        ("NVIDIA A100-SXM4-80GB", 312.0),
        ("NVIDIA H100 80GB HBM3", 989.0),
        ("NVIDIA H200", 989.0),
        ("NVIDIA H800", 989.0),
        ("NVIDIA B200", 2250.0),
        ("NVIDIA GB300", 2500.0),
    ],
)
def test_known_devices_resolve(device_name, expected):
    assert peak_bf16_tflops(device_name) == expected


def test_longer_key_wins_over_the_substring_it_contains():
    assert peak_bf16_tflops("NVIDIA GB200") == 2500.0
    assert peak_bf16_tflops("NVIDIA GB300") == 2500.0
    assert peak_bf16_tflops("NVIDIA B300") == 2250.0


def test_matching_is_case_insensitive():
    assert peak_bf16_tflops("nvidia h100 80gb hbm3") == 989.0


def test_unknown_device_returns_none_rather_than_a_default():
    assert peak_bf16_tflops("NVIDIA GeForce RTX 4090") is None
    assert peak_bf16_tflops("") is None


def test_local_lookup_uses_the_table(monkeypatch):
    monkeypatch.setattr(device_flops, "_current_device_name", lambda: "NVIDIA H100 80GB HBM3")
    assert local_peak_bf16_tflops() == 989.0


def test_none_without_a_device(monkeypatch):
    monkeypatch.setattr(device_flops, "_current_device_name", lambda: None)
    assert local_peak_bf16_tflops() is None
