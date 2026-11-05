import sys

import pytest
import tools.convert_hf_to_int4_direct as converter


@pytest.mark.parametrize(
    ("flags", "expected"), [([], True), (["--no-is-symmetric"], False), (["--is-symmetric"], True)]
)
def test_is_symmetric_flag(monkeypatch, flags, expected):
    monkeypatch.setattr(sys, "argv", ["convert", "--model-dir", "in", "--save-dir", "out", *flags])
    assert converter.parse_args().is_symmetric is expected
