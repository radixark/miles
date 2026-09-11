import importlib.util
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[4] / "miles_plugins/megatron_bridge/nemotron_h.py"
SPEC = importlib.util.spec_from_file_location("nemotron_h_mtp_under_test", PATH)
assert SPEC is not None and SPEC.loader is not None
PLUGIN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PLUGIN)


@pytest.mark.parametrize("value", [None, "", "0", "false", "FALSE", "no", "off", " false "])
def test_mtp_is_disabled_for_false_values(monkeypatch: pytest.MonkeyPatch, value: str | None) -> None:
    if value is None:
        monkeypatch.delenv("MILES_NEMOTRONH_KEEP_MTP", raising=False)
    else:
        monkeypatch.setenv("MILES_NEMOTRONH_KEEP_MTP", value)
    assert not PLUGIN._keep_mtp_enabled()


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on", " true "])
def test_mtp_requires_explicit_true(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("MILES_NEMOTRONH_KEEP_MTP", value)
    assert PLUGIN._keep_mtp_enabled()


def test_unknown_mtp_value_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MILES_NEMOTRONH_KEEP_MTP", "enabled")
    with pytest.raises(ValueError, match="MILES_NEMOTRONH_KEEP_MTP"):
        PLUGIN._keep_mtp_enabled()
