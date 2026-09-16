import sys
from pathlib import Path

import pytest
import yaml

from miles.backends.fsdp_utils.arguments import FSDPArgs, load_fsdp_args


def _config(tmp_path: Path, **entries) -> str:
    path = tmp_path / "fsdp.yaml"
    path.write_text(yaml.safe_dump(entries))
    return str(path)


def _load(monkeypatch: pytest.MonkeyPatch, *argv: str):
    monkeypatch.setattr(sys, "argv", ["prog", *argv])
    return load_fsdp_args()


def test_config_beats_the_dataclass_default(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    assert FSDPArgs.lr != 5e-5
    args = _load(monkeypatch, "--config", _config(tmp_path, lr=5e-5))
    assert args.lr == 5e-5


def test_cli_beats_the_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    args = _load(monkeypatch, "--config", _config(tmp_path, lr=5e-5), "--lr", "7e-5")
    assert args.lr == 7e-5


def test_config_toggles_bools_both_ways(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    assert FSDPArgs.gradient_checkpointing is False
    assert FSDPArgs.keep_fp32_master is True
    args = _load(
        monkeypatch,
        "--config",
        _config(tmp_path, gradient_checkpointing=True, keep_fp32_master=False),
    )
    assert args.gradient_checkpointing is True
    assert args.keep_fp32_master is False


def test_cli_beats_the_config_for_bools(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    args = _load(
        monkeypatch,
        "--config",
        _config(tmp_path, gradient_checkpointing=False),
        "--gradient-checkpointing",
    )
    assert args.gradient_checkpointing is True


def test_a_key_the_parser_does_not_know_is_rejected(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="some_plugin_knob"):
        _load(monkeypatch, "--config", _config(tmp_path, some_plugin_knob=42))


def test_a_misspelled_key_names_the_field_it_almost_matched(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="did you mean 'weight_decay'"):
        _load(monkeypatch, "--config", _config(tmp_path, weigth_decay=0.1))


def test_a_valid_config_is_not_tripped_by_the_check(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    args = _load(monkeypatch, "--config", _config(tmp_path, lr=5e-5, weight_decay=0.1, gradient_checkpointing=True))
    assert (args.lr, args.weight_decay, args.gradient_checkpointing) == (5e-5, 0.1, True)


def test_fields_untouched_by_the_config_keep_their_defaults(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    args = _load(monkeypatch, "--config", _config(tmp_path, lr=5e-5))
    assert args.weight_decay == FSDPArgs.weight_decay
    assert args.keep_fp32_master == FSDPArgs.keep_fp32_master


def test_no_config_still_lands_on_the_dataclass_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    args = _load(monkeypatch)
    assert args.lr == FSDPArgs.lr
    assert args.weight_decay == FSDPArgs.weight_decay
