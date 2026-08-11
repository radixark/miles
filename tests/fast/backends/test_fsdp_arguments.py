import dataclasses
import sys

import pytest

from miles.backends.fsdp_utils.arguments import FSDPArgs, parse_fsdp_cli


def _parse(monkeypatch: pytest.MonkeyPatch, *argv: str):
    monkeypatch.setattr(sys, "argv", ["prog", *argv])
    return parse_fsdp_cli()


def test_cli_defaults_match_the_dataclass(monkeypatch: pytest.MonkeyPatch) -> None:
    args = _parse(monkeypatch)
    for field in dataclasses.fields(FSDPArgs):
        assert field.default is not dataclasses.MISSING, (
            f"{field.name} declares no plain default, so parse_fsdp_cli registers "
            f"the dataclasses.MISSING sentinel as its CLI default; teach the parser "
            f"about default_factory before adding a field like this"
        )
        assert getattr(args, field.name) == field.default, (
            f"CLI default for --{field.name.replace('_', '-')} is {getattr(args, field.name)!r}, "
            f"but the dataclass declares {field.default!r}"
        )


def test_true_default_bools_can_be_turned_off(monkeypatch: pytest.MonkeyPatch) -> None:
    args = _parse(
        monkeypatch,
        "--no-fsdp-state-dict-cpu-offload",
        "--no-use-checkpoint-lr-scheduler",
        "--no-keep-fp32-master",
    )
    assert args.fsdp_state_dict_cpu_offload is False
    assert args.use_checkpoint_lr_scheduler is False
    assert args.keep_fp32_master is False


def test_false_default_bools_still_turn_on(monkeypatch: pytest.MonkeyPatch) -> None:
    args = _parse(monkeypatch, "--gradient-checkpointing", "--fp16")
    assert args.gradient_checkpointing is True
    assert args.fp16 is True


def test_every_bool_gets_both_forms(monkeypatch: pytest.MonkeyPatch) -> None:
    for field in dataclasses.fields(FSDPArgs):
        if field.type is not bool:
            continue
        flag = field.name.replace("_", "-")
        assert _parse(monkeypatch, f"--{flag}").__dict__[field.name] is True
        assert _parse(monkeypatch, f"--no-{flag}").__dict__[field.name] is False
