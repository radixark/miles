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
        default = field.default_factory() if field.default_factory is not dataclasses.MISSING else field.default
        assert default is not dataclasses.MISSING, (
            f"{field.name} declares no default, so parse_fsdp_cli registers "
            f"the dataclasses.MISSING sentinel as its CLI default"
        )
        assert getattr(args, field.name) == default, (
            f"CLI default for --{field.name.replace('_', '-')} is {getattr(args, field.name)!r}, "
            f"but the dataclass declares {default!r}"
        )


def test_list_fields_take_several_values(monkeypatch: pytest.MonkeyPatch) -> None:
    """A list field such as --profile-ranks takes one or more space-separated values, as Megatron's parser does."""
    assert _parse(monkeypatch, "--profile-ranks", "0", "8").profile_ranks == [0, 8]
    with pytest.raises(SystemExit):
        _parse(monkeypatch, "--profile-ranks")


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
