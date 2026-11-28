import dataclasses
import sys

import pytest

from miles.backends.fsdp_utils.arguments import (
    FSDPArgs,
    host_optimizer_threads,
    parse_fsdp_cli,
    validate_cpu_offload_threads_args,
)


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


def test_cpu_offload_threads_parse_and_validate(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _parse(monkeypatch).optimizer_cpu_offload_threads is None
    args = _parse(monkeypatch, "--optimizer-cpu-offload", "--optimizer-cpu-offload-threads", "28")
    assert args.optimizer_cpu_offload_threads == 28
    validate_cpu_offload_threads_args(args)
    validate_cpu_offload_threads_args(_parse(monkeypatch))
    with pytest.raises(ValueError, match="needs --optimizer-cpu-offload"):
        validate_cpu_offload_threads_args(_parse(monkeypatch, "--optimizer-cpu-offload-threads", "28"))
    with pytest.raises(ValueError, match="0 or positive"):
        validate_cpu_offload_threads_args(
            _parse(monkeypatch, "--optimizer-cpu-offload", "--optimizer-cpu-offload-threads", "-1")
        )


def test_host_optimizer_threads_share_the_node(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without a request, each of the node's trainer ranks gets an even share of its CPUs."""
    offload = _parse(monkeypatch, "--optimizer-cpu-offload")
    assert host_optimizer_threads(offload, ranks_per_node=8, cpu_count=224) == 28
    assert host_optimizer_threads(offload, ranks_per_node=4, cpu_count=224) == 56
    # never below one thread, and a 0 or missing rank count means the whole node
    assert host_optimizer_threads(offload, ranks_per_node=64, cpu_count=16) == 1
    assert host_optimizer_threads(offload, ranks_per_node=0, cpu_count=16) == 16
    # the knob overrides the share; 0 keeps torch's default
    requested = _parse(monkeypatch, "--optimizer-cpu-offload", "--optimizer-cpu-offload-threads", "12")
    assert host_optimizer_threads(requested, ranks_per_node=8, cpu_count=224) == 12
    default = _parse(monkeypatch, "--optimizer-cpu-offload", "--optimizer-cpu-offload-threads", "0")
    assert host_optimizer_threads(default, ranks_per_node=8, cpu_count=224) is None
    # without the host optimizer there is nothing to size
    assert host_optimizer_threads(_parse(monkeypatch), ranks_per_node=8, cpu_count=224) is None
