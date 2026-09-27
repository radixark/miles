"""Direct callers receive the same mode validation as CLI and YAML launches."""

from argparse import ArgumentParser, Namespace

import pytest

from miles.utils.weight_transfer import add_weight_transfer_arguments, is_broadcast_mode, validate_weight_transfer_args


@pytest.mark.parametrize(
    "mode,expected", [("broadcast", True), ("broadcast_packed", True), ("p2p", False), ("disk-delta", False)]
)
def test_broadcast_mode_classification(mode, expected):
    assert is_broadcast_mode(mode) == expected


@pytest.mark.parametrize("mode", ["broadcast", "broadcast_packed", "p2p", "disk-delta"])
def test_direct_caller_mode_is_preserved(mode):
    args = Namespace(train_backend="megatron", colocate=False, update_weight_transfer_mode=mode)
    validate_weight_transfer_args(args)
    assert args.update_weight_transfer_mode == mode


@pytest.mark.parametrize("mode", ["typo", None, True])
def test_direct_caller_unknown_mode_is_rejected(mode):
    with pytest.raises(ValueError, match="Unknown --update-weight-transfer-mode"):
        validate_weight_transfer_args(Namespace(update_weight_transfer_mode=mode))


@pytest.mark.parametrize("old_flag", [True, False])
def test_removed_boolean_cannot_silently_select_legacy_transfer(old_flag):
    with pytest.raises(ValueError, match="was replaced"):
        validate_weight_transfer_args(Namespace(update_weight_use_flattened_buckets=old_flag))


def test_fsdp_legacy_mode_is_unchanged():
    validate_weight_transfer_args(
        Namespace(train_backend="fsdp", colocate=False, update_weight_transfer_mode="broadcast")
    )


def test_cli_default_preserves_per_tensor_broadcast():
    parser = ArgumentParser()
    add_weight_transfer_arguments(parser)
    assert vars(parser.parse_args([])) == {"update_weight_transfer_mode": "broadcast"}


@pytest.mark.parametrize("mode", ["broadcast", "broadcast_packed", "p2p", "disk-delta"])
def test_cli_accepts_mode(mode):
    parser = ArgumentParser()
    add_weight_transfer_arguments(parser)
    assert parser.parse_args(["--update-weight-transfer-mode", mode]).update_weight_transfer_mode == mode


@pytest.mark.parametrize(
    "extra", [["--update-weight-transfer-mode", "typo"], ["--update-weight-use-flattened-buckets"]]
)
def test_cli_rejects_unknown_mode_and_removed_boolean(extra):
    parser = ArgumentParser()
    add_weight_transfer_arguments(parser)
    with pytest.raises(SystemExit):
        parser.parse_args(extra)
