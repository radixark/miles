import argparse
from pathlib import Path
from unittest.mock import create_autospec

import pytest
from huggingface_hub import HfApi

from miles.utils import hub


def parse_args(extra=()):
    parser = argparse.ArgumentParser()
    hub.add_hub_arguments(parser)
    args = parser.parse_args(extra)
    args.train_backend = "megatron"
    args.save = "/trainer"
    args.save_hf = "/hf/{rollout_id}"
    args.save_interval = 20
    args.debug_rollout_only = False
    args.dumper_enable = False
    return args


def test_disabled_by_default():
    args = parse_args()
    hub.validate_hub_args(args)
    assert not args.push_to_hub
    assert not args.hub_private_repo
    assert args.hub_strategy == "every_save"


@pytest.mark.parametrize("strategy", ["end", "every_save"])
def test_accepts_native_options(strategy):
    args = parse_args(
        ["--push-to-hub", "--hub-model-id", "user/model", "--hub-private-repo", "--hub-strategy", strategy]
    )
    hub.validate_hub_args(args)
    assert args.hub_private_repo
    assert args.hub_strategy == strategy


@pytest.mark.parametrize(
    "field,value",
    [
        ("hub_model_id", None),
        ("hub_model_id", "invalid/repo/id"),
        ("train_backend", "fsdp"),
        ("save", None),
        ("save_hf", None),
        ("save_interval", None),
        ("save_interval", 0),
        ("save_interval", -1),
        ("debug_rollout_only", True),
        ("dumper_enable", True),
    ],
)
def test_rejects_invalid_configuration(field, value):
    args = parse_args(["--push-to-hub", "--hub-model-id", "user/model"])
    setattr(args, field, value)
    with pytest.raises(ValueError):
        hub.validate_hub_args(args)


@pytest.mark.parametrize(
    "extra", [["--hub-model-id", "user/model"], ["--hub-private-repo"], ["--hub-strategy", "end"]]
)
def test_rejects_inactive_hub_options(extra):
    with pytest.raises(ValueError, match="require --push-to-hub"):
        hub.validate_hub_args(parse_args(extra))


def test_rejects_unsupported_strategy():
    with pytest.raises(SystemExit):
        parse_args(["--hub-strategy", "all_checkpoints"])


@pytest.fixture
def api(monkeypatch):
    api = create_autospec(HfApi, instance=True)
    monkeypatch.setattr(hub, "HfApi", lambda: api)
    return api


@pytest.fixture
def checkpoint(tmp_path):
    (tmp_path / ".complete").touch()
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    return str(tmp_path)


@pytest.mark.parametrize(
    "strategy,is_final,expected",
    [
        ("end", False, 0),
        ("end", True, 1),
        ("every_save", False, 1),
        ("every_save", True, 1),
    ],
)
def test_upload_schedule(api, checkpoint, strategy, is_final, expected):
    hub.push_model_to_hub(
        checkpoint_dir=checkpoint,
        repo_id="user/model",
        private=True,
        strategy=strategy,
        rollout_id=19,
        is_final=is_final,
    )
    assert api.create_repo.call_count == expected
    assert api.upload_folder.call_count == expected


def test_publishes_model_at_root(api, checkpoint):
    hub.push_model_to_hub(
        checkpoint_dir=checkpoint,
        repo_id="user/model",
        private=True,
        strategy="every_save",
        rollout_id=19,
        is_final=False,
    )
    api.create_repo.assert_called_once_with(repo_id="user/model", repo_type="model", private=True, exist_ok=True)
    api.upload_folder.assert_called_once_with(
        repo_id="user/model",
        repo_type="model",
        folder_path=checkpoint,
        commit_message="Upload Miles model at rollout 19",
        ignore_patterns=[".complete"],
        delete_patterns=["*.safetensors", "pytorch_model*.bin", "*.index.json", "adapter/*"],
    )


def test_skips_incomplete_export(api, tmp_path):
    hub.push_model_to_hub(
        checkpoint_dir=str(tmp_path),
        repo_id="user/model",
        private=False,
        strategy="every_save",
        rollout_id=19,
        is_final=False,
    )
    api.create_repo.assert_not_called()
    api.upload_folder.assert_not_called()


@pytest.mark.parametrize("operation", ["create_repo", "upload_folder"])
def test_remote_failure_preserves_checkpoint(api, checkpoint, caplog, operation):
    getattr(api, operation).side_effect = RuntimeError("synthetic-private-details")
    hub.push_model_to_hub(
        checkpoint_dir=checkpoint,
        repo_id="user/model",
        private=False,
        strategy="every_save",
        rollout_id=19,
        is_final=True,
    )
    assert (Path(checkpoint) / "model.safetensors").read_bytes() == b"weights"
    assert "Hub upload failed (RuntimeError)" in caplog.text
    assert "synthetic-private-details" not in caplog.text
