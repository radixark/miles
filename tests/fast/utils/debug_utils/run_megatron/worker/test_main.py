"""Unit tests for worker/main.py functions.

main.py has heavy top-level imports (megatron.training.arguments, etc.)
that aren't fully available in the lightweight test container.  We intercept
the import by patching sys.modules for missing leaves *before* importing
the module under test.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock, patch

import torch


def _ensure_module(dotted: str) -> ModuleType:
    """Ensure *dotted* exists in sys.modules, creating stubs for any missing segments."""
    parts = dotted.split(".")
    for i in range(len(parts)):
        partial = ".".join(parts[: i + 1])
        if partial not in sys.modules:
            sys.modules[partial] = ModuleType(partial)
    return sys.modules[dotted]


# Stub modules whose top-level imports in main.py would fail.
_STUBS: dict[str, dict[str, Any]] = {
    "megatron.training.arguments": {
        "parse_args": MagicMock(),
        "validate_args": MagicMock(),
    },
    "megatron.training.training": {
        "get_model": MagicMock(),
    },
    "megatron.core.enums": {
        "ModelType": MagicMock(),
    },
    "megatron.core.pipeline_parallel": {
        "get_forward_backward_func": MagicMock(),
    },
    "megatron.core.mpu": MagicMock(),
}

for _mod_path, _attrs in _STUBS.items():
    mod = _ensure_module(_mod_path)
    if isinstance(_attrs, dict):
        for attr_name, attr_val in _attrs.items():
            if not hasattr(mod, attr_name):
                setattr(mod, attr_name, attr_val)
    else:
        # Replace the whole module with a MagicMock
        sys.modules[_mod_path] = _attrs

# Also ensure miles.backends.megatron_utils sub-modules have their needed symbols
for _sub in [
    "miles.backends.megatron_utils.arguments",
    "miles.backends.megatron_utils.checkpoint",
    "miles.backends.megatron_utils.initialize",
    "miles.backends.megatron_utils.model_provider",
]:
    mod = _ensure_module(_sub)
    # Provide any names imported by main.py from these modules
    for name in [
        "set_default_megatron_args",
        "load_checkpoint",
        "init",
        "get_model_provider_func",
    ]:
        if not hasattr(mod, name):
            setattr(mod, name, MagicMock())

# Replay module
_ensure_module("miles.utils.debug_utils.run_megatron.worker.replay")
for name in ["load_replay_data", "save_replay_data", "setup_replay_before_model"]:
    mod = sys.modules["miles.utils.debug_utils.run_megatron.worker.replay"]
    if not hasattr(mod, name):
        setattr(mod, name, MagicMock())

from miles.utils.debug_utils.run_megatron.worker.main import (  # noqa: E402
    _apply_source_patches,
    _finalize_dumper,
    _parse_args,
    _run_forward_backward,
)

_MODULE = "miles.utils.debug_utils.run_megatron.worker.main"


class TestParseArgs:
    @patch(f"{_MODULE}.WORKER_SCRIPT_ARGS_BRIDGE")
    @patch(f"{_MODULE}.parse_args")
    def test_ref_load_overrides_args_load(
        self,
        mock_parse_args: MagicMock,
        mock_bridge: MagicMock,
    ) -> None:
        """When script_args.ref_load is set, args.load is overridden."""
        args = argparse.Namespace(load=None)
        mock_parse_args.return_value = args

        script_args = MagicMock()
        script_args.ref_load = Path("/some/path")
        mock_bridge.from_namespace.return_value = script_args

        returned_args, returned_script = _parse_args()

        assert returned_args.load == "/some/path"
        assert returned_script is script_args

    @patch(f"{_MODULE}.WORKER_SCRIPT_ARGS_BRIDGE")
    @patch(f"{_MODULE}.parse_args")
    def test_ref_load_none_preserves_original_load(
        self,
        mock_parse_args: MagicMock,
        mock_bridge: MagicMock,
    ) -> None:
        """When script_args.ref_load is None, args.load stays as-is."""
        args = argparse.Namespace(load="/orig")
        mock_parse_args.return_value = args

        script_args = MagicMock()
        script_args.ref_load = None
        mock_bridge.from_namespace.return_value = script_args

        returned_args, _ = _parse_args()

        assert returned_args.load == "/orig"


class TestApplySourcePatches:
    @patch(f"{_MODULE}.apply_patches_from_config")
    def test_reads_yaml_and_calls_patcher(
        self,
        mock_apply: MagicMock,
        tmp_path: Path,
    ) -> None:
        config_file = tmp_path / "patches.yaml"
        config_file.write_text("patches:\n  - target: foo")

        _apply_source_patches(config_file)

        mock_apply.assert_called_once()
        call_args = mock_apply.call_args
        assert call_args[0][0] == "patches:\n  - target: foo"
        assert "extra_imports" in call_args[1] or len(call_args[0]) > 1


class TestRunForwardBackward:
    @patch(f"{_MODULE}.dist")
    @patch(f"{_MODULE}.get_forward_backward_func")
    def test_forward_only_when_run_backward_false(
        self,
        mock_get_fb: MagicMock,
        mock_dist: MagicMock,
    ) -> None:
        """run_backward=False → forward_only=True passed to the func."""
        mock_fb_func = MagicMock(return_value=[])
        mock_get_fb.return_value = mock_fb_func
        mock_dist.get_rank.return_value = 1

        args = argparse.Namespace(seq_length=4, micro_batch_size=1)
        script = MagicMock()
        script.run_backward = False

        model = [MagicMock()]
        batch = {
            "input_ids": torch.tensor([[1, 2, 3, 4]]),
            "position_ids": torch.arange(4).unsqueeze(0),
            "labels": torch.tensor([[2, 3, 4, -100]]),
        }

        _run_forward_backward(args=args, script=script, model=model, batch=batch)

        call_kwargs = mock_fb_func.call_args[1]
        assert call_kwargs["forward_only"] is True

    @patch(f"{_MODULE}.dist")
    @patch(f"{_MODULE}.get_forward_backward_func")
    def test_no_logits_captured_returns_none(
        self,
        mock_get_fb: MagicMock,
        mock_dist: MagicMock,
    ) -> None:
        """If no logits captured (non-last PP stage), returns None."""
        mock_fb_func = MagicMock(return_value=[])
        mock_get_fb.return_value = mock_fb_func
        mock_dist.get_rank.return_value = 1

        args = argparse.Namespace(seq_length=4, micro_batch_size=1)
        script = MagicMock()
        script.run_backward = False

        result = _run_forward_backward(
            args=args,
            script=script,
            model=[MagicMock()],
            batch={
                "input_ids": torch.tensor([[1, 2]]),
                "position_ids": torch.arange(2).unsqueeze(0),
                "labels": torch.tensor([[2, -100]]),
            },
        )

        assert result is None


class TestFinalizeDumper:
    @patch(f"{_MODULE}.dumper")
    def test_dumper_enable_env_triggers_step_and_disable(
        self,
        mock_dumper: MagicMock,
    ) -> None:
        with patch.dict(os.environ, {"DUMPER_ENABLE": "1"}):
            _finalize_dumper()

        mock_dumper.step.assert_called_once()
        mock_dumper.configure.assert_called_once_with(enable=False)

    @patch(f"{_MODULE}.dumper")
    def test_no_dumper_enable_env_does_nothing(
        self,
        mock_dumper: MagicMock,
    ) -> None:
        with patch.dict(os.environ, {}, clear=True):
            env_backup = os.environ.pop("DUMPER_ENABLE", None)
            try:
                _finalize_dumper()
            finally:
                if env_backup is not None:
                    os.environ["DUMPER_ENABLE"] = env_backup

        mock_dumper.step.assert_not_called()
        mock_dumper.configure.assert_not_called()
