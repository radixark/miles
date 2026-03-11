"""Unit tests for worker/main.py functions.

main.py has heavy top-level imports (megatron, sglang, etc.) that are not
available in the lightweight test environment.  We pre-populate sys.modules
with MagicMock stubs so that importing main.py succeeds without those packages.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import torch

# ---------------------------------------------------------------------------
# Stub out heavy dependencies before importing the module under test.
# ---------------------------------------------------------------------------

_STUB_MODULES: list[str] = [
    "megatron.core",
    "megatron.core.enums",
    "megatron.core.mpu",
    "megatron.core.pipeline_parallel",
    "megatron.training",
    "megatron.training.arguments",
    "megatron.training.training",
    "miles.backends.megatron_utils.arguments",
    "miles.backends.megatron_utils.checkpoint",
    "miles.backends.megatron_utils.initialize",
    "miles.backends.megatron_utils.model_provider",
    "miles.utils.debug_utils.run_megatron.worker.replay",
]

_original_modules: dict[str, ModuleType] = {}
for _mod_name in _STUB_MODULES:
    if _mod_name not in sys.modules:
        stub = ModuleType(_mod_name)
        # Some modules need specific attributes that are accessed at import time
        if _mod_name == "megatron.core.enums":
            stub.ModelType = MagicMock()  # type: ignore[attr-defined]
        if _mod_name == "megatron.core.pipeline_parallel":
            stub.get_forward_backward_func = MagicMock()  # type: ignore[attr-defined]
        if _mod_name == "megatron.training.arguments":
            stub.parse_args = MagicMock()  # type: ignore[attr-defined]
            stub.validate_args = MagicMock()  # type: ignore[attr-defined]
        if _mod_name == "megatron.training.training":
            stub.get_model = MagicMock()  # type: ignore[attr-defined]
        sys.modules[_mod_name] = stub

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
