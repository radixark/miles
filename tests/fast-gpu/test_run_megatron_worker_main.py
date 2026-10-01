"""Unit tests for worker/main.py helper functions.

Lives under tests/fast-gpu/ because worker/main.py has heavy top-level
megatron / sglang imports that need a real GPU container to resolve cleanly.
Each fast-gpu file runs in its own subprocess (per-file isolation), so the
sys.modules stubbing below cannot leak across files.
"""

from __future__ import annotations

from tests.ci.ci_register import register_cuda_ci, register_rocm_ci

register_cuda_ci(
    est_time=30,
    suite="stage-b-2-gpu-h200",
    labels=["megatron"],
    hardware=["hopper", "blackwell"],
)
register_rocm_ci(
    est_time=20,
    suite="nightly-stage-c-2-gpu-mi350",
    labels=["megatron"],
)

import argparse
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


# Import the real `miles` package tree before any stubbing below. `_ensure_module`
# walks dotted ancestors and stubs every missing segment, so stubbing a
# `miles.backends.megatron_utils.*` leaf while `miles` is not yet imported would
# replace the real `miles` package with an empty (non-package) stub -- and the
# later `from miles.utils...main import` would then fail with "miles is not a
# package". Importing the real parent package here registers the genuine
# `miles`, `miles.backends`, and `miles.backends.megatron_utils` packages in
# sys.modules so only the leaf modules get stubbed.
import miles.backends.megatron_utils  # noqa: E402,F401

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

from miles.utils.debug_utils.run_megatron.worker.main import (  # noqa: E402
    _apply_source_patches,
    _finalize_dumper,
    _parse_args,
    _run_forward_backward,
)

_MODULE = "miles.utils.debug_utils.run_megatron.worker.main"


class TestParseArgs:
    def test_script_options_are_translated_for_the_shared_parser(self) -> None:
        """Use the shared parser with standalone topology and checkpoint options."""
        argv = [
            "worker",
            "--script-hf-checkpoint",
            "/model",
            "--script-token-ids-file",
            "/tokens",
            "--script-ref-load",
            "/checkpoint",
            "--script-role",
            "critic",
            "--micro-batch-size",
            "2",
            "--dsv4-impl",
            "miles",
        ]
        captured: list[str] = []
        parsed = object()

        def parse_shared() -> object:
            captured.extend(sys.argv[1:])
            return parsed

        with patch.object(sys, "argv", argv), patch.dict(
            os.environ, {"WORLD_SIZE": "8", "LOCAL_WORLD_SIZE": "4"}
        ), patch(f"{_MODULE}.parse_args", side_effect=parse_shared):
            args, script_args = _parse_args()
            assert sys.argv is argv

        assert args is parsed
        assert script_args.ref_load == Path("/checkpoint")
        for flag, value in [
            ("--hf-checkpoint", "/model"),
            ("--load", "/checkpoint"),
            ("--ref-load", "/checkpoint"),
            ("--actor-num-nodes", "2"),
            ("--actor-num-gpus-per-node", "4"),
            ("--micro-batch-size", "2"),
            ("--dsv4-impl", "miles"),
            ("--advantage-estimator", "ppo"),
        ]:
            assert captured[captured.index(flag) + 1] == value
        assert "--debug-train-only" in captured
        assert not any(argument.startswith("--script-") for argument in captured)

    def test_missing_ref_load_preserves_model_load_argument(self) -> None:
        """Preserve a model checkpoint when the script does not override it."""
        argv = [
            "worker",
            "--script-hf-checkpoint",
            "/model",
            "--script-token-ids-file",
            "/tokens",
            "--load",
            "/original",
        ]
        captured: list[str] = []

        def parse_shared() -> object:
            captured.extend(sys.argv[1:])
            return object()

        with patch.object(sys, "argv", argv), patch(f"{_MODULE}.parse_args", side_effect=parse_shared):
            _parse_args()

        assert captured.count("--load") == 1
        assert captured[captured.index("--load") + 1] == "/original"


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

        args = argparse.Namespace(backend=argparse.Namespace(seq_length=4, micro_batch_size=1))
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

        args = argparse.Namespace(backend=argparse.Namespace(seq_length=4, micro_batch_size=1))
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


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
