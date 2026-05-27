"""Unit tests for worker/main.py functions.

main.py has heavy top-level imports (megatron.training.arguments, etc.)
that aren't fully available in the lightweight test container.  We
intercept the import by patching sys.modules for missing leaves *before*
importing the module under test.

The stubbing is scoped to a single test via the ``worker_main`` pytest
fixture, which uses ``monkeypatch.setitem(sys.modules, ...)`` so the
sys.modules state is restored automatically on teardown.  This prevents
the file from polluting sys.modules for downstream tests in the same
pytest session (e.g. test_lora_model_branches.py previously failed with
``ImportError: cannot import name 'save_checkpoint'`` because this file
registered an under-populated stub at module-import time).
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

import pytest
import torch


# megatron.* leaves whose top-level imports in main.py would fail in the
# lightweight test container.  These are *always* stubbed because the
# container never ships megatron.training / megatron.core.
_MEGATRON_STUBS: dict[str, dict[str, Any] | MagicMock] = {
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

# miles.backends.megatron_utils.* leaves.  We prefer the real module
# when it imports cleanly; otherwise we fabricate a stub carrying the
# names main.py imports from each leaf.
_MILES_MEGATRON_UTILS_LEAVES: list[str] = [
    "miles.backends.megatron_utils.arguments",
    "miles.backends.megatron_utils.checkpoint",
    "miles.backends.megatron_utils.initialize",
    "miles.backends.megatron_utils.model_provider",
]

_MILES_MEGATRON_UTILS_ATTRS: list[str] = [
    "set_default_megatron_args",
    "load_checkpoint",
    "init",
    "get_model_provider_func",
]


def _ensure_parent_module(
    dotted: str,
    monkeypatch: pytest.MonkeyPatch,
) -> ModuleType | None:
    """Ensure parent packages exist in sys.modules, via monkeypatch so
    they get torn down after the test."""
    if not dotted:
        return None
    if dotted in sys.modules:
        return sys.modules[dotted]
    try:
        return importlib.import_module(dotted)
    except ImportError:
        parent_name, _, child_name = dotted.rpartition(".")
        parent = _ensure_parent_module(parent_name, monkeypatch)
        mod = ModuleType(dotted)
        mod.__path__ = []  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, dotted, mod)
        if parent is not None:
            monkeypatch.setattr(parent, child_name, mod, raising=False)
        return mod


def _make_megatron_stub(
    path: str,
    attrs: dict[str, Any] | MagicMock,
) -> ModuleType | MagicMock:
    if isinstance(attrs, MagicMock):
        return attrs
    mod = ModuleType(path)
    for name, val in attrs.items():
        setattr(mod, name, val)
    return mod


@pytest.fixture
def worker_main(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Stage the sys.modules scaffolding that worker/main.py needs, then
    import and yield the module.

    Every sys.modules mutation goes through ``monkeypatch.setitem`` so
    pytest's monkeypatch fixture restores sys.modules exactly on
    teardown.  No state leaks to subsequent tests.
    """
    # megatron.* — always stubbed (never installed in the test container).
    # Ensure parent packages exist first so attribute access on the
    # stub modules works during downstream import.
    for path, attrs in _MEGATRON_STUBS.items():
        parent_name, _, child_name = path.rpartition(".")
        parent = _ensure_parent_module(parent_name, monkeypatch)
        stub = _make_megatron_stub(path, attrs)
        monkeypatch.setitem(sys.modules, path, stub)
        if parent is not None:
            monkeypatch.setattr(parent, child_name, stub, raising=False)

    # miles.backends.megatron_utils.* — prefer real import; only stub on
    # ImportError (including transitive megatron.* failures).
    for leaf in _MILES_MEGATRON_UTILS_LEAVES:
        if leaf in sys.modules:
            continue
        try:
            importlib.import_module(leaf)
        except ImportError:
            parent_name, _, child_name = leaf.rpartition(".")
            parent = _ensure_parent_module(parent_name, monkeypatch)
            stub = ModuleType(leaf)
            for name in _MILES_MEGATRON_UTILS_ATTRS:
                setattr(stub, name, MagicMock())
            monkeypatch.setitem(sys.modules, leaf, stub)
            if parent is not None:
                monkeypatch.setattr(parent, child_name, stub, raising=False)

    # Evict any cached worker/main so it re-binds to our stubs, then
    # import.  Evict it again on teardown via monkeypatch.delitem so
    # subsequent fresh imports also get a clean rebuild.
    main_dotted = "miles.utils.debug_utils.run_megatron.worker.main"
    monkeypatch.delitem(sys.modules, main_dotted, raising=False)
    module = importlib.import_module(main_dotted)
    monkeypatch.setitem(sys.modules, main_dotted, module)
    return module


class TestParseArgs:
    def test_ref_load_overrides_args_load(self, worker_main: ModuleType) -> None:
        """When script_args.ref_load is set, args.load is overridden."""
        with (
            patch.object(worker_main, "WORKER_SCRIPT_ARGS_BRIDGE") as mock_bridge,
            patch.object(worker_main, "parse_args") as mock_parse_args,
        ):
            args = argparse.Namespace(load=None)
            mock_parse_args.return_value = args

            script_args = MagicMock()
            script_args.ref_load = Path("/some/path")
            mock_bridge.from_namespace.return_value = script_args

            returned_args, returned_script = worker_main._parse_args()

            assert returned_args.load == "/some/path"
            assert returned_script is script_args

    def test_ref_load_none_preserves_original_load(
        self,
        worker_main: ModuleType,
    ) -> None:
        """When script_args.ref_load is None, args.load stays as-is."""
        with (
            patch.object(worker_main, "WORKER_SCRIPT_ARGS_BRIDGE") as mock_bridge,
            patch.object(worker_main, "parse_args") as mock_parse_args,
        ):
            args = argparse.Namespace(load="/orig")
            mock_parse_args.return_value = args

            script_args = MagicMock()
            script_args.ref_load = None
            mock_bridge.from_namespace.return_value = script_args

            returned_args, _ = worker_main._parse_args()

            assert returned_args.load == "/orig"


class TestApplySourcePatches:
    def test_reads_yaml_and_calls_patcher(
        self,
        worker_main: ModuleType,
        tmp_path: Path,
    ) -> None:
        with patch.object(worker_main, "apply_patches_from_config") as mock_apply:
            config_file = tmp_path / "patches.yaml"
            config_file.write_text("patches:\n  - target: foo")

            worker_main._apply_source_patches(config_file)

            mock_apply.assert_called_once()
            call_args = mock_apply.call_args
            assert call_args[0][0] == "patches:\n  - target: foo"
            assert "extra_imports" in call_args[1] or len(call_args[0]) > 1


class TestRunForwardBackward:
    def test_forward_only_when_run_backward_false(
        self,
        worker_main: ModuleType,
    ) -> None:
        """run_backward=False → forward_only=True passed to the func."""
        with (
            patch.object(worker_main, "dist") as mock_dist,
            patch.object(worker_main, "get_forward_backward_func") as mock_get_fb,
        ):
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

            worker_main._run_forward_backward(args=args, script=script, model=model, batch=batch)

            call_kwargs = mock_fb_func.call_args[1]
            assert call_kwargs["forward_only"] is True

    def test_no_logits_captured_returns_none(
        self,
        worker_main: ModuleType,
    ) -> None:
        """If no logits captured (non-last PP stage), returns None."""
        with (
            patch.object(worker_main, "dist") as mock_dist,
            patch.object(worker_main, "get_forward_backward_func") as mock_get_fb,
        ):
            mock_fb_func = MagicMock(return_value=[])
            mock_get_fb.return_value = mock_fb_func
            mock_dist.get_rank.return_value = 1

            args = argparse.Namespace(seq_length=4, micro_batch_size=1)
            script = MagicMock()
            script.run_backward = False

            result = worker_main._run_forward_backward(
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
    def test_dumper_enable_env_triggers_step_and_disable(
        self,
        worker_main: ModuleType,
    ) -> None:
        with patch.object(worker_main, "dumper") as mock_dumper:
            with patch.dict(os.environ, {"DUMPER_ENABLE": "1"}):
                worker_main._finalize_dumper()

            mock_dumper.step.assert_called_once()
            mock_dumper.configure.assert_called_once_with(enable=False)

    def test_no_dumper_enable_env_does_nothing(
        self,
        worker_main: ModuleType,
    ) -> None:
        with patch.object(worker_main, "dumper") as mock_dumper:
            with patch.dict(os.environ, {}, clear=True):
                env_backup = os.environ.pop("DUMPER_ENABLE", None)
                try:
                    worker_main._finalize_dumper()
                finally:
                    if env_backup is not None:
                        os.environ["DUMPER_ENABLE"] = env_backup

            mock_dumper.step.assert_not_called()
            mock_dumper.configure.assert_not_called()
