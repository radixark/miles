import io
import sys
import types
from argparse import Namespace
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.distributed.checkpoint as dist_cp
from typing_extensions import override

from miles.backends.megatron_utils.torch_dist_checkpoint import _COMMON_STATE_KEY, load_checkpoint_args


class _PrepackedBytesSavePlanner(dist_cp.default_planner.DefaultSavePlanner):
    """The fork's planner: a ShardedObject arrives already torch.save()d into a BytesIO and is
    written as-is, where the default planner would pickle the BytesIO object once more."""

    @override
    def transform_object(self, write_item: dist_cp.planner.WriteItem, object: Any):
        if isinstance(object, io.BytesIO):
            return object
        return super().transform_object(write_item, object)


def _write_fork_layout(checkpoint_dir: Path, args: Namespace) -> None:
    """What radixark/Megatron-LM writes: the common state is one ShardedObject inside the DCP
    checkpoint, serialized as the one-element list of its data, beside the weights."""
    payload = io.BytesIO()
    torch.save([{"args": args, "iteration": 3, "checkpoint_version": 3.0}], payload)
    dist_cp.save(
        {_COMMON_STATE_KEY: payload, "embedding.weight": torch.zeros(2, 2)},
        checkpoint_id=str(checkpoint_dir),
        planner=_PrepackedBytesSavePlanner(),
        no_dist=True,
    )


def _write_legacy_layout(checkpoint_dir: Path, args: Namespace) -> None:
    dist_cp.save({"embedding.weight": torch.zeros(2, 2)}, checkpoint_id=str(checkpoint_dir), no_dist=True)
    torch.save({"args": args, "iteration": 3}, checkpoint_dir / "common.pt")


_LAYOUTS = {"fork": _write_fork_layout, "legacy-common.pt": _write_legacy_layout}


class TestLoadCheckpointArgs:
    @pytest.mark.parametrize("write", _LAYOUTS.values(), ids=_LAYOUTS.keys())
    def test_reads_the_args_from_either_layout(self, tmp_path: Path, write):
        write(tmp_path, Namespace(num_layers=4, original_hf_model_name="qwen3"))

        args = load_checkpoint_args(str(tmp_path))

        assert (args.num_layers, args.original_hf_model_name) == (4, "qwen3")

    @pytest.mark.parametrize("write", _LAYOUTS.values(), ids=_LAYOUTS.keys())
    def test_megatron_classes_inside_the_args_need_no_megatron(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, write
    ):
        """Saved args carry Megatron objects (enums, specs); the converters run where Megatron may
        be missing, and must not need the process-wide pickle patch the scripts install."""
        module = types.ModuleType("megatron_not_installed_here")
        module.Spec = type("Spec", (), {"__module__": module.__name__})
        monkeypatch.setitem(sys.modules, module.__name__, module)
        write(tmp_path, Namespace(spec=module.Spec(), num_layers=4))
        monkeypatch.delitem(sys.modules, module.__name__)

        args = load_checkpoint_args(str(tmp_path))

        assert args.num_layers == 4
        assert type(args.spec).__name__ == "DummyClass"

    def test_a_checkpoint_without_common_state_is_refused(self, tmp_path: Path):
        """Falling through to a KeyError deep inside DCP would hide which layout was expected."""
        dist_cp.save({"embedding.weight": torch.zeros(2, 2)}, checkpoint_id=str(tmp_path), no_dist=True)

        with pytest.raises(FileNotFoundError, match="common.pt"):
            load_checkpoint_args(str(tmp_path))
