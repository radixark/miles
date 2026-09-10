from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from miles.backends.fsdp_utils import checkpoint


@pytest.fixture
def checkpoint_actor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    monkeypatch.setattr(checkpoint.torch.cuda, "synchronize", Mock())
    monkeypatch.setattr(checkpoint.torch.cuda, "get_rng_state_all", Mock(return_value=[]))
    monkeypatch.setattr(checkpoint.dist, "barrier", Mock())
    monkeypatch.setattr(checkpoint.dist, "get_rank", Mock(return_value=0))
    monkeypatch.setattr(checkpoint.dist, "get_world_size", Mock(return_value=1))
    monkeypatch.setattr(checkpoint.dcp, "save", Mock())
    monkeypatch.setattr(checkpoint.dcp, "load", Mock())
    return SimpleNamespace(
        args=Namespace(
            save=str(tmp_path),
            load=str(tmp_path),
            ckpt_step=None,
            no_save_optim=True,
            no_load_optim=True,
            no_load_rng=True,
        ),
        model=object(),
        global_step=0,
        micro_step=0,
        weight_updater=SimpleNamespace(weight_version=9),
    )
