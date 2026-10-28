from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from tests.fast.fixtures.args_fixtures import make_trainer_args

from miles.backends.fsdp_utils import checkpoint


@pytest.fixture
def fsdp_checkpoint_sdk(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> SimpleNamespace:
    def save(state_dict: dict, *, checkpoint_id: str) -> None:
        torch.save(
            {key: wrapper.state_dict() for key, wrapper in state_dict.items()}, Path(checkpoint_id) / "state.pt"
        )
        (Path(checkpoint_id) / ".metadata").touch()

    def load(*, state_dict: dict, checkpoint_id: str) -> None:
        saved = torch.load(Path(checkpoint_id) / "state.pt", weights_only=False)
        for key, wrapper in state_dict.items():
            wrapper.load_state_dict(saved[key])

    monkeypatch.setattr(checkpoint.dcp, "save", save)
    monkeypatch.setattr(checkpoint.dcp, "load", load)
    monkeypatch.setattr(checkpoint.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(checkpoint.dist, "get_world_size", lambda: 1)
    monkeypatch.setattr(checkpoint.dist, "barrier", lambda: None)
    monkeypatch.setattr(checkpoint.torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(checkpoint.torch.cuda, "get_rng_state_all", lambda: [])
    monkeypatch.setattr(checkpoint.torch.cuda, "is_available", lambda: False)
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.125)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
    args = make_trainer_args(
        train_backend="fsdp", save=str(tmp_path), load=str(tmp_path), ckpt_step=4, no_save_optim=False
    )
    return SimpleNamespace(
        args=args, model=model, optimizer=optimizer, lr_scheduler=scheduler, global_step=7, micro_step=11
    )
