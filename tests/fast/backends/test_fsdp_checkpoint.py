"""FSDP checkpoint saves publish the tracker only after a complete checkpoint."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from torch.distributed.checkpoint.api import CheckpointException

from miles.backends.fsdp_utils import checkpoint
from miles.utils import distributed_utils


@pytest.fixture
def single_rank_gloo(tmp_path, monkeypatch):
    dist.init_process_group("gloo", init_method=f"file://{tmp_path / 'store'}", rank=0, world_size=1)
    monkeypatch.setattr(distributed_utils, "GLOO_GROUP", dist.new_group(backend="gloo"))
    yield
    dist.destroy_process_group()


def _actor(save_dir: Path):
    args = SimpleNamespace(save=str(save_dir), no_save_optim=False)
    return SimpleNamespace(args=args, model=None, optimizer=None, lr_scheduler=None, global_step=5, micro_step=9)


def _without_cuda(monkeypatch):
    monkeypatch.setattr(checkpoint.torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(checkpoint.torch.cuda, "get_rng_state_all", lambda: [])


def _fake_dcp_save(monkeypatch, error: BaseException | None = None):
    def save(state_dict, checkpoint_id):
        if error is not None:
            raise error
        Path(checkpoint_id).mkdir(parents=True, exist_ok=True)
        (Path(checkpoint_id) / ".metadata").write_text("shards")

    monkeypatch.setattr(checkpoint.dcp, "save", save)
    _without_cuda(monkeypatch)


def test_save_writes_checkpoint_then_tracker(tmp_path, monkeypatch, single_rank_gloo):
    _fake_dcp_save(monkeypatch)

    checkpoint.save(_actor(tmp_path / "save"), iteration=2)

    iter_dir = tmp_path / "save" / "iter_0000003"
    assert (iter_dir / "model" / ".metadata").exists()
    assert torch.load(iter_dir / "rng.pt")["cuda"] == []
    assert json.loads((iter_dir / "meta.json").read_text())["next_rollout_id"] == 3
    assert (tmp_path / "save" / "latest_checkpointed_iteration.txt").read_text() == "3"
    assert not list((tmp_path / "save").glob("*.tmp"))


@pytest.mark.parametrize(
    "error, reported",
    [
        (OSError("disk full"), "OSError: disk full"),
        # dcp.save reports write failures as CheckpointException, a BaseException
        (CheckpointException("optimizer shards failed", {}), "CheckpointException"),
    ],
)
def test_failed_save_keeps_previous_tracker_and_removes_partial_checkpoint(
    tmp_path, monkeypatch, single_rank_gloo, error, reported
):
    save_dir = tmp_path / "save"
    save_dir.mkdir()
    (save_dir / "latest_checkpointed_iteration.txt").write_text("2")
    _fake_dcp_save(monkeypatch, error)

    with pytest.raises(distributed_utils.RankFailureError, match=f"rank 0: {reported}"):
        checkpoint.save(_actor(save_dir), iteration=2)

    assert not (save_dir / "iter_0000003").exists()
    assert (save_dir / "latest_checkpointed_iteration.txt").read_text() == "2"


def test_load_skips_empty_optimizer_dirs_left_by_older_no_save_optim_saves(tmp_path, monkeypatch, single_rank_gloo):
    _without_cuda(monkeypatch)
    torch.manual_seed(0)
    saved = torch.nn.Linear(4, 3)
    save_dir = tmp_path / "save"
    actor = _actor(save_dir)
    actor.model = saved
    checkpoint.save(actor, iteration=2)
    # older saves created these dirs even when --no-save-optim skipped writing them
    (save_dir / "iter_0000003" / "optimizer").mkdir()
    (save_dir / "iter_0000003" / "lr_scheduler").mkdir()

    model = torch.nn.Linear(4, 3)
    optimizer = torch.optim.Adam(model.parameters())
    resumed = _actor(save_dir)
    resumed.args.load, resumed.args.ckpt_step, resumed.args.no_load_optim = str(save_dir), None, False
    resumed.model, resumed.optimizer = model, optimizer
    resumed.lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)

    payload = checkpoint.load(resumed)

    assert payload["iteration"] == 3
    assert torch.equal(model.weight, saved.weight)
