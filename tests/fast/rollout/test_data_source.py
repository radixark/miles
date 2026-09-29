import logging
from pathlib import Path
from types import SimpleNamespace

import torch

from miles.rollout.data_source import DataSource, RolloutDataSource
from miles.utils.types import Sample


class TestReadOnlyDataSource:
    def test_a_custom_source_only_implements_read_and_checkpoint_operations(self, tmp_path: Path) -> None:
        """A custom source can instantiate without implementing sample recycling."""
        source = _ReadOnlyDataSource()

        assert source.get_samples(num_samples=1)[0][0].prompt == "0"
        source.save(tmp_path)
        source.load(tmp_path)


def _make_args(**overrides) -> SimpleNamespace:
    defaults = dict(rollout_global_dataset=False, save=None, load=None, rollout_shuffle=False)
    return SimpleNamespace(**{**defaults, **overrides})


def test_save_writes_nothing_without_a_global_dataset(tmp_path: Path) -> None:
    """The built-in source guards itself, so the executor needs no outer guard to keep it silent."""
    source = RolloutDataSource(_make_args(save=str(tmp_path)))

    source.save(tmp_path)

    assert list(tmp_path.iterdir()) == []


def test_load_reads_nothing_without_a_global_dataset(tmp_path: Path) -> None:
    """The load side has always been called unconditionally and relies on the same internal guard."""
    source = RolloutDataSource(_make_args(load=str(tmp_path)))

    source.load(tmp_path)

    assert source.sample_offset == 0
    assert source.epoch_id == 0


def _bare_source(**overrides) -> RolloutDataSource:
    source = RolloutDataSource.__new__(RolloutDataSource)
    source.args = _make_args(**overrides)
    source.metadata = {}
    return source


def test_load_says_so_when_it_finds_no_state(tmp_path: Path, caplog) -> None:
    """A dataset silently starting over is a run replaying samples its trainers already trained on."""
    source = _bare_source(rollout_global_dataset=True)

    with caplog.at_level(logging.WARNING, logger="miles.rollout.data_source"):
        source.load(tmp_path)

    assert "no dataset state under" in caplog.text


def test_load_says_so_when_the_run_keeps_no_global_dataset(tmp_path: Path, caplog) -> None:
    """A custom rollout function keeps its own state, and the operator has to know this one restored none."""
    source = _bare_source(rollout_global_dataset=False)

    with caplog.at_level(logging.WARNING, logger="miles.rollout.data_source"):
        source.load(tmp_path)

    assert "rollout-global-dataset" in caplog.text


def test_load_restores_the_state_it_finds(tmp_path: Path) -> None:
    """This is the ordinary resume, and the position it restores is what keeps a run off samples it has seen."""
    torch.save({"sample_offset": 7, "epoch_id": 1}, tmp_path / "state.pt")
    source = _bare_source(rollout_global_dataset=True)

    source.load(tmp_path)

    assert (source.sample_offset, source.epoch_id) == (7, 1)


class _ReadOnlyDataSource(DataSource):
    def get_samples(self, num_samples: int) -> list[list[Sample]]:
        return [[Sample(prompt=str(index))] for index in range(num_samples)]

    def save(self, directory: Path) -> None:
        pass

    def load(self, directory: Path) -> None:
        pass
