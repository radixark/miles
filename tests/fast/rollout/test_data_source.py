import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from miles.rollout.data_source import DataSource, RolloutDataSource
from miles.utils.types import Sample


class TestReadOnlyDataSource:
    def test_a_custom_source_only_implements_read_and_checkpoint_operations(self) -> None:
        """A custom source can instantiate without implementing sample recycling."""
        source = _ReadOnlyDataSource()

        assert source.get_samples(num_samples=1)[0][0].prompt == "0"
        source.save(rollout_id=0)
        source.load(rollout_id=0)


def _make_args(**overrides) -> SimpleNamespace:
    defaults = dict(rollout_global_dataset=False, save=None, load=None, rollout_shuffle=False)
    return SimpleNamespace(**{**defaults, **overrides})


def test_save_restores_sample_id_cursors_without_a_global_dataset(tmp_path: Path) -> None:
    """Pending samples cannot collide with newly issued identities after resume."""
    source = RolloutDataSource(_make_args(save=str(tmp_path)))
    source.sample_group_index = 7
    source.sample_index = 21

    source.save(rollout_id=3)
    restored = RolloutDataSource(_make_args(load=str(tmp_path)))
    restored.load(rollout_id=3)

    assert restored.sample_group_index == 7
    assert restored.sample_index == 21


def _bare_source(**overrides) -> RolloutDataSource:
    source = RolloutDataSource.__new__(RolloutDataSource)
    source.args = _make_args(**overrides)
    source.metadata = {}
    return source


def test_configured_load_rejects_a_missing_data_source_checkpoint(tmp_path: Path) -> None:
    """A configured resume refuses to reset dataset cursors when its checkpoint is missing."""
    source = _bare_source(rollout_global_dataset=True, load=str(tmp_path))

    with pytest.raises(FileNotFoundError, match="global_dataset_state_dict_3.pt"):
        source.load(rollout_id=3)


def test_load_says_so_when_the_run_names_no_load_directory(tmp_path: Path, caplog) -> None:
    """A run told to write but not to read still has to say that its dataset starts from the beginning."""
    source = _bare_source(rollout_global_dataset=True, load=None)

    with caplog.at_level(logging.WARNING, logger="miles.utils.simple_checkpointer"):
        source.load(rollout_id=3)

    assert "no --load" in caplog.text


def test_load_restores_the_state_it_finds(tmp_path: Path) -> None:
    """This is the ordinary resume, and the position it restores is what keeps a run off samples it has seen."""
    import torch

    from miles.rollout.data_source import compute_global_dataset_state_path

    path = Path(compute_global_dataset_state_path(str(tmp_path), rollout_id=3))
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"sample_offset": 7, "epoch_id": 1}, path)
    source = _bare_source(rollout_global_dataset=True, load=str(tmp_path))

    source.load(rollout_id=3)

    assert (source.sample_offset, source.epoch_id) == (7, 1)


class _ReadOnlyDataSource(DataSource):
    def get_samples(self, num_samples: int) -> list[list[Sample]]:
        return [[Sample(prompt=str(index))] for index in range(num_samples)]

    def save(self, rollout_id: int) -> None:
        pass

    def load(self, rollout_id: int | None = None) -> None:
        pass
