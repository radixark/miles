import logging
from pathlib import Path
from types import SimpleNamespace

from miles.rollout.data_source import DataSource, LegacyRolloutDataSourceWithBuffer, RolloutDataSource
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


def test_load_says_so_when_it_finds_no_state(tmp_path: Path, caplog) -> None:
    """A dataset silently starting over is a run replaying samples its trainers already trained on."""
    source = _bare_source(rollout_global_dataset=True, load=str(tmp_path))

    with caplog.at_level(logging.WARNING, logger="miles.rollout.data_source"):
        source.load(rollout_id=3)

    assert "no data source state under" in caplog.text


def test_load_says_so_when_the_run_names_no_load_directory(tmp_path: Path, caplog) -> None:
    """A run told to write but not to read still has to say that its dataset starts from the beginning."""
    source = _bare_source(rollout_global_dataset=True, load=None)

    with caplog.at_level(logging.WARNING, logger="miles.rollout.data_source"):
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


def test_legacy_buffer_checkpoint_restores_pending_groups_and_cursors(tmp_path: Path) -> None:
    """Partial rollout resumes its backlog before issuing samples with later identities."""
    args = _make_args(
        save=str(tmp_path),
        load=str(tmp_path),
        buffer_filter_path=None,
        n_samples_per_prompt=1,
    )
    source = LegacyRolloutDataSourceWithBuffer(args)
    source.add_samples([[Sample(index=4, group_index=4, prompt="pending")]])
    source.sample_group_index = 5
    source.sample_index = 5

    source.save(rollout_id=2)
    restored = LegacyRolloutDataSourceWithBuffer(args)
    restored.load(rollout_id=2)

    groups = restored.get_samples(num_samples=2)

    assert [group[0].prompt for group in groups] == ["pending", ""]
    assert groups[1][0].group_index == 5
    assert groups[1][0].index == 5
    assert restored.get_buffer_length() == 0
    assert restored.sample_group_index == 6
    assert restored.sample_index == 6


class _ReadOnlyDataSource(DataSource):
    def get_samples(self, num_samples: int) -> list[list[Sample]]:
        return [[Sample(prompt=str(index))] for index in range(num_samples)]

    def save(self, rollout_id: int) -> None:
        pass

    def load(self, rollout_id: int | None = None) -> None:
        pass
