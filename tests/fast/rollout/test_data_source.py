import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from miles.rollout.data_source import (
    DataSource,
    LegacyRolloutDataSourceWithBuffer,
    RolloutDataSource,
    compute_global_dataset_state_path,
    pop_first,
)
from miles.utils.types import Sample


class TestReadOnlyDataSource:
    def test_a_custom_source_only_implements_read_and_checkpoint_operations(self) -> None:
        """A custom source can instantiate without implementing sample recycling."""
        source = _MinimalDataSource()

        assert source.get_samples(num_samples=1)[0][0].prompt == "0"
        source.save(rollout_id=0)
        source.load(rollout_id=0)


def _make_args(**overrides) -> SimpleNamespace:
    defaults = dict(rollout_global_dataset=False, save=None, load=None, rollout_shuffle=False)
    return SimpleNamespace(**{**defaults, **overrides})


def test_save_writes_nothing_without_a_global_dataset(tmp_path: Path) -> None:
    """The built-in source guards itself, so the executor needs no outer guard to keep it silent."""
    source = RolloutDataSource(_make_args(save=str(tmp_path)))

    source.save(rollout_id=3)

    assert list(tmp_path.iterdir()) == []


def test_load_reads_nothing_without_a_global_dataset(tmp_path: Path) -> None:
    """The load side has always been called unconditionally and relies on the same internal guard."""
    source = RolloutDataSource(_make_args(load=str(tmp_path)))

    source.load(rollout_id=3)

    assert source.sample_offset == 0
    assert source.epoch_id == 0


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

    assert "no dataset state under" in caplog.text


def test_load_says_so_when_the_run_names_no_load_directory(tmp_path: Path, caplog) -> None:
    """A run told to write but not to read still has to say that its dataset starts from the beginning."""
    source = _bare_source(rollout_global_dataset=True, load=None)

    with caplog.at_level(logging.WARNING, logger="miles.rollout.data_source"):
        source.load(rollout_id=3)

    assert "no --load" in caplog.text


def test_load_says_so_when_the_run_keeps_no_global_dataset(tmp_path: Path, caplog) -> None:
    """A custom rollout function keeps its own state, and the operator has to know this one restored none."""
    source = _bare_source(rollout_global_dataset=False, load=str(tmp_path))

    with caplog.at_level(logging.WARNING, logger="miles.rollout.data_source"):
        source.load(rollout_id=3)

    assert "rollout-global-dataset" in caplog.text


def test_load_restores_the_state_it_finds(tmp_path: Path) -> None:
    """This is the ordinary resume, and the position it restores is what keeps a run off samples it has seen."""
    path = Path(compute_global_dataset_state_path(str(tmp_path), rollout_id=3))
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"sample_offset": 7, "epoch_id": 1}, path)
    source = _bare_source(rollout_global_dataset=True, load=str(tmp_path))

    source.load(rollout_id=3)

    assert (source.sample_offset, source.epoch_id) == (7, 1)


def _bare_buffer_source(**overrides) -> LegacyRolloutDataSourceWithBuffer:
    source = LegacyRolloutDataSourceWithBuffer.__new__(LegacyRolloutDataSourceWithBuffer)
    source.args = _make_args(n_samples_per_prompt=1, **overrides)
    source.metadata = {}
    source.epoch_id = 0
    source.sample_group_index = 0
    source.sample_index = 0
    source.sample_offset = 0
    source.dataset = None
    source.buffer = []
    source.buffer_filter = pop_first
    return source


def _recycled_group(*, index: int, group_index: int) -> list[Sample]:
    return [Sample(index=index, group_index=group_index, prompt="what is 1+7?")]


class TestRolloutDataSourceWithBufferCheckpoint:
    def test_a_recycled_group_survives_a_save_and_load(self, tmp_path: Path) -> None:
        """A group returned to the buffer is a rollout already paid for, and a resume that drops it loses that work."""
        source = _bare_buffer_source(rollout_global_dataset=True, save=str(tmp_path))
        source.add_samples([_recycled_group(index=11, group_index=4)])

        source.save(rollout_id=3)
        restored = _bare_buffer_source(rollout_global_dataset=True, load=str(tmp_path))
        restored.load(rollout_id=3)

        groups = restored.get_samples(1)
        assert [(sample.index, sample.group_index, sample.prompt) for group in groups for sample in group] == [
            (11, 4, "what is 1+7?")
        ]

    def test_the_cursor_and_the_buffer_come_back_from_one_file(self, tmp_path: Path) -> None:
        """The buffer rides in the dataset state file, so restoring it must not cost the cursor that shares it."""
        source = _bare_buffer_source(rollout_global_dataset=True, save=str(tmp_path))
        source.sample_offset = 7
        source.epoch_id = 1
        source.sample_group_index = 5
        source.sample_index = 13
        source.add_samples([_recycled_group(index=13, group_index=5)])

        source.save(rollout_id=3)
        restored = _bare_buffer_source(rollout_global_dataset=True, load=str(tmp_path))
        restored.load(rollout_id=3)

        assert (
            restored.sample_offset,
            restored.epoch_id,
            restored.sample_group_index,
            restored.sample_index,
        ) == (7, 1, 5, 13)
        assert len(restored.buffer) == 1

    def test_a_checkpoint_written_before_the_buffer_was_saved_still_loads(self, tmp_path: Path) -> None:
        """Runs resume from checkpoints older than this feature, and a missing key must not stop them."""
        path = Path(compute_global_dataset_state_path(str(tmp_path), rollout_id=3))
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"sample_offset": 7, "epoch_id": 1, "sample_group_index": 5, "sample_index": 13},
            path,
        )
        restored = _bare_buffer_source(rollout_global_dataset=True, load=str(tmp_path))

        restored.load(rollout_id=3)

        assert restored.buffer == []
        assert restored.sample_offset == 7

    def test_saving_leaves_no_temporary_file_behind(self, tmp_path: Path) -> None:
        """A leftover temporary in the rollout directory is indistinguishable from a real checkpoint file."""
        source = _bare_buffer_source(rollout_global_dataset=True, save=str(tmp_path))
        source.add_samples([_recycled_group(index=11, group_index=4)])

        source.save(rollout_id=3)

        assert [entry.name for entry in (tmp_path / "rollout").iterdir()] == ["global_dataset_state_dict_3.pt"]

    def test_a_failed_save_keeps_the_previous_checkpoint(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A half-written dataset state would resume a run onto samples its trainers already trained on."""
        source = _bare_buffer_source(rollout_global_dataset=True, save=str(tmp_path))
        source.sample_offset = 7
        source.save(rollout_id=3)
        source.sample_offset = 9
        monkeypatch.setattr(torch, "save", _raising_save)

        with pytest.raises(RuntimeError, match="no save today"):
            source.save(rollout_id=3)

        path = Path(compute_global_dataset_state_path(str(tmp_path), rollout_id=3))
        assert [entry.name for entry in (tmp_path / "rollout").iterdir()] == [path.name]
        assert torch.load(path, weights_only=False)["sample_offset"] == 7


def _raising_save(*args, **kwargs):
    raise RuntimeError("no save today")


class _MinimalDataSource(DataSource):
    def get_samples(self, num_samples: int) -> list[list[Sample]]:
        return [[Sample(prompt=str(index))] for index in range(num_samples)]

    def save(self, rollout_id: int) -> None:
        pass

    def load(self, rollout_id: int | None = None) -> None:
        pass
