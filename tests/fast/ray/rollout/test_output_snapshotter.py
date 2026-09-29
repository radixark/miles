from argparse import Namespace
from pathlib import Path

import pytest

from miles.ray.rollout.output_snapshotter import _MAX_RETAINED_OUTPUTS, _RolloutExecutorOutputSnapshotter
from miles.utils.types import Sample


def _args() -> Namespace:
    return Namespace()


class TestRolloutExecutorOutputSnapshotter:
    def test_a_replayed_batch_cannot_be_captured_twice(self, tmp_path: Path) -> None:
        """Re-capturing a key would overwrite the snapshot a replayed rollout still owns."""
        snapshotter = _RolloutExecutorOutputSnapshotter(args=_args())
        snapshotter.capture(trainer_model_id=None, rollout_id=2, data=[], metadata={})

        with pytest.raises(AssertionError, match="captured before"):
            snapshotter.capture(trainer_model_id=None, rollout_id=2, data=[], metadata={})

    def test_a_batch_this_process_generated_is_never_replayed(self, tmp_path: Path) -> None:
        """Replaying what the running process just produced would train the same batch twice."""
        snapshotter = _RolloutExecutorOutputSnapshotter(args=_args())
        snapshotter.capture(trainer_model_id=None, rollout_id=2, data=[], metadata={})

        with pytest.raises(AssertionError, match="must not be replayed"):
            snapshotter.get(trainer_model_id=None, rollout_id=2)

    def test_only_the_newest_captures_are_retained(self, tmp_path: Path) -> None:
        """An unbounded store would hold every rollout a long run ever generated."""
        snapshotter = _RolloutExecutorOutputSnapshotter(args=_args())
        for rollout_id in range(_MAX_RETAINED_OUTPUTS + 1):
            snapshotter.capture(trainer_model_id=None, rollout_id=rollout_id, data=[], metadata={})

        snapshotter.save(tmp_path)
        restored = _RolloutExecutorOutputSnapshotter(args=_args())
        restored.load(tmp_path)

        assert restored.get(trainer_model_id=None, rollout_id=0) is None
        for rollout_id in range(1, _MAX_RETAINED_OUTPUTS + 1):
            assert restored.get(trainer_model_id=None, rollout_id=rollout_id) is not None

    def test_restored_batches_do_not_push_out_the_newest_captures(self, tmp_path: Path) -> None:
        """Retention bounds what this process generated, so a batch awaiting replay cannot be evicted by it."""
        snapshotter = _RolloutExecutorOutputSnapshotter(args=_args())
        snapshotter.capture(trainer_model_id=None, rollout_id=3, data=[Sample(index=7)], metadata={})
        snapshotter.save(tmp_path)
        restored = _RolloutExecutorOutputSnapshotter(args=_args())
        restored.load(tmp_path)
        for rollout_id in range(4, 4 + _MAX_RETAINED_OUTPUTS):
            restored.capture(trainer_model_id=None, rollout_id=rollout_id, data=[], metadata={})

        assert restored.get(trainer_model_id=None, rollout_id=3) is not None

    def test_a_restored_batch_is_replayed_exactly_once(self, tmp_path: Path) -> None:
        """A replayed batch reaches training once; asking for it twice is a bug, not a regeneration."""
        snapshotter = _RolloutExecutorOutputSnapshotter(args=_args())
        snapshotter.capture(trainer_model_id=None, rollout_id=3, data=[Sample(index=7)], metadata={})
        snapshotter.save(tmp_path)
        restored = _RolloutExecutorOutputSnapshotter(args=_args())
        restored.load(tmp_path)

        assert restored.get(trainer_model_id=None, rollout_id=3).data == [Sample(index=7)]
        with pytest.raises(AssertionError, match="was already replayed"):
            restored.get(trainer_model_id=None, rollout_id=3)

    def test_an_unreplayed_restored_batch_survives_the_next_save(self, tmp_path: Path) -> None:
        """A checkpoint taken before the restored batch is consumed must keep it."""
        snapshotter = _RolloutExecutorOutputSnapshotter(args=_args())
        snapshotter.capture(trainer_model_id=None, rollout_id=3, data=[Sample(index=7)], metadata={})
        snapshotter.save(tmp_path)
        restored = _RolloutExecutorOutputSnapshotter(args=_args())
        restored.load(tmp_path)

        restored.save(tmp_path)
        resumed = _RolloutExecutorOutputSnapshotter(args=_args())
        resumed.load(tmp_path)

        assert resumed.get(trainer_model_id=None, rollout_id=3) is not None

    def test_a_replayed_batch_still_reaches_the_next_checkpoint(self, tmp_path: Path) -> None:
        """A save writes everything the executor holds, so a restore finds captured and replayed batches alike."""
        snapshotter = _RolloutExecutorOutputSnapshotter(args=_args())
        snapshotter.capture(trainer_model_id=None, rollout_id=3, data=[Sample(index=7)], metadata={})
        snapshotter.save(tmp_path)
        restored = _RolloutExecutorOutputSnapshotter(args=_args())
        restored.load(tmp_path)
        restored.capture(trainer_model_id=None, rollout_id=4, data=[Sample(index=8)], metadata={})
        assert restored.get(trainer_model_id=None, rollout_id=3) is not None

        restored.save(tmp_path)
        resumed = _RolloutExecutorOutputSnapshotter(args=_args())
        resumed.load(tmp_path)

        assert resumed.get(trainer_model_id=None, rollout_id=3).data == [Sample(index=7)]
        assert resumed.get(trainer_model_id=None, rollout_id=4).data == [Sample(index=8)]

    def test_captured_outputs_are_isolated_from_later_mutations(self, tmp_path: Path) -> None:
        """Conversion mutates the samples it was handed, and the checkpoint must not follow."""
        snapshotter = _RolloutExecutorOutputSnapshotter(args=_args())
        samples = [Sample(index=7, tokens=[1, 2])]
        metadata = {"prompt_group_sizes": [1]}
        snapshotter.capture(trainer_model_id="actor", rollout_id=3, data=samples, metadata=metadata)
        samples[0].tokens.append(3)
        metadata["prompt_group_sizes"].append(2)

        snapshotter.save(tmp_path)
        restored = _RolloutExecutorOutputSnapshotter(args=_args())
        restored.load(tmp_path)

        output = restored.get(trainer_model_id="actor", rollout_id=3)
        assert output.data == [Sample(index=7, tokens=[1, 2])]
        assert output.metadata == {"prompt_group_sizes": [1]}

    def test_replayed_outputs_are_isolated_from_later_mutations(self, tmp_path: Path) -> None:
        """A replayed batch is mutated by conversion, and a save before training must not follow."""
        snapshotter = _RolloutExecutorOutputSnapshotter(args=_args())
        snapshotter.capture(trainer_model_id=None, rollout_id=3, data=[Sample(index=7, tokens=[1])], metadata={})
        snapshotter.save(tmp_path)
        restored = _RolloutExecutorOutputSnapshotter(args=_args())
        restored.load(tmp_path)

        restored.get(trainer_model_id=None, rollout_id=3).data[0].tokens.append(2)
        restored.save(tmp_path)
        resumed = _RolloutExecutorOutputSnapshotter(args=_args())
        resumed.load(tmp_path)

        assert resumed.get(trainer_model_id=None, rollout_id=3).data == [Sample(index=7, tokens=[1])]
