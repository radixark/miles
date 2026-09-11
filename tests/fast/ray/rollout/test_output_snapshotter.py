from argparse import Namespace
from pathlib import Path

import pytest

from miles.ray.rollout.output_snapshotter import _RolloutExecutorOutputSnapshotter
from miles.utils.types import Sample


class TestRolloutExecutorOutputSnapshotter:
    def test_checkpoint_preserves_untrained_sample_outputs_and_isolates_mutations(self, tmp_path: Path) -> None:
        """All pending model-step outputs survive conversion mutations and repeated recovery."""
        args = Namespace(save=tmp_path, load=tmp_path, ci_test=False, ci_inject_missing_prefetched_batch_bug=False)
        snapshotter = _RolloutExecutorOutputSnapshotter(args=args)
        samples = [Sample(index=7, tokens=[1, 2])]
        metadata = {"prompt_group_sizes": [1]}
        for model_id, step in [("actor", 2), ("actor", 3), ("critic", 3), ("actor", 4)]:
            snapshotter.capture(trainer_model_id=model_id, rollout_id=step, data=samples, metadata=metadata)
        samples[0].tokens.append(3)
        metadata["prompt_group_sizes"].append(2)

        snapshotter.save(2)
        restored = _RolloutExecutorOutputSnapshotter(args=args)
        restored.load(2)
        assert not restored.has(trainer_model_id="actor", rollout_id=2)
        with pytest.raises(KeyError):
            restored.get(trainer_model_id="actor", rollout_id=2)
        for model_id, step in [("actor", 3), ("critic", 3), ("actor", 4)]:
            assert restored.has(trainer_model_id=model_id, rollout_id=step)
            output = restored.get(trainer_model_id=model_id, rollout_id=step)
            assert output.data == [Sample(index=7, tokens=[1, 2])]
            assert output.metadata == {"prompt_group_sizes": [1]}
            output.data[0].tokens.append(4)
            output.metadata.clear()

        restored.save(3)
        resumed_again = _RolloutExecutorOutputSnapshotter(args=args)
        resumed_again.load(3)
        assert not resumed_again.has(trainer_model_id="actor", rollout_id=3)
        with pytest.raises(KeyError):
            resumed_again.get(trainer_model_id="actor", rollout_id=3)
        assert not resumed_again.has(trainer_model_id="critic", rollout_id=3)
        with pytest.raises(KeyError):
            resumed_again.get(trainer_model_id="critic", rollout_id=3)
        remaining = resumed_again.get(trainer_model_id="actor", rollout_id=4)
        assert remaining.data == [Sample(index=7, tokens=[1, 2])]
        assert remaining.metadata == {"prompt_group_sizes": [1]}
