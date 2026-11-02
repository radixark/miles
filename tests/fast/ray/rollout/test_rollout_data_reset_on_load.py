from argparse import Namespace
from pathlib import Path

import pytest

from miles.ray.rollout.rollout_executor import RolloutExecutor, compute_rollout_checkpoint_dir


class _Recorder:
    def __init__(self, name: str, loaded: list[str]) -> None:
        self.name = name
        self.loaded = loaded

    def load(self, directory: Path) -> None:
        self.loaded.append(self.name)

    def discard_completed(self, *, completed_rollout_ids: dict) -> None:
        pass


def _executor(*, reset: bool, loaded: list[str]) -> RolloutExecutor:
    executor = RolloutExecutor.__new__(RolloutExecutor)
    executor.args = Namespace(rollout_data_reset_on_load=reset)
    executor.use_legacy_rollout_v1 = False
    executor._output_snapshotter = _Recorder("executor", loaded)
    executor.data_source = _Recorder("data_source", loaded)
    executor.generate_rollout = _Recorder("generate_rollout", loaded)
    executor.eval_generate_rollout = _Recorder("eval_generate_rollout", loaded)
    executor._delete_store_refs = None
    return executor


@pytest.mark.parametrize(
    "reset, expected",
    [
        (False, ["executor", "data_source", "generate_rollout", "eval_generate_rollout"]),
        # the in-flight prompt groups came from the saved data source, so they are dropped with it
        (True, ["executor", "eval_generate_rollout"]),
    ],
)
async def test_a_data_reset_restores_everything_but_the_data_and_its_in_flight_groups(
    tmp_path: Path, reset: bool, expected: list[str]
) -> None:
    compute_rollout_checkpoint_dir(str(tmp_path), rollout_id=3).mkdir(parents=True)
    loaded: list[str] = []

    await _executor(reset=reset, loaded=loaded).load(3, load=str(tmp_path))

    assert loaded == expected
