from argparse import Namespace
from pathlib import Path

import pytest

from miles.ray.rollout.output_snapshotter import _RolloutExecutorOutputSnapshotter
from miles.ray.rollout.rollout_executor import RolloutExecutor, compute_rollout_checkpoint_dir
from miles.rollout import fully_async_rollout
from miles.rollout.data_source import RolloutDataResumeMismatch
from miles.rollout.fully_async_data_buffer import DataBufferInput, DefaultDataBuffer
from miles.rollout.fully_async_rollout import FullyAsyncRolloutFn, _FullyAsyncRolloutState
from miles.utils.simple_checkpointer import save_simple_checkpoint
from miles.utils.types import Sample


class _Recorder:
    def __init__(self, name: str, calls: list[str], *, mismatch: bool = False, needs_flag: bool = True) -> None:
        self.name = name
        self.calls = calls
        self.mismatch = mismatch
        self.needs_flag = needs_flag

    def load(self, directory: Path) -> None:
        if self.mismatch:
            raise RolloutDataResumeMismatch("rollout dataset 'math' changed rm_type", needs_flag=self.needs_flag)
        self.calls.append(f"{self.name}.load")

    def load_sample_ids(self, directory: Path) -> None:
        self.calls.append(f"{self.name}.load_sample_ids")

    def drop_saved_in_flight(self, directory: Path) -> None:
        self.calls.append(f"{self.name}.drop_saved_in_flight")

    def drop_loaded(self, *, reason: str) -> None:
        self.calls.append(f"{self.name}.drop_loaded")

    def discard_completed(self, *, completed_rollout_ids: dict) -> None:
        pass


async def _load(tmp_path: Path, *, reset: bool, mismatch: bool, needs_flag: bool = True) -> list[str]:
    compute_rollout_checkpoint_dir(str(tmp_path), rollout_id=3).mkdir(parents=True)
    calls: list[str] = []
    executor = RolloutExecutor.__new__(RolloutExecutor)
    executor.args = Namespace(rollout_data_reset_on_load=reset)
    executor.use_legacy_rollout_v1 = False
    executor._output_snapshotter = _Recorder("snapshotter", calls)
    executor.data_source = _Recorder("data_source", calls, mismatch=mismatch, needs_flag=needs_flag)
    executor.generate_rollout = _Recorder("train_fn", calls)
    executor.eval_generate_rollout = _Recorder("eval_fn", calls)
    executor._delete_store_refs = None
    await executor.load(3, load=str(tmp_path))
    return calls


_RESTORED = ["snapshotter.load", "data_source.load", "train_fn.load", "eval_fn.load"]


@pytest.mark.parametrize("reset", [False, True])
async def test_a_matching_checkpoint_resumes_in_place_even_with_the_flag(tmp_path: Path, reset: bool) -> None:
    """Keeping the flag in the launch command must not rewind the datasets on every later resume."""
    assert await _load(tmp_path, reset=reset, mismatch=False) == _RESTORED


async def test_a_mismatch_without_the_flag_refuses_the_resume(tmp_path: Path) -> None:
    with pytest.raises(RolloutDataResumeMismatch, match="changed rm_type"):
        await _load(tmp_path, reset=False, mismatch=True)


@pytest.mark.parametrize("reset, needs_flag", [(True, True), (False, False)])
async def test_a_mismatch_with_the_flag_resets_the_data_and_drops_what_came_from_it(
    tmp_path: Path, reset: bool, needs_flag: bool
) -> None:
    """A mix resumed under --prompt-data needs no flag; every other mismatch does."""
    assert await _load(tmp_path, reset=reset, mismatch=True, needs_flag=needs_flag) == [
        "snapshotter.load",
        "data_source.load_sample_ids",
        "snapshotter.drop_loaded",
        "train_fn.drop_saved_in_flight",
        "eval_fn.load",
    ]


def test_dropping_loaded_batches_leaves_nothing_to_replay(tmp_path: Path) -> None:
    """A batch drawn from the old datasets must not be trained after a reset."""
    args = Namespace(
        use_output_store=False,
        enable_sample_ownership_checker=False,
        ci_inject_missing_prefetched_batch_bug=False,
    )
    saved = _RolloutExecutorOutputSnapshotter(args=args)
    saved.capture(trainer_model_id=None, rollout_id=4, data=[Sample(index=0)], metadata={})
    saved.save(tmp_path)

    restored = _RolloutExecutorOutputSnapshotter(args=args)
    restored.load(tmp_path)
    restored.drop_loaded(reason="data_reset_on_load")

    assert restored.get(trainer_model_id=None, rollout_id=4) is None


def test_a_fully_async_reset_records_every_saved_group_as_dropped(tmp_path: Path, monkeypatch) -> None:
    """Queued, running and finished-but-untrained groups all came from the reset data source."""
    save_simple_checkpoint(
        directory=tmp_path,
        data=_FullyAsyncRolloutState(
            retry_buffer=[[Sample(index=0)]],
            running=[[Sample(index=1)]],
            output=[DataBufferInput(prompt_group=[Sample(index=2)], group=[Sample(index=2)])],
        ),
    )
    dropped: list[int] = []
    monkeypatch.setattr(
        fully_async_rollout.SampleOwnershipRecorder,
        "log_dropped_samples",
        lambda *, args, samples, reason: dropped.extend(sample.index for sample in samples),
    )
    rollout_fn = FullyAsyncRolloutFn.__new__(FullyAsyncRolloutFn)
    rollout_fn.args = Namespace()
    rollout_fn._output = DefaultDataBuffer.__new__(DefaultDataBuffer)
    rollout_fn._output._args = Namespace()

    rollout_fn.drop_saved_in_flight(tmp_path)

    assert dropped == [0, 1, 2]
    assert not rollout_fn.__dict__.get("_retry_buffer")


def test_a_custom_buffer_drops_its_own_saved_layout(tmp_path: Path) -> None:
    """Only the buffer knows its saved layout, so the rollout hands the finished groups back to it."""
    save_simple_checkpoint(
        directory=tmp_path,
        data=_FullyAsyncRolloutState(retry_buffer=[], running=[], output={"custom": "layout"}),
    )
    handed: list[object] = []
    rollout_fn = FullyAsyncRolloutFn.__new__(FullyAsyncRolloutFn)
    rollout_fn.args = Namespace(enable_sample_ownership_checker=False)
    rollout_fn._output = Namespace(drop_saved_state=lambda state, *, reason: handed.append(state))

    rollout_fn.drop_saved_in_flight(tmp_path)

    assert handed == [{"custom": "layout"}]
