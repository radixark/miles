from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

from miles.utils import profile_utils


class TestBackendProfilerOwnership:
    @pytest.mark.parametrize("start,wait,warmup", [(0, 0, 0), (3, 2, 1)])
    def test_overall_profiler_uses_backend_schedule_and_output_directory(
        self, monkeypatch: pytest.MonkeyPatch, start: int, wait: int, warmup: int
    ) -> None:
        """The profiler uses backend schedule settings and starts only after initialization."""
        sdk = Mock()
        monkeypatch.setattr(profile_utils.torch.profiler, "profile", sdk.profile)
        monkeypatch.setattr(profile_utils.torch.profiler, "schedule", sdk.schedule)
        monkeypatch.setattr(profile_utils.torch.profiler, "tensorboard_trace_handler", sdk.trace_handler)
        monkeypatch.setattr(profile_utils.torch.distributed, "get_rank", lambda: 7)
        args = SimpleNamespace(
            profile_target=["train_overall"],
            backend=SimpleNamespace(
                use_pytorch_profiler=True,
                record_memory_history=False,
                profile_step_start=start,
                profile_step_end=8,
                tensorboard_dir="backend-traces",
            ),
        )

        profiler = profile_utils.TrainProfiler(args)
        sdk.profile.return_value.start.assert_not_called()
        profiler.on_init_end()
        profiler.step(rollout_id=0)

        sdk.schedule.assert_called_once_with(wait=wait, warmup=warmup, active=8 - start, repeat=1)
        sdk.trace_handler.assert_called_once_with("backend-traces", worker_name="train_overall_rank_7", use_gzip=True)
        sdk.profile.assert_called_once_with(
            schedule=sdk.schedule.return_value,
            on_trace_ready=sdk.trace_handler.return_value,
            record_shapes=True,
            with_stack=True,
            profile_memory=True,
            with_flops=True,
        )
        sdk.profile.return_value.start.assert_called_once_with()
        sdk.profile.return_value.step.assert_called_once_with()

    @pytest.mark.parametrize("name", ["train_actor", "train_log_probs"])
    @pytest.mark.parametrize("enabled", [False, True])
    def test_profiled_iterators_preserve_values_and_step_each_consumed_item(
        self, monkeypatch: pytest.MonkeyPatch, name: str, enabled: bool
    ) -> None:
        """Loop profiling preserves yielded values while enabling SDK work only for the selected target."""
        sdk = Mock()
        monkeypatch.setattr(profile_utils.torch.profiler, "profile", sdk)
        monkeypatch.setattr(profile_utils.torch.profiler, "schedule", Mock())
        monkeypatch.setattr(profile_utils.torch.profiler, "tensorboard_trace_handler", Mock())
        monkeypatch.setattr(profile_utils.torch.distributed, "get_rank", lambda: 0)
        args = SimpleNamespace(
            profile_target=[name],
            backend=SimpleNamespace(
                use_pytorch_profiler=enabled,
                record_memory_history=False,
                profile_step_start=0,
                profile_step_end=2,
                tensorboard_dir="traces",
            ),
        )
        profiler = profile_utils.TrainProfiler(args)
        values = [object(), object()]
        iterator = (
            profiler.iterate_train_actor(values) if name == "train_actor" else profiler.iterate_train_log_probs(values)
        )

        assert list(iterator) == values
        if enabled:
            sdk.return_value.start.assert_called_once_with()
            assert sdk.return_value.step.call_count == 2
        else:
            sdk.assert_not_called()

    def test_memory_dump_uses_backend_suffix_and_stops_at_requested_rollout(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Memory snapshots retain backend filename ownership and the requested rollout boundary."""
        sdk = Mock()
        monkeypatch.setattr(profile_utils.time, "time", lambda: 123)
        monkeypatch.setattr(profile_utils.torch.distributed, "get_rank", lambda: 4)
        monkeypatch.setattr(profile_utils.torch.cuda.memory, "_record_memory_history", sdk.record)
        monkeypatch.setattr(profile_utils.torch.cuda.memory, "_dump_snapshot", sdk.dump)
        monkeypatch.setattr(profile_utils.torch._C, "_cuda_attach_out_of_memory_observer", sdk.attach, raising=False)
        args = SimpleNamespace(
            profile_target=["train_overall"],
            memory_recorder="torch",
            memory_snapshot_dir=tmp_path,
            memory_snapshot_num_steps=2,
            backend=SimpleNamespace(
                use_pytorch_profiler=False, record_memory_history=True, memory_snapshot_path="backend.pickle"
            ),
        )

        profiler = profile_utils.TrainProfiler(args)
        profiler.step(rollout_id=0)
        sdk.dump.assert_not_called()
        profiler.step(rollout_id=1)

        sdk.dump.assert_called_once_with(tmp_path / "memory_snapshot_time123_rank4_backend.pickle")
        assert sdk.record.call_args_list == [
            call(max_entries=1000000, stacks="all"),
            call(enabled=None),
        ]
