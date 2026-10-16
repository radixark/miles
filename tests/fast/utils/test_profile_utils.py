from types import SimpleNamespace

import pytest
import torch

from miles.utils import profile_utils
from miles.utils.profile_utils import TrainProfiler


@pytest.fixture
def profiles_built(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """Records the keyword arguments of every torch profiler built, without starting one."""
    built: list[dict] = []
    stub = SimpleNamespace(start=lambda: None, step=lambda: None)
    monkeypatch.setattr(profile_utils.torch.profiler, "profile", lambda **kwargs: built.append(kwargs) or stub)
    return built


def _args(tmp_path, **overrides) -> SimpleNamespace:
    values = dict(
        use_pytorch_profiler=True,
        profile_target=["train_overall", "train_actor"],
        profile_ranks=[0, 8],
        profile_step_start=1,
        profile_step_end=2,
        pytorch_profiler_collect_shapes=False,
        pytorch_profiler_collect_callstack=False,
        record_memory_history=False,
        tensorboard_dir=str(tmp_path),
    )
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize(
    ("rank", "profile_ranks", "expected_profilers"),
    [(3, [0, 8], 0), (8, [0, 8], 2), (3, [], 2)],
)
def test_only_listed_ranks_build_a_torch_profiler(
    monkeypatch: pytest.MonkeyPatch, tmp_path, profiles_built, rank, profile_ranks, expected_profilers
):
    """Both the train_overall and the per-loop profilers follow --profile-ranks; an empty list traces every rank."""
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: rank)
    prof = TrainProfiler(_args(tmp_path, profile_ranks=profile_ranks))

    assert list(prof.iterate_train_actor(iter([1, 2]))) == [1, 2]
    assert len(profiles_built) == expected_profilers


@pytest.mark.parametrize("collect", [False, True])
def test_trace_options_follow_the_megatron_flags(monkeypatch: pytest.MonkeyPatch, tmp_path, profiles_built, collect):
    """Shapes and stacks follow --pytorch-profiler-collect-{shapes,callstack}; memory and FLOPs are never recorded."""
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    TrainProfiler(_args(tmp_path, pytorch_profiler_collect_shapes=collect, pytorch_profiler_collect_callstack=collect))

    (kwargs,) = profiles_built
    assert (kwargs["record_shapes"], kwargs["with_stack"]) == (collect, collect)
    assert not kwargs.get("profile_memory") and not kwargs.get("with_flops")
