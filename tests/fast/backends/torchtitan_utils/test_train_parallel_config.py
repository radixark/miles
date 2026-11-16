from types import ModuleType, SimpleNamespace

import pytest


@pytest.mark.parametrize("pp_size,expected", [(1, True), (2, False)])
def test_rollout_side_schedule_only_without_pipeline_parallelism(
    titan_actor_module: ModuleType, monkeypatch: pytest.MonkeyPatch, pp_size: int, expected: bool
) -> None:
    parallel_state = SimpleNamespace(
        pp=SimpleNamespace(size=pp_size),
        train_parallel_config=lambda *, supports_precomputed_schedule: supports_precomputed_schedule,
    )
    monkeypatch.setattr(titan_actor_module, "get_parallel_state", lambda: parallel_state)
    actor_type = titan_actor_module.TorchtitanTrainRayActor

    assert actor_type.__new__(actor_type).train_parallel_config is expected
