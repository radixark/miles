from unittest.mock import patch

import pytest
from tests.fast.ray.rollout import test_external_engine_provider as _fleet_tests
from tests.fast.ray.rollout.test_external_engine_provider import TestARaggedExternalFleet as _RaggedFleet
from tests.fast.ray.rollout.test_external_engine_provider import _make_args


class TestExternalEngineDeclaredTotal:
    def test_unequal_engines_pass_validation_when_the_declared_total_differs(self) -> None:
        """Discovered engines of four and two GPUs are accepted despite a declared total of sixteen."""
        with patch.object(
            _fleet_tests,
            "_make_args",
            side_effect=lambda *args, **kwargs: _make_args(*args, **{**kwargs, "rollout_num_gpus": 16}),
        ):
            _RaggedFleet().test_engines_of_unequal_size_pass_the_argument_check()

    async def test_discovery_preserves_engine_sizes_when_the_declared_total_differs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A declared total of sixteen does not replace discovered sizes or their cumulative offsets."""
        with patch.object(
            _fleet_tests,
            "_make_args",
            side_effect=lambda *args, **kwargs: _make_args(*args, **{**kwargs, "rollout_num_gpus": 16}),
        ):
            await _RaggedFleet().test_a_fleet_of_unequal_engines_hands_the_run_each_engine_at_its_own_size(monkeypatch)
