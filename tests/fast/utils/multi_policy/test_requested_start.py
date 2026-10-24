from unittest.mock import AsyncMock

import pytest
from tests.fast.utils.multi_policy.conftest import _FreshPolicyStartup

from miles.utils.multi_policy.utils import assert_consistent_restore, create_trainers


class TestRequestedPolicyStart:
    async def test_an_explicit_run_start_is_not_replaced_by_fresh_per_trainer_defaults(
        self, fresh_policy_startup: _FreshPolicyStartup
    ) -> None:
        """A requested replay position reaches every policy and remains subject to global checkpoint validation."""
        startup = fresh_policy_startup
        startup.args.start_rollout_id = 9
        assert all(handle.args.start_rollout_id is None for handle in startup.handles.values())
        rollout = AsyncMock()

        trainers = await create_trainers(startup.args, rollout_executor=rollout, handles=startup.handles)

        assert {name: trainer.start_rollout_id for name, trainer in trainers.items()} == {"solver": 9, "verifier": 9}
        rollout.load.assert_awaited_once_with(8, load=None, completed_rollout_ids={"solver": 8, "verifier": 8})
        with pytest.raises(AssertionError, match="holds no record"):
            assert_consistent_restore(startup.args, trainers=trainers, leader_model_id="solver")
