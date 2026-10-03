from typing import cast

from tests.fast.utils.multi_policy.conftest import _FreshPolicyStartup

from miles.utils.multi_policy.utils import assert_consistent_restore, create_trainers
from miles.utils.workers.backend_capability.base import BackendCapability
from miles.utils.workers.worker_handle import BaseWorkerHandle


class TestFreshMultiPolicyStartup:
    async def test_hf_initialized_policies_start_at_zero_without_restore_metadata(
        self, fresh_policy_startup: _FreshPolicyStartup
    ) -> None:
        """Fresh HF policy workers must not turn an unspecified start into a checkpoint resume."""
        startup = fresh_policy_startup
        assert startup.args.start_rollout_id is None
        assert all(handle.args.start_rollout_id is None for handle in startup.handles.values())
        assert all("finetune" not in vars(handle.args.backend) for handle in startup.handles.values())

        trainers = await create_trainers(
            startup.args,
            rollout_executor=cast(BaseWorkerHandle, startup.rollout),
            capability=cast(BackendCapability, None),
        )
        assert_consistent_restore(startup.args, trainers=trainers, leader_model_id="solver")

        assert {name: trainer.start_rollout_id for name, trainer in trainers.items()} == {"solver": 0, "verifier": 0}
        assert startup.rollout.restored_rollouts == []
