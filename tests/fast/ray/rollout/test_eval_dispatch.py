from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-cpu")

import asyncio

import pytest
from tests.fast.ray.rollout.conftest import make_args

from miles.ray.rollout.eval_dispatch import EvalDispatcher


class FakeRemoteMethod:
    def __init__(self, log: list, name: str):
        self.log = log
        self.name = name

    def remote(self, *args):
        self.log.append((self.name, args))
        fut = asyncio.get_event_loop().create_future()
        fut.set_result(None)
        return fut


class FakeRolloutManager:
    def __init__(self):
        self.log: list = []
        self.report_eval_skip = FakeRemoteMethod(self.log, "report_eval_skip")


def make_dispatcher(**overrides) -> EvalDispatcher:
    args = make_args(eval_keep_snapshots=2, **overrides)
    return EvalDispatcher(args, actor_model=None, rollout_manager=FakeRolloutManager())


def failed_ref(error: Exception):
    fut = asyncio.get_event_loop().create_future()
    fut.set_exception(error)
    return fut


class TestSettle:
    async def test_a_crashed_eval_fails_when_reporting_the_ci_skip(self, tmp_path):
        """A crashed eval fails CI through the centralized skipped-point contract."""
        dispatcher = make_dispatcher(ci_test=True)
        exported_dir = str(tmp_path / "step_3")

        class FailingRemoteMethod(FakeRemoteMethod):
            def remote(self, *args):
                self.log.append((self.name, args))
                fut = asyncio.get_event_loop().create_future()
                fut.set_exception(RuntimeError("CI eval 3 skipped: crashed"))
                return fut

        dispatcher.rollout_manager.report_eval_skip = FailingRemoteMethod(
            dispatcher.rollout_manager.log, "report_eval_skip"
        )

        with pytest.raises(RuntimeError, match="CI eval 3 skipped: crashed"):
            await dispatcher._settle(3, failed_ref(RuntimeError("engine gone")), exported_dir)

        assert dispatcher.rollout_manager.log == [("report_eval_skip", (3, "crashed"))]
        assert dispatcher._exported == [exported_dir]

    async def test_a_crashed_eval_degrades_to_a_skipped_point_outside_ci(self, tmp_path):
        """Without --ci-test a raised eval only logs a crashed skip and training continues."""
        dispatcher = make_dispatcher(ci_test=False)
        exported_dir = str(tmp_path / "step_3")

        await dispatcher._settle(3, failed_ref(RuntimeError("engine gone")), exported_dir)

        assert dispatcher.rollout_manager.log == [("report_eval_skip", (3, "crashed"))]
        assert dispatcher._exported == [exported_dir]

    async def test_a_settled_eval_reports_nothing_and_retires_its_snapshot(self, tmp_path):
        """A point that finished cleanly is neither reported as skipped nor re-raised."""
        dispatcher = make_dispatcher(ci_test=True)
        exported_dir = str(tmp_path / "step_3")
        ref = asyncio.get_event_loop().create_future()
        ref.set_result(None)

        await dispatcher._settle(3, ref, exported_dir)

        assert dispatcher.rollout_manager.log == []
        assert dispatcher._exported == [exported_dir]
