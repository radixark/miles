from tests.e2e.deploy.conftest_deploy.hot_restart.cluster_observer import compute_hot_restart_workloads
from tests.e2e.deploy.conftest_deploy.hot_restart.driver import compute_release_of_config
from tests.e2e.deploy.conftest_deploy.hot_restart.fault_form import restamped_replaced_workloads

from miles.ray.specs.rollout import ROLLOUT_EXECUTOR_POOL_ID
from miles.utils.external_utils.command_utils.base_backend import ExecuteTrainConfig
from miles.utils.external_utils.command_utils.helm_backend.naming import ORCHESTRATOR_COMPONENT
from miles.utils.workers.worker_provider.kubernetes.helm.naming import component_name

CELL: dict = {"metadata": {"name": "actor-0"}}
CONFIG: ExecuteTrainConfig = ExecuteTrainConfig(run_id="demo", namespace="rl")
RELEASE: str = compute_release_of_config(CONFIG)
STAMPED: frozenset[str] = compute_hot_restart_workloads(RELEASE)
ORCHESTRATOR: str = component_name(RELEASE, ORCHESTRATOR_COMPONENT)
ROLLOUT_EXECUTOR: str = component_name(RELEASE, ROLLOUT_EXECUTOR_POOL_ID)
TRAINER: str = "a-workload-a-take-over-leaves-alone"


class TestWhatCountsAsATakeOverThatLanded:
    def test_a_take_over_replaces_the_orchestrator_and_the_rollout_executor_of_the_release(self):
        """These two names are what every stamp assertion below is written against."""
        assert STAMPED == {ORCHESTRATOR, ROLLOUT_EXECUTOR}

    def test_a_run_whose_replaced_workloads_were_all_restamped_counts(self):
        """A take-over rewrites the stamp of each workload it replaces, whatever the run had trained."""
        assert restamped_replaced_workloads(
            before=_stamps(orchestrator=None, rollout_executor=None), after=_stamps(), workloads=STAMPED
        )

    def test_a_second_take_over_rewriting_the_first_ones_stamps_counts(self):
        """One object carries one stamp, rewritten each time, so a landing is a value that changed."""
        assert restamped_replaced_workloads(
            before=_stamps(orchestrator="t1", rollout_executor="t1"), after=_stamps(), workloads=STAMPED
        )

    def test_a_run_whose_workloads_still_carry_the_stamps_they_carried_does_not_count(self):
        """The relaunch is still installing, and the stamps it will rewrite are the ones drawn against."""
        assert not restamped_replaced_workloads(before=_stamps(), after=_stamps(), workloads=STAMPED)

    def test_a_run_only_half_of_whose_workloads_were_restamped_does_not_count(self):
        """The two workloads are rolled by one upgrade but observed apart, so one of them is a half-landing."""
        assert not restamped_replaced_workloads(
            before=_stamps(orchestrator=None, rollout_executor=None),
            after=_stamps(rollout_executor=None),
            workloads=STAMPED,
        )

    def test_a_workload_the_read_did_not_return_does_not_count(self):
        """A workload absent from the read has not been seen carrying anything, which is not evidence."""
        assert not restamped_replaced_workloads(before={}, after={ORCHESTRATOR: "t2"}, workloads=STAMPED)

    def test_a_stamp_the_run_never_carried_on_another_workload_does_not_count(self):
        """Only the two workloads a take-over replaces are stamped; anything else says nothing about it."""
        assert not restamped_replaced_workloads(
            before=_stamps(orchestrator=None, rollout_executor=None),
            after={**_stamps(orchestrator=None, rollout_executor=None), TRAINER: "t2"},
            workloads=STAMPED,
        )

    def test_a_stamp_read_that_failed_does_not_count(self):
        """A kubectl call that did not answer must not become a take-over that never landed."""
        assert not restamped_replaced_workloads(before=_stamps(), after=None, workloads=STAMPED)


def _stamps(
    at: str = "t2", *, orchestrator: str | None = "", rollout_executor: str | None = ""
) -> dict[str, str | None]:
    return {
        ORCHESTRATOR: at if orchestrator == "" else orchestrator,
        ROLLOUT_EXECUTOR: at if rollout_executor == "" else rollout_executor,
    }
