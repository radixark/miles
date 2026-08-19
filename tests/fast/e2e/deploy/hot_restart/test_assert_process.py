import pytest
from tests.e2e.deploy.conftest_deploy.hot_restart.assert_process import (
    _compute_trainer_boot_uuids,
    assert_run_watched_closely,
    assert_trainer_not_rebooted,
)
from tests.e2e.deploy.conftest_deploy.hot_restart.cluster_observer import POD_KIND
from tests.fast.e2e.deploy.hot_restart.cluster_facts import cluster_snapshot
from tests.fast.e2e.deploy.hot_restart.restart_facts import (
    evidence_of,
    quiet_run_snapshot,
    restart_snapshot,
    restarted_snapshot,
    two_restarts,
)


class TestAssertTheTrainerNeverRebooted:
    def test_one_boot_uuid_across_every_script_passes(self):
        """The trainer process outliving both scripts is what makes this a hot restart at all."""
        assert_trainer_not_rebooted(evidence_of(snapshots=two_restarts(), records=[]))

    def test_a_second_boot_uuid_fails(self):
        """A restarted rpc server serves a new process, whatever its pod's uid says."""
        snapshots = [*two_restarts(), restart_snapshot(uid_of_pod={}, stamp_of_workload={}, boot_uuid="boot-b")]

        with pytest.raises(AssertionError, match="boot uuid"):
            assert_trainer_not_rebooted(evidence_of(snapshots=snapshots, records=[]))

    def test_a_trainer_never_reached_again_after_a_take_over_fails(self):
        """A uuid only ever read before the last take-over says nothing about what survived it."""
        snapshots = [
            quiet_run_snapshot(),
            restarted_snapshot(stamp="t1", uid_suffix="2"),
            restarted_snapshot(stamp="t2", uid_suffix="3", boot_uuid=None),
        ]

        with pytest.raises(AssertionError, match="never reached after"):
            assert_trainer_not_rebooted(evidence_of(snapshots=snapshots, records=[]))

    def test_a_trainer_the_rpc_server_never_answered_for_fails_as_such(self):
        """One uuid out of zero readings used to fail as "the uuid(s) [] , not one", which reads as a reboot."""
        never_answered = [restart_snapshot(uid_of_pod={}, stamp_of_workload={}, boot_uuid=None) for _ in range(3)]

        with pytest.raises(AssertionError, match="never reached across 3 observation"):
            assert_trainer_not_rebooted(evidence_of(snapshots=never_answered, records=[]))


class TestAssertTheRunWasWatchedCloselyEnough:
    def test_a_run_observed_throughout_passes(self):
        """Every verdict about pods is worth exactly what the observations behind it cost."""
        assert_run_watched_closely(evidence_of(snapshots=two_restarts(), records=[]))

    def test_a_run_nobody_looked_at_twice_fails(self):
        """Every workload verdict is a comparison between observations, and one of them says nothing."""
        with pytest.raises(AssertionError, match="never looked at twice"):
            assert_run_watched_closely(evidence_of(snapshots=[quiet_run_snapshot()], records=[]))

    def test_a_run_whose_observations_never_saw_a_whole_release_fails(self):
        """A cluster nobody could read whole looks exactly like a cluster where nothing was replaced."""
        partial = [one.model_copy(update={"reads_missing": (POD_KIND,)}) for one in two_restarts()]

        with pytest.raises(AssertionError, match="never looked at twice"):
            assert_run_watched_closely(evidence_of(snapshots=partial, records=[]))

    def test_a_run_nothing_ever_observed_fails(self):
        """Evidence written before anything was collected would pass every count there is."""
        with pytest.raises(AssertionError, match="never looked at twice"):
            assert_run_watched_closely(evidence_of(snapshots=[], records=[]))

    def test_a_run_read_whole_only_half_as_often_as_it_was_tried_fails(self):
        """A cluster answering half the time looks exactly like a cluster where nothing was replaced."""
        with pytest.raises(AssertionError, match=r"attempt\(s\), under"):
            assert_run_watched_closely(
                evidence_of(snapshots=two_restarts(), records=[], observation_attempts=40, observation_failures=37)
            )

    def test_evidence_written_before_the_counts_existed_still_faces_the_absolute_floor(self):
        """Older dumps carry no attempt count, and must not become unverifiable because of it."""
        assert_run_watched_closely(evidence_of(snapshots=two_restarts(), records=[], observation_attempts=0))


class TestComputeTrainerBootUuids:
    def test_a_trainer_that_outlived_every_script_answers_with_one_boot_uuid(self):
        """A second uuid means the process a hot restart promised to keep alive was replaced."""
        snapshots = [
            cluster_snapshot(pods=[], workloads=[], trainer_boot_uuid="boot-a"),
            cluster_snapshot(pods=[], workloads=[], trainer_boot_uuid=None),
            cluster_snapshot(pods=[], workloads=[], trainer_boot_uuid="boot-a"),
        ]

        assert _compute_trainer_boot_uuids(snapshots) == {"boot-a"}
