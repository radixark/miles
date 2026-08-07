import pytest

from miles.utils.external_utils.command_utils.helm_backend import staging

ONE_PAIR = ("/cluster-storage/ckpt:/scratch/ckpt",)
ROOT = "/scratch"


class TestParsePairs:
    def test_reads_a_source_and_a_destination(self):
        """A run states where a checkpoint comes from and where the pod wants it."""
        assert staging.parse_pairs(ONE_PAIR) == [("/cluster-storage/ckpt", "/scratch/ckpt")]

    def test_rejects_a_pair_missing_a_side(self):
        """Silently staging nothing would leave training to read the checkpoint over the network."""
        with pytest.raises(AssertionError, match="source:destination"):
            staging.parse_pairs(("/only-a-source",))


class TestStagingCommand:
    def test_puts_the_lock_where_the_colliding_pods_can_both_see_it(self):
        """A lock in the container's own filesystem is private per pod and serialises nothing."""
        command = staging.staging_command(ONE_PAIR, node_local_root=ROOT)

        assert "/scratch/.miles-staging-locks/ckpt.lock" in command
        assert "flock " in command

    def test_refuses_to_stage_without_a_node_local_volume(self):
        """Otherwise both the copy and the lock land in a filesystem no other pod can reach."""
        with pytest.raises(AssertionError, match="needs a node-local volume"):
            staging.staging_command(ONE_PAIR, node_local_root="")

    def test_refuses_a_destination_outside_that_volume(self):
        """A destination elsewhere is either pod-private or cluster-wide, and this lock guards neither."""
        with pytest.raises(AssertionError, match="not under the node-local root"):
            staging.staging_command(("/a:/cluster-storage/shared",), node_local_root=ROOT)

    def test_copies_into_the_destination(self):
        """The point of staging is that the copy exists locally before training opens it."""
        assert "rsync -a --info=progress2 /cluster-storage/ckpt/ /scratch/ckpt" in staging.staging_command(
            ONE_PAIR, node_local_root=ROOT
        )

    def test_locks_each_destination_separately(self):
        """Two unrelated directories would otherwise queue behind each other for no reason."""
        command = staging.staging_command(("/a/one:/scratch/one", "/b/two:/scratch/two"), node_local_root=ROOT)

        assert "/scratch/.miles-staging-locks/one.lock" in command
        assert "/scratch/.miles-staging-locks/two.lock" in command

    def test_stages_nothing_when_a_run_asks_for_nothing(self):
        """Most runs read straight from shared storage and must not gain a shell wrapper."""
        assert staging.staging_command((), node_local_root=ROOT) is None


class TestWithStaging:
    def test_leaves_a_command_alone_when_there_is_nothing_to_stage(self):
        """An unnecessary bash -c would make the training process a child rather than the pod's process."""
        assert staging.with_staging(["python", "-m", "x"], ()) == ["python", "-m", "x"]

    def test_hands_the_pod_over_to_training_after_staging(self):
        """exec keeps signals going to training rather than to a shell that ignores them."""
        wrapped = staging.with_staging(["python", "-m", "x"], ONE_PAIR, node_local_root=ROOT)

        assert wrapped[:2] == ["bash", "-c"]
        assert wrapped[2].endswith("&& exec python -m x")

    def test_keeps_an_argument_with_spaces_together(self):
        """A model arg holding json must survive being folded into a shell command."""
        wrapped = staging.with_staging(["python", "--kwargs", '{"a": 1}'], ONE_PAIR, node_local_root=ROOT)

        assert """exec python --kwargs '{"a": 1}'""" in wrapped[2]
