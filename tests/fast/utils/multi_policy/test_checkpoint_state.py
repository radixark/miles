import pytest

from miles.utils.multi_policy.checkpoint_state import MULTI_POLICY_STATE_DIRNAME, MultiPolicyCheckpointState


class TestMultiPolicyCheckpointState:
    def test_the_global_state_is_indexed_by_the_leader_rollout_id(self, tmp_path):
        """DataSource and RolloutExecutor are global, so one index must identify the whole run."""
        state = MultiPolicyCheckpointState(leader_model_id="a", rollout_ids={"a": 7, "b": 5})

        state.save(tmp_path)

        assert (tmp_path / MULTI_POLICY_STATE_DIRNAME / "7.json").exists()
        assert MultiPolicyCheckpointState.load(tmp_path, leader_rollout_id=7) == state

    def test_the_policies_are_recorded_at_the_positions_they_really_reached(self, tmp_path):
        """Fully async policies run at their own pace, so their rollout ids need not agree."""
        MultiPolicyCheckpointState(leader_model_id="a", rollout_ids={"a": 7, "b": 5}).save(tmp_path)

        assert MultiPolicyCheckpointState.load(tmp_path, leader_rollout_id=7).rollout_ids == {"a": 7, "b": 5}

    def test_a_state_of_another_rollout_is_not_read_back(self, tmp_path):
        """Each global checkpoint has its own record, so a resume must not pick up a neighbour's."""
        MultiPolicyCheckpointState(leader_model_id="a", rollout_ids={"a": 7, "b": 5}).save(tmp_path)

        assert MultiPolicyCheckpointState.load(tmp_path, leader_rollout_id=5) is None

    def test_a_missing_state_file_reads_as_none(self, tmp_path):
        """A run that never saved must start fresh instead of exploding on load."""
        assert MultiPolicyCheckpointState.load(tmp_path, leader_rollout_id=3) is None

    def test_a_record_that_disagrees_with_its_own_file_name_is_refused(self, tmp_path):
        """The file name is how a resume finds the record, so a record naming another rollout is corrupt."""
        MultiPolicyCheckpointState(leader_model_id="a", rollout_ids={"a": 7, "b": 5}).save(tmp_path)
        path = tmp_path / MULTI_POLICY_STATE_DIRNAME / "9.json"
        path.write_text(
            MultiPolicyCheckpointState(leader_model_id="a", rollout_ids={"a": 7, "b": 5}).model_dump_json()
        )

        with pytest.raises(AssertionError, match="record of rollout 9"):
            MultiPolicyCheckpointState.load(tmp_path, leader_rollout_id=9)
