from datetime import datetime, timedelta, timezone
from pathlib import Path

from miles.utils.audit_utils.event_logger.models import TrainerCheckpointEvent
from miles.utils.audit_utils.process_identity import TrainerControllerProcessIdentity
from miles.utils.audit_utils.sample_ownership import checkpoint_ids_of


class TestCheckpointIds:
    def test_a_repeated_save_keeps_policy_and_role_boundaries_despite_clock_offsets(self, tmp_path: Path) -> None:
        """Each policy and role selects its own latest completed save identity."""
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        events = [
            TrainerCheckpointEvent(
                timestamp=start + timedelta(seconds=seconds),
                source=TrainerControllerProcessIdentity(trainer_id=f"{policy}-{role}", model_id=policy),
                rollout_id=rollout_id,
                role=role,
                checkpoint_id=checkpoint_id,
                cell_index=cell_index,
                rank_count=2,
                alive_cell_indices=[cell_index],
            )
            for policy, role, seconds, rollout_id, checkpoint_id, cell_index in [
                ("solver", "actor", 10, 3, "old", 0),
                ("solver", "actor", 11, 3, "new", 1),
                ("solver", "critic", -100, 3, "critic", 0),
                ("verifier", "actor", 999, 3, "other-policy", 0),
                ("solver", "actor", 12, 4, "next-rollout", 1),
            ]
        ]
        (tmp_path / "events.jsonl").write_text("\n".join(event.model_dump_json() for event in reversed(events)))

        assert checkpoint_ids_of(event_dir=tmp_path, rollout_id=3, trainer_model_id="solver") == {
            "actor": "new",
            "critic": "critic",
        }
