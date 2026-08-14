import logging
from pathlib import Path

from miles.utils.file_utils import atomic_write_text
from miles.utils.pydantic_utils import FrozenStrictBaseModel

logger = logging.getLogger(__name__)

MULTI_POLICY_STATE_DIRNAME = "multi_policy_state"


class MultiPolicyCheckpointState(FrozenStrictBaseModel):
    leader_model_id: str
    rollout_ids: dict[str, int]

    def save(self, save_dir: Path) -> None:
        path = _state_path(save_dir, leader_rollout_id=self.rollout_ids[self.leader_model_id])
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(path, self.model_dump_json())
        logger.info(f"Saved multi policy checkpoint state {self.rollout_ids} to {path}")

    @classmethod
    def load(cls, save_dir: Path, *, leader_rollout_id: int) -> "MultiPolicyCheckpointState | None":
        path = _state_path(save_dir, leader_rollout_id=leader_rollout_id)
        if not path.exists():
            return None
        ans = cls.model_validate_json(path.read_text(encoding="utf-8"))
        recorded = ans.rollout_ids[ans.leader_model_id]
        assert recorded == leader_rollout_id, (
            f"{path} records its leader policy {ans.leader_model_id!r} at rollout {recorded}, but it is the "
            f"record of rollout {leader_rollout_id}"
        )
        return ans


def _state_path(save_dir: Path, *, leader_rollout_id: int) -> Path:
    return Path(save_dir) / MULTI_POLICY_STATE_DIRNAME / f"{leader_rollout_id}.json"
