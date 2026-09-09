from copy import deepcopy


class ConnStatusManager:
    def __init__(self) -> None:
        self._initialized: bool = False
        self._trainer_stale: bool = False
        self._rollout_snapshot_cell_id_to_hashes: dict[str, str] = {}

    def needs_reconnect(self, new_rollout_snapshot_cell_id_to_hashes: dict[str, str]) -> bool:
        return (
            (not self._initialized)
            or self._trainer_stale
            or (new_rollout_snapshot_cell_id_to_hashes != self._rollout_snapshot_cell_id_to_hashes)
        )

    def mark_trainer_stale(self) -> None:
        self._trainer_stale = True

    def mark_reconnected(self, new_rollout_snapshot_cell_id_to_hashes: dict[str, str]) -> None:
        self._initialized = True
        self._trainer_stale = False
        self._rollout_snapshot_cell_id_to_hashes = deepcopy(new_rollout_snapshot_cell_id_to_hashes)
