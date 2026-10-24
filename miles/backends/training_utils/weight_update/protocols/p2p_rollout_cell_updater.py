import logging
from collections.abc import Callable

from miles.backends.training_utils.weight_update import checksum_utils
from miles.backends.training_utils.weight_update.protocols.transports.remote_shard import RemoteShard
from miles.backends.training_utils.weight_update.rollout_cell_updater import _RolloutCellUpdater

logger = logging.getLogger(__name__)


# This class, like the rest of the p2p weight-update code, is kept deliberately naive until yueming's refactor part 2 reshapes it.
class _P2PRolloutCellUpdater(_RolloutCellUpdater):
    def run_write(
        self,
        write: Callable[[], None],
        *,
        remote_shard: RemoteShard,
        sent_checksums: dict[str, str] | None,
    ) -> None:
        if self.is_errored:
            logger.warning(f"[P2P-Shared] skipping a queued write to rollout cell {self.cell_id}")
            return
        try:
            write()
            if sent_checksums is not None:
                self._verify_transfer_checksums(remote_shard=remote_shard, sent_checksums=sent_checksums)
        except Exception as error:
            self.mark_errored(error)
            raise

    def _verify_transfer_checksums(self, *, remote_shard: RemoteShard, sent_checksums: dict[str, str]) -> None:
        engine_body = self.submit_client_call(
            "check_weights", action="raw_checksum", names=list(sent_checksums), selector=remote_shard.runner_role
        ).result()
        if engine_body is None:
            return
        checksum_utils.verify_transfer_checksums(
            sent_checksums=(
                sent_checksums
                if remote_shard.runner_role == "target"
                else {f"{remote_shard.runner_role}.{name}": checksum for name, checksum in sent_checksums.items()}
            ),
            engine_body=engine_body,
            cell_id=self.cell_id,
            rank=remote_shard.rollout_engine_rank,
        )
