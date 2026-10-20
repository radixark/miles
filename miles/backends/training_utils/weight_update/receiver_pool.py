"""Independent protocol and connection state for rollout and dedicated eval pools."""

from contextlib import contextmanager

from miles.backends.training_utils.weight_update.conn_status import ConnStatusManager
from miles.backends.training_utils.weight_update.protocol import get_weight_transfer_protocol


class ReceiverPools:
    def __init__(self):
        self.active = "rollout"
        self._states = {}

    @contextmanager
    def activate(self, updater, pool_id):
        if pool_id not in {"rollout", "eval"}:
            raise ValueError(f"Unknown receiver pool: {pool_id!r}")
        if not updater.protocol.supports_receiver_pools:
            if pool_id != "rollout":
                raise ValueError("Separate eval synchronization requires a protocol with receiver-pool support")
            yield
            return
        if pool_id == self.active:
            yield
            return
        previous = self.active
        self._save(updater)
        if pool_id not in self._states:
            self._states[pool_id] = (
                get_weight_transfer_protocol(updater.args),
                ConnStatusManager(),
                set(),
            )
        self._select(updater, pool_id)
        try:
            yield
        finally:
            # Preserve state on failure; the caller must not retry an uncertain update.
            self._save(updater)
            self._select(updater, previous)

    def _save(self, updater):
        self._states[self.active] = (
            updater.protocol,
            updater.conn_status,
            updater._registered_adapters,
        )

    def _select(self, updater, pool_id):
        updater.protocol, updater.conn_status, updater._registered_adapters = self._states[pool_id]
        self.active = pool_id

    def mark_trainer_stale(self, updater):
        """A rebuilt trainer group invalidates every pool's connection."""
        updater.conn_status.mark_trainer_stale()
        for _, connection, _ in self._states.values():
            if connection is not updater.conn_status:
                connection.mark_trainer_stale()
