class ConnStatusManager:
    def __init__(self) -> None:
        self._initialized: bool = False
        self._trainer_stale: bool = False

    def needs_reconnect(self) -> bool:
        return (not self._initialized) or self._trainer_stale

    def mark_trainer_stale(self) -> None:
        self._trainer_stale = True

    def mark_reconnected(self) -> None:
        self._initialized = True
        self._trainer_stale = False
