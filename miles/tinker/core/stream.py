"""Per-model command stream: ordering, idempotency, batch/barrier structure.

Rectifies the HTTP world (out-of-order arrival, retries) into runs of batch
ops — whose datums may execute in any order, grouping, or interleaving —
separated by barriers that each wait for every batch op ahead of it. The
stream never touches the trainer: the planner decides what runs when.
"""

from collections import deque
from dataclasses import dataclass, field

from miles.tinker.core.types import Command


@dataclass
class PendingRequest:
    """One submitted command and its completion accounting."""

    command: Command
    datums: list[dict] = field(default_factory=list)  # batch ops only
    issued: int = 0
    outputs: list[dict | None] = field(default_factory=list)
    remaining: int = 0

    @property
    def is_batch_op(self) -> bool:
        return self.command.op.is_batch()

    def record_output(self, local_index: int, output: dict) -> bool:
        """Store one datum's result; True once every datum has reported."""
        self.outputs[local_index] = output
        self.remaining -= 1
        return self.remaining == 0

    def pack_key(self) -> tuple:
        """Datums pack into one BatchUnit only within the same (op, loss_fn, config)."""
        config = self.command.payload.get("loss_fn_config") or {}
        return (self.command.op, self.command.payload["loss_fn"], tuple(sorted(config.items())))


class ModelStream:
    def __init__(self, model_id: str, tenant: str, slot: int) -> None:
        self.model_id = model_id
        self.tenant = tenant
        self.slot = slot
        # seq_ids are 1-based; watermark = last seq_id accepted into the queue
        self.watermark = 0
        self.arrivals: dict[int, Command] = {}
        self.request_id_by_seq: dict[int, str] = {}
        self.queue: deque[PendingRequest] = deque()

    def submit(self, command: Command) -> None:
        """Accept one deduplicated command; feed the queue in seq order."""
        self.arrivals[command.seq_id] = command
        while (next_command := self.arrivals.pop(self.watermark + 1, None)) is not None:
            self.watermark += 1
            pending = PendingRequest(command=next_command)
            if pending.is_batch_op:
                pending.datums = next_command.payload["datums"]
                pending.remaining = len(pending.datums)
                pending.outputs = [None] * len(pending.datums)
                if not pending.datums:
                    continue  # admission-rejected: the position is consumed, nothing runs
            self.queue.append(pending)

    def open_batch_run(self) -> list[PendingRequest]:
        """The leading run of batch-op commands; their datums are all issuable."""
        run = []
        for pending in self.queue:
            if not pending.is_batch_op:
                break
            run.append(pending)
        return run

    def ready_barrier(self) -> PendingRequest | None:
        """The head barrier, executable once every batch op ahead of it completed."""
        if self.queue and not self.queue[0].is_batch_op:
            return self.queue[0]
        return None

    def finish(self, pending: PendingRequest) -> None:
        self.queue.remove(pending)
