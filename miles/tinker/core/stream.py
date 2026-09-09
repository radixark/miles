"""Per-model command stream: ordering, idempotency, window compilation.

Rectifies the HTTP world (out-of-order arrival, retries) into the stream's
semantic structure: runs of forward commands form commutative windows whose
datums may execute in any order, in any grouping, interleaved with other
models; every other command is a barrier that waits for its window. The
stream never touches the trainer: the planner decides what runs when.
"""

from collections import deque
from dataclasses import dataclass, field

from miles.tinker.core.types import Command

WINDOW_OPS = ("forward_backward", "forward_only")


@dataclass
class PendingRequest:
    """One submitted command and its completion accounting."""

    command: Command
    datums: list[dict] = field(default_factory=list)  # window commands only
    issued: int = 0
    outputs: list[dict | None] = field(default_factory=list)
    remaining: int = 0

    @property
    def is_window(self) -> bool:
        return self.command.op in WINDOW_OPS

    def record_output(self, local_index: int, output: dict) -> bool:
        """Store one datum's result; True once every datum has reported."""
        self.outputs[local_index] = output
        self.remaining -= 1
        return self.remaining == 0

    def pack_key(self) -> tuple:
        """Rows pack into one BatchOp only within the same (op, loss_fn, config)."""
        if self.command.op == "forward_only":
            return ("forward_only",)
        config = self.command.payload.get("loss_fn_config") or {}
        return ("forward_backward", self.command.payload["loss_fn"], tuple(sorted(config.items())))


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
            if pending.is_window:
                pending.datums = next_command.payload["datums"]
                pending.remaining = len(pending.datums)
                pending.outputs = [None] * len(pending.datums)
                if not pending.datums:
                    continue  # admission-rejected: the position is consumed, nothing runs
            self.queue.append(pending)

    def open_window(self) -> list[PendingRequest]:
        """The leading run of window commands; their datums are all issuable."""
        window = []
        for pending in self.queue:
            if not pending.is_window:
                break
            window.append(pending)
        return window

    def ready_barrier(self) -> PendingRequest | None:
        """The head barrier, executable once its window fully completed."""
        if self.queue and not self.queue[0].is_window:
            return self.queue[0]
        return None

    def finish(self, pending: PendingRequest) -> None:
        self.queue.remove(pending)
