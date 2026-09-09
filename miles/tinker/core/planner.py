"""Turns ready work from all streams into the next trainer call, one at a time.

Arrival-order greedy: the oldest ready item goes first; datums pack across
requests and models up to the token budget, ready optim barriers merge.
"""

from dataclasses import dataclass

from miles.tinker.core.stream import ModelStream, PendingRequest
from miles.tinker.core.types import CommandOp


@dataclass
class DatumRef:
    """Pointer to ``request.datums[local_index]``; the datum's output is written back through it."""

    stream: ModelStream
    request: PendingRequest
    local_index: int

    @property
    def datum(self) -> dict:
        return self.request.datums[self.local_index]

    @property
    def arrival(self) -> int:
        return self.request.command.arrival


@dataclass
class BatchUnit:
    """One forward pass on the trainer: datums packed from any number of requests."""

    op: CommandOp  # FORWARD_BACKWARD | FORWARD_ONLY
    loss_fn: str | None
    loss_fn_config: dict | None
    datums: list[DatumRef]


@dataclass
class BarrierUnit:
    """One non-forward trainer call, run only after its stream's window drained."""

    op: CommandOp  # OPTIM_STEP | SAVE_STATE | LOAD_STATE | SAVE_WEIGHTS_FOR_SAMPLER
    entries: list[tuple[ModelStream, PendingRequest]]


class Planner:
    def __init__(self, batch_token_budget: int) -> None:
        self.batch_token_budget = batch_token_budget
        self._streams: dict[str, ModelStream] = {}

    def add_stream(self, stream: ModelStream) -> None:
        self._streams[stream.model_id] = stream

    def remove_stream(self, model_id: str) -> None:
        del self._streams[model_id]

    def stream(self, model_id: str) -> ModelStream:
        return self._streams[model_id]

    def next_to_run(self) -> BatchUnit | BarrierUnit | None:
        datums = self._ready_datums()
        barriers = self._ready_barriers()

        oldest_datum = min(datums, key=lambda ref: ref.arrival) if datums else None
        oldest_barrier = min(barriers, key=lambda e: e[1].command.arrival) if barriers else None
        if oldest_datum is None and oldest_barrier is None:
            return None
        if oldest_barrier is not None and (
            oldest_datum is None or oldest_barrier[1].command.arrival < oldest_datum.arrival
        ):
            return self._merge_barriers(oldest_barrier, barriers)
        return self._pack_batch(oldest_datum, datums)

    def _ready_datums(self) -> list[DatumRef]:
        datums = []
        for stream in self._streams.values():
            for request in stream.open_batch_run():
                datums.extend(DatumRef(stream, request, index) for index in range(request.issued, len(request.datums)))
        return datums

    def _ready_barriers(self) -> list[tuple[ModelStream, PendingRequest]]:
        return [
            (stream, barrier) for stream in self._streams.values() if (barrier := stream.ready_barrier()) is not None
        ]

    def _pack_batch(self, seed: DatumRef, datums: list[DatumRef]) -> BatchUnit:
        pack_key = seed.request.pack_key()
        compatible = sorted(
            (ref for ref in datums if ref.request.pack_key() == pack_key),
            key=lambda ref: (ref.arrival, ref.local_index),
        )
        picked: list[DatumRef] = []
        tokens = 0
        for ref in compatible:
            datum_tokens = len(ref.datum["tokens"])
            if picked and tokens + datum_tokens > self.batch_token_budget:
                break
            picked.append(ref)
            tokens += datum_tokens
        for ref in picked:
            ref.request.issued += 1
        command = seed.request.command
        return BatchUnit(
            op=command.op,
            loss_fn=command.payload.get("loss_fn"),
            loss_fn_config=command.payload.get("loss_fn_config"),
            datums=picked,
        )

    def _merge_barriers(
        self,
        oldest: tuple[ModelStream, PendingRequest],
        barriers: list[tuple[ModelStream, PendingRequest]],
    ) -> BarrierUnit:
        op = oldest[1].command.op
        if op == CommandOp.OPTIM_STEP:
            # optim barriers of different models step in one trainer call
            entries = [(stream, barrier) for stream, barrier in barriers if barrier.command.op == op]
        else:
            entries = [oldest]
        return BarrierUnit(op=op, entries=entries)
