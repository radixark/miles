"""Execution planning: turn ready work from all streams into the next unit.

The single decision point between streams and the trainer. Policy is
arrival-order greedy: the oldest ready item goes first; a datum pulls every
compatible ready datum (same kind / loss_fn / config) across all models into
one WorkUnit up to the token budget — cross-tenant packing and large-request
splitting are both just this bin-packer at datum granularity. Ready optim
barriers of different models merge into one BarrierUnit.
"""

from dataclasses import dataclass

from miles.tinker.core.stream import ModelStream, PendingRequest


@dataclass
class RowRef:
    stream: ModelStream
    request: PendingRequest
    local_index: int

    @property
    def row(self) -> dict:
        return self.request.rows[self.local_index]

    @property
    def arrival(self) -> int:
        return self.request.command.arrival


@dataclass
class WorkUnit:
    kind: str  # forward_backward | forward_only
    loss_fn: str | None
    loss_fn_config: dict | None
    rows: list[RowRef]


@dataclass
class BarrierUnit:
    kind: str  # optim_step | save_state | load_state | save_weights_for_sampler
    entries: list[tuple[ModelStream, PendingRequest]]


class Planner:
    def __init__(self, unit_token_budget: int) -> None:
        self.unit_token_budget = unit_token_budget
        self._streams: dict[str, ModelStream] = {}

    def add_stream(self, stream: ModelStream) -> None:
        self._streams[stream.model_id] = stream

    def remove_stream(self, model_id: str) -> None:
        del self._streams[model_id]

    def stream(self, model_id: str) -> ModelStream:
        return self._streams[model_id]

    def next_unit(self) -> WorkUnit | BarrierUnit | None:
        rows = self._ready_rows()
        barriers = self._ready_barriers()

        oldest_row = min(rows, key=lambda ref: ref.arrival) if rows else None
        oldest_barrier = min(barriers, key=lambda e: e[1].command.arrival) if barriers else None
        if oldest_row is None and oldest_barrier is None:
            return None
        if oldest_barrier is not None and (
            oldest_row is None or oldest_barrier[1].command.arrival < oldest_row.arrival
        ):
            return self._build_barrier(oldest_barrier, barriers)
        return self._build_work(oldest_row, rows)

    def _ready_rows(self) -> list[RowRef]:
        rows = []
        for stream in self._streams.values():
            for request in stream.open_window():
                rows.extend(RowRef(stream, request, index) for index in range(request.issued, len(request.rows)))
        return rows

    def _ready_barriers(self) -> list[tuple[ModelStream, PendingRequest]]:
        return [
            (stream, barrier) for stream in self._streams.values() if (barrier := stream.ready_barrier()) is not None
        ]

    def _build_work(self, seed: RowRef, rows: list[RowRef]) -> WorkUnit:
        loss_class = seed.request.loss_class()
        compatible = sorted(
            (ref for ref in rows if ref.request.loss_class() == loss_class),
            key=lambda ref: (ref.arrival, ref.local_index),
        )
        picked: list[RowRef] = []
        tokens = 0
        for ref in compatible:
            row_tokens = len(ref.row["tokens"])
            if picked and tokens + row_tokens > self.unit_token_budget:
                break
            picked.append(ref)
            tokens += row_tokens
        for ref in picked:
            ref.request.issued += 1
        command = seed.request.command
        return WorkUnit(
            kind=command.kind,
            loss_fn=command.payload.get("loss_fn"),
            loss_fn_config=command.payload.get("loss_fn_config"),
            rows=picked,
        )

    def _build_barrier(
        self,
        oldest: tuple[ModelStream, PendingRequest],
        barriers: list[tuple[ModelStream, PendingRequest]],
    ) -> BarrierUnit:
        kind = oldest[1].command.kind
        if kind == "optim_step":
            # optim barriers of different models step in one trainer call
            entries = [(stream, barrier) for stream, barrier in barriers if barrier.command.kind == kind]
        else:
            entries = [oldest]
        return BarrierUnit(kind=kind, entries=entries)
