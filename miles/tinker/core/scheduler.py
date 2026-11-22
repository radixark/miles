"""Choose the earliest ready seed, then pack compatible work across model queues."""

from dataclasses import dataclass

from miles.tinker.core.model_queue import ModelRequestQueue, PendingRequest
from miles.tinker.core.types import CommandOp


@dataclass
class DatumRef:
    """Pointer to ``request.datums[local_index]``; the datum's output is written back through it."""

    model_queue: ModelRequestQueue
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
    """One barrier call after all preceding batch requests in its model queue finish."""

    op: CommandOp  # OPTIM_STEP | SAVE_STATE | LOAD_STATE | SAVE_WEIGHTS_FOR_SAMPLER
    entries: list[tuple[ModelRequestQueue, PendingRequest]]


class RequestScheduler:
    def __init__(self, batch_token_budget: int) -> None:
        self.batch_token_budget = batch_token_budget
        self._model_queues: dict[str, ModelRequestQueue] = {}

    def add_model_queue(self, model_queue: ModelRequestQueue) -> None:
        self._model_queues[model_queue.model_id] = model_queue

    def remove_model_queue(self, model_id: str) -> None:
        del self._model_queues[model_id]

    def model_queue(self, model_id: str) -> ModelRequestQueue:
        return self._model_queues[model_id]

    def ready_rejections(self) -> list[tuple[ModelRequestQueue, PendingRequest]]:
        return [
            (model_queue, model_queue.queue[0])
            for model_queue in self._model_queues.values()
            if model_queue.queue and model_queue.queue[0].command.validation_error is not None
        ]

    def schedule_next(self) -> BatchUnit | BarrierUnit | None:
        """Arrival order selects the seed; compatible work may overtake intervening requests."""
        datums = self._ready_datums()
        barriers = self._ready_barriers()

        datum_seed = min(datums, key=lambda ref: ref.arrival) if datums else None
        barrier_seed = min(barriers, key=lambda e: e[1].command.arrival) if barriers else None
        if datum_seed is None and barrier_seed is None:
            return None
        if barrier_seed is not None and (datum_seed is None or barrier_seed[1].command.arrival < datum_seed.arrival):
            return self._merge_barriers(barrier_seed, barriers)
        return self._pack_batch(datum_seed, datums)

    def _ready_datums(self) -> list[DatumRef]:
        datums = []
        for model_queue in self._model_queues.values():
            for request in model_queue.open_batch_run():
                datums.extend(
                    DatumRef(model_queue, request, index)
                    for index in range(request.num_issued_datums, len(request.datums))
                )
        return datums

    def _ready_barriers(self) -> list[tuple[ModelRequestQueue, PendingRequest]]:
        return [
            (model_queue, barrier)
            for model_queue in self._model_queues.values()
            if (barrier := model_queue.ready_barrier()) is not None
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
            ref.request.num_issued_datums += 1
        command = seed.request.command
        return BatchUnit(
            op=command.op,
            loss_fn=command.payload.get("loss_fn"),
            loss_fn_config=command.payload.get("loss_fn_config"),
            datums=picked,
        )

    def _merge_barriers(
        self,
        seed: tuple[ModelRequestQueue, PendingRequest],
        barriers: list[tuple[ModelRequestQueue, PendingRequest]],
    ) -> BarrierUnit:
        op = seed[1].command.op
        if op == CommandOp.OPTIM_STEP:
            # optim barriers of different models step in one trainer call
            entries = [(model_queue, barrier) for model_queue, barrier in barriers if barrier.command.op == op]
        else:
            entries = [seed]
        return BarrierUnit(op=op, entries=entries)
