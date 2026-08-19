from __future__ import annotations

from miles.utils.workers.worker_spec import BaseWorkerSpec

RENDERED_CELL_INDEX = 0

LEADER_ADDRESS_PLACEHOLDER = "$(LWS_LEADER_ADDRESS)"

WORKER_INDEX_SENTINEL = 987654321
_WORKER_INDEX_PLACEHOLDER = "$(LWS_WORKER_INDEX)"


def with_worker_index(argv: list[str], spec: BaseWorkerSpec) -> list[str]:
    sentinel = str(WORKER_INDEX_SENTINEL)
    embedded = [argument for argument in argv if sentinel in argument and argument != sentinel]
    assert not embedded, (
        f"Spec '{spec.name}' builds {embedded} out of its pod index; kubelet substitutes a whole "
        f"argument, so the index has to reach the command unchanged"
    )
    return [_WORKER_INDEX_PLACEHOLDER if argument == sentinel else argument for argument in argv]
