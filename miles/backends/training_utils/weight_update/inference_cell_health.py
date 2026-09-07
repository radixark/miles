import logging
from collections.abc import Sequence

import torch.distributed as dist

logger = logging.getLogger(__name__)


class InferenceCellHealth:
    def __init__(self, cell_ids: Sequence[str] = ()) -> None:
        assert len(set(cell_ids)) == len(
            cell_ids
        ), f"every inference cell must appear once in the update, got {list(cell_ids)}"
        self._cell_ids: tuple[str, ...] = tuple(cell_ids)
        self._errors: dict[str, BaseException] = {}

    @property
    def cell_ids(self) -> tuple[str, ...]:
        return self._cell_ids

    @property
    def errored_cell_ids(self) -> list[str]:
        return sorted(self._errors)

    @property
    def healthy_cell_ids(self) -> list[str]:
        return [cell_id for cell_id in self._cell_ids if cell_id not in self._errors]

    def is_errored(self, cell_id: str) -> bool:
        return cell_id in self._errors

    def error_of(self, cell_id: str) -> BaseException | None:
        return self._errors.get(cell_id)

    def synchronize(self, group) -> None:
        reported: list = [None] * dist.get_world_size(group=group)
        dist.all_gather_object(reported, self.errored_cell_ids, group=group)

        for rank, cell_ids in enumerate(reported):
            for cell_id in cell_ids:
                if not self.is_errored(cell_id):
                    self.mark_errored(cell_id, RuntimeError(f"trainer rank {rank} failed to update cell {cell_id}"))

    def mark_errored(self, cell_id: str, error: BaseException) -> None:
        assert cell_id in self._cell_ids, f"cell {cell_id} is not part of this weight update, got {self._cell_ids}"
        if cell_id in self._errors:
            logger.warning(f"inference cell {cell_id} failed again, keeping the first error", exc_info=error)
            return
        self._errors[cell_id] = error
        logger.error(f"inference cell {cell_id} can no longer be updated", exc_info=error)
