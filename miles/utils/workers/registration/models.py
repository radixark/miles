from __future__ import annotations


from miles.utils.pydantic_utils import FrozenStrictBaseModel
from miles.utils.workers.worker_info import WorkerInfo
from miles.utils.workers.worker_provider.base import CellInfo


class RegisteredCellInfo(FrozenStrictBaseModel):
    reporter_id: str
    info: CellInfo
    workers: list[WorkerInfo]


class RegistrationSnapshot(FrozenStrictBaseModel):
    reporter_id: str
    sequence_number: int
    cells: list[RegisteredCellInfo]
