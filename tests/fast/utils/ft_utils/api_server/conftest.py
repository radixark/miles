from miles.utils.test_utils.fault_injector import FailureMode
from miles.utils.workers.cell_operations.base import FaultTarget, StaleFaultTargetError
from miles.utils.workers.worker_provider.base import CellInfo


class PinnedCellOperations:
    def __init__(self, *, dispatch_error: Exception | None = None) -> None:
        self.dispatch_error = dispatch_error
        self.target = FaultTarget(
            cell_id="actor-0", sub_index=0, workers_hash="generation-0", boot_uuid="boot-0", pod_uid="pod-0"
        )
        self.dispatched: list[FaultTarget] = []
        self.modes: list[FailureMode] = []
        self.request_ids: list[str | None] = []
        self.receipt_urls: list[str | None] = []

    async def cell_infos(self, *, pool_ids: list[str]) -> dict[str, CellInfo]:
        return {
            "actor-0": CellInfo(
                cell_id="actor-0",
                pool_id="actor",
                alive=True,
                worker_names=["actor-0-0"],
                workers_hash=self.target.workers_hash,
                meta={},
            )
        }

    async def observe_fault_target(self, *, cell_id: str, sub_index: int) -> FaultTarget:
        if (cell_id, sub_index) != (self.target.cell_id, self.target.sub_index):
            raise StaleFaultTargetError("No matching live worker")
        return self.target

    async def inject_fault(
        self,
        *,
        cell_id: str,
        mode: FailureMode,
        sub_index: int,
        expected_target: FaultTarget | None = None,
        request_id: str | None = None,
        receipt_url: str | None = None,
    ) -> None:
        if expected_target != self.target or (cell_id, sub_index) != (self.target.cell_id, self.target.sub_index):
            raise StaleFaultTargetError("Target changed")
        self.dispatched.append(expected_target)
        self.modes.append(mode)
        self.request_ids.append(request_id)
        self.receipt_urls.append(receipt_url)
        if self.dispatch_error is not None:
            raise self.dispatch_error
