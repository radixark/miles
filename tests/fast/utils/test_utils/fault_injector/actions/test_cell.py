import pytest
from tests.fast.utils.test_utils.fault_injector.fakes import _CellOperations, _Controller

from miles.utils.test_utils.fault_injector.actions.base import FaultHookContext, FaultHookResources
from miles.utils.test_utils.fault_injector.actions.cell import StartCellAction, StopCellAction


class TestCellActionResources:
    async def test_stopping_without_cell_operations_fails_loudly(self) -> None:
        """A process that holds no cell operations must refuse a stop instead of skipping it."""
        with pytest.raises(AssertionError, match="needs the controller's cell operations"):
            await StopCellAction(cell_id="cell-0")(context=FaultHookContext(), resources=FaultHookResources())

    @pytest.mark.parametrize("with_controller,with_operations", [(False, True), (True, False), (False, False)])
    async def test_starting_without_the_controller_and_its_operations_fails_before_resuming(
        self, with_controller: bool, with_operations: bool
    ) -> None:
        """A start must refuse to resume a cell it cannot then observe rejoining."""
        operations = _CellOperations()
        resources = FaultHookResources(
            controller=_Controller(cell_ids=("cell-0",)) if with_controller else None,
            cell_operations=operations if with_operations else None,
        )
        with pytest.raises(AssertionError, match="needs the controller"):
            await StartCellAction(cell_id="cell-0")(context=FaultHookContext(), resources=resources)
        assert operations.started == []
