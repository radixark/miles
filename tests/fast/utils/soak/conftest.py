import pytest
from tests.fast.utils.soak.utils import typed_cell
from tests.utils.soak.state import SoakActionRequest

from miles.utils.workers.cell_operations.base import FaultTarget


@pytest.fixture
def batch_request() -> SoakActionRequest:
    requests = [
        SoakActionRequest(
            request_id=f"victim-{index}",
            target=typed_cell(f"rollout-{index}", "rollout"),
            form_name="inject_fault:sigkill",
            harms_cell=True,
            fault_target=FaultTarget(cell_id=f"rollout-{index}", sub_index=0, workers_hash="generation-0"),
        )
        for index in range(2)
    ]
    return requests[0].model_copy(
        update={
            "form_name": "remote_hook:trainer_before_weight_send:inject_fault:sigkill:0ms:all",
            "additional_requests": requests[1:],
            "hook_trigger": FaultTarget(cell_id="actor-0", sub_index=0, workers_hash="generation-0"),
        }
    )
