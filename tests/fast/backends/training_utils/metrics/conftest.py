import pytest
import torch.distributed as dist
from tests.fast.backends.training_utils.loss.loss_test_utils import make_parallel_state

from miles.backends.training_utils import parallel


@pytest.fixture
def single_process_parallel_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(parallel, "_parallel_state", None)
    monkeypatch.setattr(dist, "is_initialized", lambda: False)
    make_parallel_state()
