import pytest
from tests.fast.backends.training_utils.loss.loss_test_utils import make_parallel_state

from miles.backends.training_utils import parallel


@pytest.fixture
def native_single_rank(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(parallel, "_parallel_state", None)
    make_parallel_state()
