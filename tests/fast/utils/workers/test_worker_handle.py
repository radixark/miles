import pytest

from miles.utils.workers.rpc.client.handle import RpcWorkerHandle
from miles.utils.workers.worker_handle import BaseWorkerHandle, WorkerUnreachableError


class TestBaseWorkerHandle:
    def test_rpc_handle_implements_the_contract(self):
        """The rpc client is a worker handle, so callers can hold the base type."""
        assert issubclass(RpcWorkerHandle, BaseWorkerHandle)

    def test_incomplete_implementation_rejected(self):
        """A handle that does not implement wait_ready cannot be instantiated."""

        class Incomplete(BaseWorkerHandle):
            pass

        with pytest.raises(TypeError):
            Incomplete()


class TestWorkerUnreachableError:
    def test_is_plain_exception_without_submission_state(self) -> None:
        """The error carries only its message and standard exception state."""
        error = WorkerUnreachableError("boom")

        assert str(error) == "boom"
        assert not hasattr(error, "submitted")
