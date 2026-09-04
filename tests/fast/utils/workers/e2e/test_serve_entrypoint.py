import socket

from tests.fast.utils.workers.e2e.e2e_worker import WORKER_FACTORY_ERROR

RAISING_WORKER_PATH = "tests.fast.utils.workers.e2e.e2e_worker.make_raising_worker"
EXIT_TIMEOUT_SECONDS = 60.0


class TestWorkerFactoryFailure:
    def test_raising_worker_factory_fails_before_binding_the_port(self, spawn) -> None:
        """A worker factory that raises fails startup before the port is bound and reports its own error."""
        server = spawn(wait=False, worker_path=RAISING_WORKER_PATH)
        exit_code = server.wait(EXIT_TIMEOUT_SECONDS)

        assert exit_code is not None and exit_code != 0, f"server did not exit:\n{server.logs()}"
        assert (
            WORKER_FACTORY_ERROR in server.logs()
        ), f"startup failed before reaching the worker factory:\n{server.logs()}"
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", server.port))
