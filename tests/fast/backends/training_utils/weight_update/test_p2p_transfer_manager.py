import threading


def _blocking_task(started: threading.Event, release: threading.Event):
    def run() -> None:
        started.set()
        assert release.wait(timeout=30.0)

    return run


class TestSubmit:
    """Submission of P2P writes to the background executor."""

    def test_a_submitted_write_runs_in_the_background(self, p2p_transfer_utils) -> None:
        """Running the write inline would serialize every cell and defeat the fire-and-forget last rank."""
        manager = p2p_transfer_utils.P2PTransferManager(num_workers=2, transfer_timeout=30.0)
        started, release = threading.Event(), threading.Event()

        future = manager.submit(_blocking_task(started, release))
        assert started.wait(timeout=30.0)

        assert not future.done()

        release.set()
        assert future.result(timeout=30.0) is None

    def test_every_submitted_write_runs_even_when_they_outnumber_the_workers(self, p2p_transfer_utils) -> None:
        """More writes than workers are submitted per bucket, so the queued ones must run too."""
        manager = p2p_transfer_utils.P2PTransferManager(num_workers=2, transfer_timeout=30.0)
        lock = threading.Lock()
        finished: list[int] = []

        futures = []
        for index in range(6):

            def run(index: int = index) -> None:
                with lock:
                    finished.append(index)

            futures.append(manager.submit(run))

        for future in futures:
            future.result(timeout=30.0)

        assert sorted(finished) == list(range(6))

    def test_a_failing_write_surfaces_through_its_own_future(self, p2p_transfer_utils) -> None:
        """The submitting cell updater is the only place that knows whose failure this is."""
        manager = p2p_transfer_utils.P2PTransferManager(num_workers=2, transfer_timeout=30.0)

        def failing() -> None:
            raise RuntimeError("[P2P-Shared] Transfer failed for session s0, error: -1")

        future = manager.submit(failing)

        assert isinstance(future.exception(timeout=30.0), RuntimeError)
