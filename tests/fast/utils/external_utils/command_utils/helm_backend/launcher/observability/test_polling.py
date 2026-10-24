import threading

from miles.utils.external_utils.command_utils.helm_backend.launcher.observability.polling import polling_in_background


class TestPollingInBackground:
    def test_context_exit_is_bounded_when_the_polling_step_is_blocked(self) -> None:
        """Context exit waits for an in-flight polling step before completing."""
        step_entered = threading.Event()
        release = threading.Event()
        exit_completed = threading.Event()

        def blocked_step() -> None:
            step_entered.set()
            release.wait()

        def own_context() -> None:
            with polling_in_background(blocked_step, description="poll"):
                assert step_entered.wait(timeout=1.0)
            exit_completed.set()

        owner = threading.Thread(target=own_context)
        owner.start()

        try:
            assert step_entered.wait(timeout=1.0)
            assert not exit_completed.is_set()
            release.set()
            assert exit_completed.wait(timeout=1.0)
        finally:
            release.set()
            owner.join(timeout=1.0)

        assert not owner.is_alive()
