from subprocess import TimeoutExpired
from typing import Any


class _TimedOutCommand:
    def __init__(self, *, error: TimeoutExpired) -> None:
        self.error = error

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        raise self.error
