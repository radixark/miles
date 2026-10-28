"""Where the synchronous Monarch calls of a test run, so they leave pytest's own context free of Monarch.

Once a context has made a Monarch call, Monarch prefixes every log message emitted in it with ``[actor=<root>] ``,
which would break the exact-message ``caplog`` checks of tests that run later in the same process. Async tests need
none of this: each runs in a task of its own, whose context changes stay with the task.
"""

import contextvars
from collections.abc import Callable
from typing import Any

_MONARCH_CONTEXT = contextvars.copy_context()


def in_monarch_context(fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
    return _MONARCH_CONTEXT.run(fn, *args, **kwargs)
