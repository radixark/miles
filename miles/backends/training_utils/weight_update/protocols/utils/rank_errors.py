import contextlib
from collections.abc import Iterator

import torch.distributed as dist

from miles.utils.distributed_utils import get_gloo_group


@contextlib.contextmanager
def fail_on_every_rank(step: str) -> Iterator[None]:
    """Collective: an error inside raises on every rank, so none waits in a collective the failed rank never joins."""
    local_error: Exception | None = None
    try:
        yield
    except Exception as error:
        local_error = error
    messages: list[str | None] = [None] * dist.get_world_size()
    local_message = None if local_error is None else f"{type(local_error).__name__}: {local_error}"
    dist.all_gather_object(messages, local_message, group=get_gloo_group())
    failures = [(rank, message) for rank, message in enumerate(messages) if message is not None]
    if failures:
        rank, message = failures[0]
        raise RuntimeError(f"{step} failed on rank {rank}: {message}") from local_error
