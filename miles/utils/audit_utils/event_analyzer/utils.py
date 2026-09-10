from collections.abc import Sequence
from typing import TypeVar

from miles.utils.audit_utils.event_logger.models import Event

_EventT = TypeVar("_EventT")


def filter_by_type(events: Sequence[Event], ty: type[_EventT]) -> list[_EventT]:
    return [event for event in events if isinstance(event, ty)]
