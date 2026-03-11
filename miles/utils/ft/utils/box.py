from __future__ import annotations

from typing import Generic, TypeVar

T = TypeVar("T")


class Box(Generic[T]):
    __slots__ = ("value",)

    def __init__(self, value: T) -> None:
        self.value = value
