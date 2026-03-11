from __future__ import annotations


class MockEngine:
    def __init__(self, alive: bool = True) -> None:
        self.alive = alive

    async def health_check(self) -> None:
        if not self.alive:
            raise ConnectionError("engine dead")
