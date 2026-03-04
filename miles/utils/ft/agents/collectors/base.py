from abc import ABC, abstractmethod

from miles.utils.ft.models import CollectorOutput


class BaseCollector(ABC):
    @abstractmethod
    async def collect(self) -> CollectorOutput:
        ...
