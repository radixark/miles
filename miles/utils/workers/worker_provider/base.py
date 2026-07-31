import abc

from miles.utils.workers.worker_spec import HostAndPort


class BaseWorkerProvider(abc.ABC):
    @abc.abstractmethod
    async def get_addr(self, worker_name: str) -> HostAndPort: ...
