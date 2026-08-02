import abc

from miles.utils.workers.worker_spec import NamedHostAndPorts


class BaseWorkerProvider(abc.ABC):
    @abc.abstractmethod
    async def get_addrs(self, worker_name: str) -> NamedHostAndPorts: ...
