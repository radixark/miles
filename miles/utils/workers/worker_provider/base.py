import abc

from miles.utils.workers.worker_spec import HostAndPort, NamedHostAndPorts


class BaseWorkerProvider(abc.ABC):
    @abc.abstractmethod
    async def get_addr(self, worker_name: str) -> HostAndPort: ...

    @abc.abstractmethod
    async def get_addrs(self, worker_name: str) -> NamedHostAndPorts: ...
