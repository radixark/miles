from __future__ import annotations

from pydantic import ConfigDict

from miles.utils.pydantic_utils import StrictBaseModel
from miles.utils.workers.worker_handle import BaseWorkerHandle
from miles.utils.workers.worker_spec import NamedHostAndPorts


class WorkerInfo(StrictBaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    name: str
    generation: int
    self_addrs: NamedHostAndPorts
    gpu_ids: list[int]
    handle: BaseWorkerHandle
