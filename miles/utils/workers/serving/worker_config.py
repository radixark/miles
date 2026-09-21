from typing import Any

from miles.utils.pydantic_utils import FrozenStrictBaseModel
from miles.utils.workers.connection_config import StaticConnConfig


class ServeWorkerConfig(FrozenStrictBaseModel):
    worker_type: str
    args: dict[str, Any]
    static_connections: StaticConnConfig
