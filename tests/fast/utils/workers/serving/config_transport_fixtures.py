from dataclasses import dataclass
from typing import ClassVar

from miles.utils.args.runtime_base import BaseLeafConfig


@dataclass(frozen=True)
class _ConfigTransportWorkerSpec:
    args: BaseLeafConfig


class _ConfigTransportSpec:
    worker_type: ClassVar[str] = "config-transport-probe"
    config_class: ClassVar[type[BaseLeafConfig]] = BaseLeafConfig

    @classmethod
    def create(cls, config: BaseLeafConfig) -> _ConfigTransportWorkerSpec:
        return _ConfigTransportWorkerSpec(args=config)
