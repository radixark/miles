from __future__ import annotations

from miles.utils.pydantic_utils import FrozenStrictBaseModel
from miles.utils.workers.worker_provider.kubernetes.core.pod_view import CellLabelKeys
from miles.utils.workers.worker_spec import BaseWorkerSpec


class KubernetesRunInfo(FrozenStrictBaseModel):
    namespace: str
    label_selector: str
    label_keys: CellLabelKeys
    specs: dict[str, BaseWorkerSpec]
