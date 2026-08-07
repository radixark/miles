from __future__ import annotations

from miles.utils.workers.worker_provider.kubernetes.core.pod_view import CellLabelKeys

INSTANCE_LABEL = "app.kubernetes.io/instance"

DEFAULT_LABEL_KEYS = CellLabelKeys(
    pool_id="miles.radixark.io/pool",
    cell_index="leaderworkerset.sigs.k8s.io/group-index",
    pod_index="leaderworkerset.sigs.k8s.io/worker-index",
    cell_size="leaderworkerset.sigs.k8s.io/size",
    meta_annotation_prefix="miles.radixark.io/meta-",
    gpu_ids_meta="gpu_ids",
)
