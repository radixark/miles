from __future__ import annotations

from miles.utils.workers.worker_provider.kubernetes.views.pod_info import CellLabelKeys

INSTANCE_LABEL = "app.kubernetes.io/instance"

LWS_GROUP_INDEX_LABEL = "leaderworkerset.sigs.k8s.io/group-index"
LWS_WORKER_INDEX_LABEL = "leaderworkerset.sigs.k8s.io/worker-index"
LWS_SIZE_LABEL = "leaderworkerset.sigs.k8s.io/size"

POOL_LABEL = "miles.radixark.io/pool"
META_ANNOTATION_PREFIX = "miles.radixark.io/meta-"
GPU_IDS_META = "gpu_ids"

DEFAULT_LABEL_KEYS = CellLabelKeys(
    pool_id=POOL_LABEL,
    cell_ordinal=LWS_GROUP_INDEX_LABEL,
    pod_index=LWS_WORKER_INDEX_LABEL,
    cell_size=LWS_SIZE_LABEL,
    meta_annotation_prefix=META_ANNOTATION_PREFIX,
    gpu_ids_meta=GPU_IDS_META,
)
