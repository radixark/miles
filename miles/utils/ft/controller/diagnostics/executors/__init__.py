from miles.utils.ft.adapters.types import ClusterExecutorProtocol
from miles.utils.ft.controller.diagnostics.executors.gpu import GpuClusterExecutor
from miles.utils.ft.controller.diagnostics.executors.neighbor_nccl import NeighborNcclClusterExecutor
from miles.utils.ft.controller.diagnostics.executors.per_node import PerNodeClusterExecutor
from miles.utils.ft.controller.diagnostics.executors.stack_trace import StackTraceClusterExecutor

__all__ = [
    "GpuClusterExecutor",
    "NeighborNcclClusterExecutor",
    "PerNodeClusterExecutor",
    "StackTraceClusterExecutor",
    "build_all_cluster_executors",
]


def build_all_cluster_executors() -> dict[str, ClusterExecutorProtocol]:
    return {
        "gpu": GpuClusterExecutor(),
        "nccl_simple": PerNodeClusterExecutor(diagnostic_type="nccl_simple"),
        "nccl_neighbor": NeighborNcclClusterExecutor(diagnostic_type="nccl_pairwise"),
    }
