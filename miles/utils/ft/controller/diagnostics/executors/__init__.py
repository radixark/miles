from miles.utils.ft.controller.diagnostics.executors.gpu import GpuClusterExecutor
from miles.utils.ft.controller.diagnostics.executors.pairwise import PairwiseClusterExecutor
from miles.utils.ft.controller.diagnostics.executors.per_node import PerNodeClusterExecutor
from miles.utils.ft.controller.diagnostics.executors.stack_trace import StackTraceClusterExecutor
from miles.utils.ft.protocols.agents import ClusterExecutorProtocol

__all__ = [
    "GpuClusterExecutor",
    "PairwiseClusterExecutor",
    "PerNodeClusterExecutor",
    "StackTraceClusterExecutor",
    "build_all_cluster_executors",
]


def build_all_cluster_executors() -> dict[str, ClusterExecutorProtocol]:
    return {
        "gpu": GpuClusterExecutor(),
        "nccl_simple": PerNodeClusterExecutor(diagnostic_type="nccl_simple"),
        "nccl_pairwise": PairwiseClusterExecutor(diagnostic_type="nccl_pairwise"),
    }
