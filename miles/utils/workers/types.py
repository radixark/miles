from enum import Enum


class ClusterBackend(Enum):
    RAY = "ray"
    KUBERNETES = "kubernetes"
