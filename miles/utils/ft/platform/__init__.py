"""Platform adapters for the FT subsystem.

Lightweight symbols are available at package level. Heavy symbols that
depend on Ray or Kubernetes are lazily imported on first access.

Usage::

    from miles.utils.ft.platform import FtControllerConfig, build_ft_controller
    from miles.utils.ft.platform import FtControllerActor  # lazy, requires ray
"""
from __future__ import annotations

from miles.utils.ft.platform.config import FtControllerConfig
from miles.utils.ft.platform.controller_factory import build_ft_controller
from miles.utils.ft.platform.embedded_agents import create_tracking_agent, create_training_rank_agent
from miles.utils.ft.platform.stubs import StubNodeManager, StubNotifier, StubTrainingJob

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "FtControllerActor": ("miles.utils.ft.platform.ray_wrappers.controller_actor", "FtControllerActor"),
    "FtNodeAgentActor": ("miles.utils.ft.platform.ray_wrappers.node_agent_actor", "FtNodeAgentActor"),
    "RayControllerClient": ("miles.utils.ft.platform.ray_wrappers.controller_client", "RayControllerClient"),
    "RayTrainingJob": ("miles.utils.ft.platform.ray_wrappers.training_job", "RayTrainingJob"),
    "K8sNodeManager": ("miles.utils.ft.platform.k8s_node_manager", "K8sNodeManager"),
}


def __getattr__(name: str) -> object:
    if name in _LAZY_IMPORTS:
        module_path, attr = _LAZY_IMPORTS[name]
        import importlib
        mod = importlib.import_module(module_path)
        return getattr(mod, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "FtControllerActor",
    "FtControllerConfig",
    "FtNodeAgentActor",
    "K8sNodeManager",
    "RayControllerClient",
    "RayTrainingJob",
    "StubNodeManager",
    "StubNotifier",
    "StubTrainingJob",
    "build_ft_controller",
    "create_tracking_agent",
    "create_training_rank_agent",
]
