"""Factory functions for creating agents with Ray-based controller communication.

External callers (tracking_utils, megatron_utils, test helpers) use these
factories instead of constructing agents + RayControllerClient themselves.
All Ray wiring is encapsulated here in the factories layer.
"""

from __future__ import annotations

import logging
from typing import Any

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from miles.utils.ft.adapters.impl.ray.controller_client import RayControllerClient
from miles.utils.ft.adapters.impl.ray.node_agent_actor import FtNodeAgentActor
from miles.utils.ft.adapters.types import ft_node_agent_actor_name
from miles.utils.ft.agents.core.tracking_agent import FtTrackingAgent
from miles.utils.ft.agents.core.training_rank_agent import FtTrainingRankAgent
from miles.utils.ft.factories.node_agent import build_node_agent
from miles.utils.ft.utils.env import get_ft_id
from miles.utils.ft.utils.graceful_degrade import graceful_degrade

logger = logging.getLogger(__name__)


def build_tracking_agent(
    run_id: str | None = None,
    ft_id: str = "",
) -> FtTrackingAgent:
    client = RayControllerClient(ft_id=ft_id or get_ft_id())
    return FtTrackingAgent(run_id=run_id, controller_client=client)


@graceful_degrade()
def build_training_rank_agent(
    rank: int,
    world_size: int,
    ft_id: str = "",
    enabled: bool = True,
    node_id: str | None = None,
) -> FtTrainingRankAgent | None:
    if not enabled:
        return None
    client = RayControllerClient(ft_id=ft_id or get_ft_id())
    return FtTrainingRankAgent(rank=rank, world_size=world_size, controller_client=client, node_id=node_id)


def _ensure_ray_actor_on_node(
    actor_cls: type,
    name: str,
    node_id: str,
    *,
    actor_kwargs: dict[str, Any] | None = None,
    start_method: str = "start",
) -> None:
    """Idempotently create a Ray actor pinned to *node_id*.

    Safe to call from multiple ranks on the same node concurrently:
    the second caller sees a ``ValueError`` from ``actor_cls.options(...).remote()``
    and treats it as a benign race.
    """
    try:
        ray.get_actor(name)
        logger.debug("Actor %s already exists, skipping creation", name)
        return
    except ValueError:
        pass

    try:
        handle = actor_cls.options(
            name=name,
            scheduling_strategy=NodeAffinitySchedulingStrategy(
                node_id=node_id,
                soft=False,
            ),
        ).remote(**(actor_kwargs or {}))
        getattr(handle, start_method).remote()
        logger.info("Created actor %s on node %s", name, node_id)
    except ValueError:
        logger.info(
            "Actor %s was created by another rank concurrently, skipping",
            name,
        )
    except Exception:
        logger.warning(
            "Failed to create actor %s on node %s",
            name,
            node_id,
            exc_info=True,
        )


@graceful_degrade(msg="Failed to ensure node agent on current node")
def ensure_node_agent(ft_id: str = "") -> None:
    resolved_ft_id = ft_id or get_ft_id()
    node_id = ray.get_runtime_context().get_node_id()
    name = ft_node_agent_actor_name(resolved_ft_id, node_id)

    _ensure_ray_actor_on_node(
        actor_cls=FtNodeAgentActor,
        name=name,
        node_id=node_id,
        actor_kwargs={
            "builder": build_node_agent,
            "node_id": node_id,
            "ft_id": resolved_ft_id,
        },
    )
