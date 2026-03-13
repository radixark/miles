"""Factory for creating FtRolloutAgent with controller registration."""

from __future__ import annotations

import logging
from collections.abc import Callable

import ray

from miles.utils.ft.adapters.types import REGISTER_TIMEOUT_SECONDS, ft_controller_actor_name
from miles.utils.ft.agents.core.rollout.rollout_agent import FtRolloutAgent
from miles.utils.ft.utils.env import get_ft_id
from miles.utils.ft.utils.retry import retry_sync

logger = logging.getLogger(__name__)

_REGISTER_MAX_ATTEMPTS = 3
_REGISTER_RETRY_DELAY = 2.0


def build_rollout_agent(
    *,
    cell_ids: list[str],
    get_engines: Callable[[str], list[object]],
    health_checker: Callable[[object], object] | None = None,
    check_interval: float = 30.0,
    ft_id: str = "",
    cell_node_ids: dict[str, list[str]] | None = None,
) -> FtRolloutAgent:
    if health_checker is None:
        health_checker = lambda engine: engine.health_generate.remote()  # type: ignore[attr-defined]

    agent = FtRolloutAgent(
        cell_ids=cell_ids,
        get_engines=get_engines,
        health_checker=health_checker,
        check_interval=check_interval,
    )
    _register_with_controller(
        agent=agent,
        ft_id=ft_id or get_ft_id(),
        cell_node_ids=cell_node_ids,
    )
    return agent


def _register_with_controller(
    *,
    agent: FtRolloutAgent,
    ft_id: str,
    cell_node_ids: dict[str, list[str]] | None = None,
) -> None:
    controller = ray.get_actor(ft_controller_actor_name(ft_id))
    self_handle = ray.get_runtime_context().current_actor

    def _do_register() -> None:
        ray.get(
            controller.register_rollout.remote(
                rollout_manager_handle=self_handle,
                metrics_address=agent.address,
                cell_node_ids=cell_node_ids,
            ),
            timeout=REGISTER_TIMEOUT_SECONDS,
        )

    result = retry_sync(
        func=_do_register,
        description=f"register_rollout(ft_id={ft_id})",
        max_retries=_REGISTER_MAX_ATTEMPTS,
        backoff_base=_REGISTER_RETRY_DELAY,
        max_backoff=_REGISTER_RETRY_DELAY,
    )
    if not result.ok:
        raise RuntimeError(
            f"rollout agent registration failed after {_REGISTER_MAX_ATTEMPTS} attempts"
        )
    logger.info(
        "rollout_agent_registered ft_id=%s metrics_address=%s cell_node_ids=%s",
        ft_id,
        agent.address,
        {k: sorted(v) for k, v in cell_node_ids.items()} if cell_node_ids else "(none)",
    )
