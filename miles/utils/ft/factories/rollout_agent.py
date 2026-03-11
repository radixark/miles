"""Factory for creating FtRolloutAgent with controller registration."""

from __future__ import annotations

import logging

import ray

from miles.utils.ft.adapters.types import REGISTER_TIMEOUT_SECONDS, ft_controller_actor_name
from miles.utils.ft.agents.core.rollout.rollout_agent import FtRolloutAgent
from miles.utils.ft.utils.env import get_ft_id
from miles.utils.ft.utils.graceful_degrade import graceful_degrade
from miles.utils.ft.utils.retry import retry_sync

logger = logging.getLogger(__name__)

_REGISTER_MAX_ATTEMPTS = 3
_REGISTER_RETRY_DELAY = 2.0


async def _default_engine_health_checker(engine: object) -> None:
    await engine.health_generate.remote()  # type: ignore[attr-defined]


def create_rollout_agent(
    rollout_manager: object,
    *,
    check_interval: float = 10.0,
    ft_id: str = "",
) -> FtRolloutAgent:
    agent = FtRolloutAgent(
        rollout_manager,
        health_checker=_default_engine_health_checker,
        check_interval=check_interval,
    )
    _register_with_controller(agent=agent, ft_id=ft_id or get_ft_id())
    return agent


@graceful_degrade(msg="Failed to register rollout agent with controller")
def _register_with_controller(*, agent: FtRolloutAgent, ft_id: str) -> None:
    controller = ray.get_actor(ft_controller_actor_name(ft_id))
    self_handle = ray.get_runtime_context().current_actor

    def _do_register() -> None:
        ray.get(
            controller.register_rollout.remote(
                reward_manager_handle=self_handle,
                metrics_address=agent.address,
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
    if result.ok:
        logger.info(
            "rollout_agent_registered ft_id=%s metrics_address=%s",
            ft_id,
            agent.address,
        )
