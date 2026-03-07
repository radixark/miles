"""Factory functions for creating agents with Ray-based controller communication.

External callers (tracking_utils, megatron_utils, test helpers) use these
factories instead of constructing agents + RayControllerClient themselves.
All Ray wiring is encapsulated here in the platform layer.
"""
from __future__ import annotations

import os

from miles.utils.ft.agents.core.tracking_agent import FtTrackingAgent
from miles.utils.ft.agents.core.training_rank_agent import FtTrainingRankAgent
from miles.utils.ft.platform.ray_controller_client import RayControllerClient
from miles.utils.ft.utils.graceful_degrade import graceful_degrade


def create_tracking_agent(
    run_id: str | None = None,
    ft_id: str = "",
) -> FtTrackingAgent:
    client = RayControllerClient(ft_id=ft_id or os.environ.get("MILES_FT_ID", ""))
    return FtTrackingAgent(run_id=run_id, controller_client=client)


@graceful_degrade()
def create_training_rank_agent(
    rank: int,
    world_size: int,
    ft_id: str = "",
    enabled: bool = True,
) -> FtTrainingRankAgent | None:
    if not enabled:
        return None
    client = RayControllerClient(ft_id=ft_id or os.environ.get("MILES_FT_ID", ""))
    return FtTrainingRankAgent(rank=rank, world_size=world_size, controller_client=client)
