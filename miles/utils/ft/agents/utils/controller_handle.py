from __future__ import annotations

import logging
from typing import Any

from miles.utils.ft.protocols.platform import ft_controller_actor_name

logger = logging.getLogger(__name__)


def get_controller_handle(ft_id: str) -> Any | None:
    """Look up the ft_controller Ray actor by *ft_id*. Returns None on failure."""
    import ray

    try:
        return ray.get_actor(ft_controller_actor_name(ft_id))
    except Exception:
        logger.warning("Failed to get ft_controller actor handle", exc_info=True)
        return None
