import asyncio
import logging
from collections.abc import Sequence
from typing import Any

import ray

from miles.utils.function_registry import load_function
from miles.utils.http_utils import is_port_available

logger = logging.getLogger(__name__)


async def call_agent_abort_hook(args) -> None:
    """Invoke the agent plugin's optional abort hook, if it defines one.

    When oversampling collects enough samples, the rollout aborts SGLang, but an
    external agent loop (driven by ``--custom-agent-function-path``) keeps running
    and keeps issuing fresh completion requests until it hits its own limit. The
    agent integration knows how to tell its backend to stop, so we look for a
    sibling ``abort`` callable in the same module as the configured agent function
    and call it. Backends that don't expose one are left to drain as before.
    """
    agent_function_path = getattr(args, "custom_agent_function_path", None)
    if not agent_function_path:
        return

    module_path, _, _ = agent_function_path.rpartition(".")
    if not module_path:
        return
    try:
        abort_hook = load_function(f"{module_path}.abort")
    except (AttributeError, ModuleNotFoundError):
        return  # plugin doesn't expose an abort hook; nothing to tear down

    try:
        await abort_hook(args)
    except Exception as e:
        logger.warning(f"Agent abort hook {module_path}.abort failed: {e}")


class SingletonMeta(type):
    """
    A metaclass for creating singleton classes.
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]

    @staticmethod
    def clear_all_instances():
        SingletonMeta._instances.clear()


def get_current_node_ip():
    address = ray._private.services.get_node_ip_address()
    # strip ipv6 address
    address = address.strip("[]")
    return address


def get_free_port(start_port=10000, consecutive=1):
    # find the port where port, port + 1, port + 2, ... port + consecutive - 1 are all available
    port = start_port
    while not all(is_port_available(port + i) for i in range(consecutive)):
        port += 1
    return port


class NodeProbeMixin:
    @staticmethod
    def _get_node_ip() -> str:
        return get_current_node_ip()

    @staticmethod
    def _get_free_port_block(*, start_port: int, count: int) -> int:
        return get_free_port(start_port=start_port, consecutive=count)


def should_run_periodic_action(
    rollout_id: int,
    interval: int | None,
    num_rollout_per_epoch: int | None = None,
    num_rollout: int | None = None,
) -> bool:
    """
    Return True when a periodic action (eval/save/checkpoint) should run.

    Args:
        rollout_id: The current rollout index (0-based).
        interval: Desired cadence; disables checks when None.
        num_rollout_per_epoch: Optional epoch boundary to treat as a trigger.
    """
    if interval is None:
        return False

    if num_rollout is not None and rollout_id == num_rollout - 1:
        return True

    step = rollout_id + 1
    return (step % interval == 0) or (num_rollout_per_epoch is not None and step % num_rollout_per_epoch == 0)


async def as_completed_async(tasks):
    for coro in asyncio.as_completed(tasks):
        yield await coro


def filter_keys(d: dict[str, Any], interest_keys: Sequence[str]) -> dict[str, Any]:
    try:
        return {k: d[k] for k in interest_keys}
    except Exception:
        logger.error(f"filter_keys d.keys={list(d)} {interest_keys=}", exc_info=True)
        raise
