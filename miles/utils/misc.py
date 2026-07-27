import asyncio
import importlib
import inspect
import logging
from collections.abc import Sequence
from contextlib import contextmanager
from typing import Any

import ray

from miles.utils.http_utils import is_port_available

logger = logging.getLogger(__name__)


# Mainly used for test purpose where `load_function` needs to load many in-flight generated functions
class FunctionRegistry:
    def __init__(self):
        self._registry: dict[str, object] = {}

    @contextmanager
    def temporary(self, name: str, fn: object):
        self._register(name, fn)
        try:
            yield
        finally:
            self._unregister(name)

    def get(self, name: str) -> object | None:
        return self._registry.get(name)

    def _register(self, name: str, fn: object) -> None:
        assert name not in self._registry
        self._registry[name] = fn

    def _unregister(self, name: str) -> None:
        assert name in self._registry
        self._registry.pop(name)


function_registry = FunctionRegistry()


# TODO may rename to `load_object` since it can be used to load things like tool_specs
def load_function(path, *, sync_required=False):
    """
    Load a function from registry or module.
    :param path: The path to the function, e.g. "module.submodule.function".
    :param sync_required: Reject coroutine functions, for callers that run the
        loaded function synchronously on an event loop.
    :return: The function object.
    """
    if not path:
        return None

    fn = function_registry.get(path)
    if fn is None:
        module_path, _, attr = path.rpartition(".")
        module = importlib.import_module(module_path)
        fn = getattr(module, attr)
    if sync_required:
        if not callable(fn):
            raise ValueError(f"load_function({path!r}) did not resolve to a callable")
        if inspect.iscoroutinefunction(fn):
            raise ValueError(f"load_function({path!r}) resolved to an async function; a synchronous one is required")
    return fn


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
