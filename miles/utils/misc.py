import asyncio
import importlib
import re
import subprocess
from contextlib import contextmanager

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from miles.utils.http_utils import is_port_available


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
def load_function(path):
    """
    Load a function from registry or module.
    :param path: The path to the function, e.g. "module.submodule.function".
    :return: The function object.
    """
    if path is None:
        return None

    registered = function_registry.get(path)
    if registered is not None:
        return registered

    module_path, _, attr = path.rpartition(".")
    module = importlib.import_module(module_path)
    return getattr(module, attr)


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


def exec_command(cmd: str, capture_output: bool = False) -> str | None:
    print(f"EXEC: {cmd}", flush=True)

    try:
        result = subprocess.run(
            ["bash", "-c", cmd],
            shell=False,
            check=True,
            capture_output=capture_output,
            **(dict(text=True) if capture_output else {}),
        )
    except subprocess.CalledProcessError as e:
        if capture_output:
            print(f"{e.stdout=} {e.stderr=}")
        raise

    if capture_output:
        print(f"Captured stdout={result.stdout} stderr={result.stderr}")
        return result.stdout


@ray.remote(num_cpus=0.001)
def _exec_command_on_node(cmd: str, capture_output: bool) -> str | None:
    return exec_command(f"unset CUDA_VISIBLE_DEVICES; {cmd}", capture_output=capture_output)


def exec_command_all_ray_node(
    cmd: str, capture_output: bool = False, num_nodes: int | None = None
) -> list[str | None]:
    """Execute a shell command on every alive Ray node in parallel.

    Supported placeholders in `cmd` (replaced per-node before execution):
        {{node_rank}}   - 0-based index of the node
        {{nnodes}}      - total number of alive nodes (or num_nodes if specified)
        {{master_addr}} - NodeManagerAddress of the first node
        {{node_ip}}     - NodeManagerAddress of the current node

    Args:
        num_nodes: If set, only use the first `num_nodes` nodes instead of all alive nodes.
    """
    ray.init(address="auto")
    try:
        current_ip = get_current_node_ip()
        nodes = sorted(
            [n for n in ray.nodes() if n.get("Alive")],
            key=lambda n: (n["NodeManagerAddress"] != current_ip, n["NodeManagerAddress"]),
        )
        assert len(nodes) > 0

        if num_nodes is not None:
            assert num_nodes <= len(nodes), f"Requested {num_nodes} nodes but only {len(nodes)} alive nodes available."
            nodes = nodes[:num_nodes]

        master_addr = nodes[0]["NodeManagerAddress"]
        nnodes = str(len(nodes))

        placeholder_pattern = re.compile(
            "|".join(map(re.escape, ["{{node_rank}}", "{{nnodes}}", "{{master_addr}}", "{{node_ip}}"]))
        )

        refs = []
        for rank, node in enumerate(nodes):
            substitutions = {
                "{{node_rank}}": str(rank),
                "{{nnodes}}": nnodes,
                "{{master_addr}}": master_addr,
                "{{node_ip}}": node["NodeManagerAddress"],
            }
            node_cmd = placeholder_pattern.sub(lambda m, s=substitutions: s[m.group(0)], cmd)
            refs.append(
                _exec_command_on_node.options(
                    scheduling_strategy=NodeAffinitySchedulingStrategy(
                        node_id=node["NodeID"],
                        soft=False,
                    ),
                ).remote(node_cmd, capture_output=capture_output)
            )
        return ray.get(refs)
    finally:
        ray.shutdown()


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
