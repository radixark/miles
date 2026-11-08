"""Where a ``--cluster-backend monarch`` run finds the hosts its gpu workers run on."""

from __future__ import annotations

import ipaddress
import logging
import os

from monarch.actor import HostMesh
from monarch.job import job_load

from miles.utils.object_store_config import MOONCAKE_MASTER_ADDRESS_KEY
from miles.utils.workers.monarch.env_vars import MONARCH_JOB_STATE_ENV_VAR
from miles.utils.workers.worker_provider.static import parse_host_and_port

logger = logging.getLogger(__name__)

# The one host mesh a submitted Monarch job holds all its hosts in
WORKER_MESH_NAME = "workers"
_ATTACH_HINT = (
    "see whether the worker loops started, in the slurm_<job id>_*.err files next to the job state, whether their "
    "port (the submission's --monarch-port) is open between the nodes, and, on a dual-stack cluster, "
    "submit with --bind-to ::"
)


def check_mooncake_master_reaches_every_host(mooncake_store_init_kwargs: dict | None, *, num_hosts: int) -> None:
    """Workers on the other hosts would fail to reach a loopback master, each with a bare connection refused."""
    if num_hosts <= 1 or (address := (mooncake_store_init_kwargs or {}).get(MOONCAKE_MASTER_ADDRESS_KEY)) is None:
        return
    assert not _is_loopback(parse_host_and_port(address).host), (
        f"the run's workers sit on {num_hosts} hosts, and the other hosts cannot reach the mooncake master at "
        f"{address}; launch the run through its launcher, which points MOONCAKE_MASTER at the head node, or set "
        f"MOONCAKE_MASTER=<head-node-address>:<port>"
    )


def _is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        return False


def acquire_worker_hosts() -> HostMesh | None:
    """The hosts of the Monarch job the run runs in, or None outside one, where every worker runs on the driver's
    host."""
    if not (job_state := os.environ.get(MONARCH_JOB_STATE_ENV_VAR)):
        logger.info("The run is in no monarch job, so every worker runs on the driver's host")
        return None
    logger.info(f"Attaching to the hosts of the monarch job saved at {job_state}")
    state = job_load(job_state).state()
    hosts = getattr(state, WORKER_MESH_NAME)  # config-access-exempt: JobState names its meshes as attributes
    try:
        hosts.initialized.get()
    except Exception as e:
        raise RuntimeError(f"The run's hosts never answered; {_ATTACH_HINT}") from e
    return hosts
