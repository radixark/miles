from __future__ import annotations

import json
import shlex

from miles.utils.external_utils.command_utils.helm_backend import naming

COMPONENT = "mooncake-master"
BACKEND_FLAG = "--object-store-backend"
BACKEND_NAME = "mooncake"
INIT_KWARGS_FLAG = "--mooncake-store-init-kwargs"
MASTER_ADDRESS_KEY = "master_server_address"


def master_service_host(release: str, namespace: str) -> str:
    return f"{naming.component_name(release, COMPONENT)}.{namespace}.svc.cluster.local"


def uses_mooncake(train_argv: list[str]) -> bool:
    return any(
        argument == BACKEND_FLAG and index + 1 < len(train_argv) and train_argv[index + 1] == BACKEND_NAME
        for index, argument in enumerate(train_argv)
    )


def master_port_of(train_argv: list[str], default_port: int) -> int:
    address = _init_kwargs(train_argv).get(MASTER_ADDRESS_KEY)
    if not isinstance(address, str) or ":" not in address:
        return default_port
    return int(address.rsplit(":", 1)[1])


def with_cluster_master(train_argv: list[str], host: str) -> list[str]:
    if not uses_mooncake(train_argv):
        return train_argv

    kwargs = _init_kwargs(train_argv)
    assert kwargs, f"{INIT_KWARGS_FLAG} is missing, so the mooncake master address cannot be rewritten"
    port = master_port_of(train_argv, default_port=0)
    assert port, f"{MASTER_ADDRESS_KEY} carries no port, so the in-cluster address cannot be built"
    kwargs[MASTER_ADDRESS_KEY] = f"{host}:{port}"

    rewritten = list(train_argv)
    rewritten[rewritten.index(INIT_KWARGS_FLAG) + 1] = json.dumps(kwargs)
    return rewritten


def _init_kwargs(train_argv: list[str]) -> dict:
    if INIT_KWARGS_FLAG not in train_argv:
        return {}
    raw = train_argv[train_argv.index(INIT_KWARGS_FLAG) + 1]
    if raw.startswith("'") or raw.startswith('"'):
        raw = shlex.split(raw)[0]
    return json.loads(raw)
