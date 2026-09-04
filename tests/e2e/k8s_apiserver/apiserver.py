# doc-dev: docs/developer/reconcile-loop.md
from __future__ import annotations

import json
import logging
import socket
from dataclasses import dataclass
from pathlib import Path

from miles.utils.misc import exec_command

logger = logging.getLogger(__name__)

_ETCD_IMAGE = "registry.k8s.io/etcd:3.5.15-0"
_APISERVER_IMAGE = "registry.k8s.io/kube-apiserver:v1.31.4"
_SECURE_PORT = 6443
_KEY_MOUNT_DIR = "/etc/miles-k8s"
_SERVICE_ACCOUNT_KEY = "service-account.key"
_TOKEN_FILE = "token.csv"
_TOKEN = "miles-k8s-test-token"


@dataclass(frozen=True)
class ApiserverEnvironment:
    host_port: int
    token: str
    network_name: str
    etcd_name: str
    apiserver_name: str

    @property
    def endpoint(self) -> str:
        return f"https://127.0.0.1:{self.host_port}"


def start_apiserver(*, run_id: str, work_dir: Path, watch_cache: bool = True) -> ApiserverEnvironment:
    network_name = f"{run_id}-net"
    etcd_name = f"{run_id}-etcd"
    apiserver_name = f"{run_id}-apiserver"

    try:
        exec_command(f"openssl genrsa -out {work_dir / _SERVICE_ACCOUNT_KEY} 2048")
        (work_dir / _TOKEN_FILE).write_text(f"{_TOKEN},miles-test,miles-test-uid,system:masters\n")
        exec_command(f"docker network create {network_name}")
        exec_command(
            f"docker run --detach --name {etcd_name} --network {network_name} {_ETCD_IMAGE} "
            f"etcd --data-dir /tmp/etcd "
            f"--advertise-client-urls http://0.0.0.0:2379 --listen-client-urls http://0.0.0.0:2379"
        )
        exec_command(f"docker pull --quiet {_APISERVER_IMAGE}")
        host_port = _free_host_port()
        exec_command(
            f"docker run --detach --name {apiserver_name} --network {network_name} "
            f"--publish 127.0.0.1:{host_port}:{_SECURE_PORT} --volume {work_dir}:{_KEY_MOUNT_DIR}:ro "
            f"{_APISERVER_IMAGE} kube-apiserver {_apiserver_flags(etcd_name=etcd_name, watch_cache=watch_cache)}"
        )

        return ApiserverEnvironment(
            host_port=host_port,
            token=_TOKEN,
            network_name=network_name,
            etcd_name=etcd_name,
            apiserver_name=apiserver_name,
        )
    except BaseException:
        _remove_environment_idempotently(apiserver_name=apiserver_name, etcd_name=etcd_name, network_name=network_name)
        raise


def stop_apiserver(environment: ApiserverEnvironment) -> None:
    _remove_environment_idempotently(
        apiserver_name=environment.apiserver_name,
        etcd_name=environment.etcd_name,
        network_name=environment.network_name,
    )


def _remove_environment_idempotently(*, apiserver_name: str, etcd_name: str, network_name: str) -> None:
    for command in (
        f"docker rm --force --volumes {apiserver_name} {etcd_name}",
        f"docker network rm {network_name}",
    ):
        try:
            exec_command(command)
        except Exception:
            logger.error(f"apiserver environment cleanup command failed {command=}", exc_info=True)


def log_apiserver_diagnostics(environment: ApiserverEnvironment) -> None:
    for container in (environment.apiserver_name, environment.etcd_name):
        logs = exec_command(f"docker logs --tail 50 {container} 2>&1", capture_output=True)
        logger.error(f"apiserver environment diagnostics {container=}\n{logs}")


def apiserver_started_at(environment: ApiserverEnvironment) -> str:
    started_at = exec_command(
        f"docker inspect --format {{{{.State.StartedAt}}}} {environment.apiserver_name}", capture_output=True
    )
    assert started_at is not None, f"docker inspect returned nothing for {environment.apiserver_name=}"
    return started_at.strip()


def restart_apiserver(environment: ApiserverEnvironment) -> None:
    exec_command(f"docker restart {environment.apiserver_name}")


def compact_etcd_to_head(environment: ApiserverEnvironment) -> None:
    status = exec_command(
        f"docker exec {environment.etcd_name} etcdctl endpoint status --write-out=json", capture_output=True
    )
    assert status is not None, f"etcdctl returned nothing for {environment.etcd_name=}"
    revision = json.loads(status)[0]["Status"]["header"]["revision"]
    exec_command(f"docker exec {environment.etcd_name} etcdctl compact {revision} --physical")


def _apiserver_flags(*, etcd_name: str, watch_cache: bool) -> str:
    service_account_key = f"{_KEY_MOUNT_DIR}/{_SERVICE_ACCOUNT_KEY}"
    flags = [
        f"--etcd-servers=http://{etcd_name}:2379",
        f"--secure-port={_SECURE_PORT}",
        "--bind-address=0.0.0.0",
        "--advertise-address=127.0.0.1",
        "--cert-dir=/tmp/certs",
        "--authorization-mode=AlwaysAllow",
        f"--token-auth-file={_KEY_MOUNT_DIR}/{_TOKEN_FILE}",
        "--disable-admission-plugins=ServiceAccount",
        "--service-cluster-ip-range=10.0.0.0/24",
        "--service-account-issuer=https://kubernetes.default.svc",
        f"--service-account-key-file={service_account_key}",
        f"--service-account-signing-key-file={service_account_key}",
    ]
    if not watch_cache:
        flags.append("--watch-cache=false")
    return " ".join(flags)


def _free_host_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])
