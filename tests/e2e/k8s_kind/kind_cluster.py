# doc-dev: docs/developer/reconcile-loop.md
from __future__ import annotations

import hashlib
import logging
import platform
import shutil
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from miles.utils.misc import exec_command

logger = logging.getLogger(__name__)

_READY_TIMEOUT = "180s"

_KIND_VERSION = "v0.32.0"
_KIND_CACHE_DIR = Path.home() / ".cache" / "miles" / "bin"
_KIND_SHA256 = {
    "linux-amd64": "50030de23cf40a18505f20426f6a8506bedf13c6e509244bd1fa9463721b0f54",
    "linux-arm64": "b92cd615e97585de8ddade28ed5cd7feb4248d717c233eea5b03c37298900f5d",
    "darwin-amd64": "295ac6d0d634c9819c9907df45e3017d1f13166bd13c3404c45e79f7faa47498",
    "darwin-arm64": "dca67911095a110c2b5c36e26df6cac860c602033e456c0db47be498cdef1ebb",
}


@dataclass(frozen=True)
class KindCluster:
    name: str
    kubeconfig: Path


def create_cluster(*, run_id: str, kubeconfig: Path) -> KindCluster:
    kind = _resolve_kind_binary()
    exec_command(f"{kind} create cluster --name {run_id} --kubeconfig {kubeconfig} --wait {_READY_TIMEOUT}")
    return KindCluster(name=run_id, kubeconfig=kubeconfig)


def delete_cluster(cluster: KindCluster) -> None:
    kind = _resolve_kind_binary()
    exec_command(f"{kind} delete cluster --name {cluster.name} --kubeconfig {cluster.kubeconfig}")


def _resolve_kind_binary() -> str:
    installed = shutil.which("kind")
    if installed is not None:
        return installed

    cached = _KIND_CACHE_DIR / f"kind-{_KIND_VERSION}"
    if not cached.exists():
        _download_kind(cached)
    return str(cached)


def _download_kind(destination: Path) -> None:
    platform_key = f"{_host_os()}-{_host_arch()}"
    expected_digest = _KIND_SHA256[platform_key]
    url = f"https://kind.sigs.k8s.io/dl/{_KIND_VERSION}/kind-{platform_key}"
    logger.warning(f"kind is not installed, downloading it once {url=} {destination=}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(".partial")
    urllib.request.urlretrieve(url, partial)
    actual_digest = hashlib.sha256(partial.read_bytes()).hexdigest()
    assert (
        actual_digest == expected_digest
    ), f"downloaded kind does not match its pinned digest {url=} {expected_digest=} {actual_digest=}"
    partial.chmod(0o755)
    partial.replace(destination)


def _host_os() -> str:
    assert sys.platform in ("linux", "darwin"), f"kind is only wired up for linux and darwin, got {sys.platform=}"
    return sys.platform


def _host_arch() -> str:
    machine = platform.machine()
    arch = {"x86_64": "amd64", "amd64": "amd64", "arm64": "arm64", "aarch64": "arm64"}.get(machine)
    assert arch is not None, f"unsupported machine for a kind download {machine=}"
    return arch
