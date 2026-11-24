"""Submit, stream, and clean up a 0-GPU Miles CI Driver Job in Kubernetes."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

CI_JOB_ID_LABEL = "miles.radixark.io/ci-job-id"
CI_ROLE_LABEL = "miles.radixark.io/role"
CI_DRIVER_ROLE = "ci-driver"

_DIGEST_PATTERN = re.compile(r"@sha256:[0-9a-f]{64}$")
_SHA256_VALUE_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_K8S_NAME_SANITIZER = re.compile(r"[^a-z0-9-]+")
_BRIDGE_PR_DIRECTIVE = re.compile(r"^ci-megatron-bridge-pr:\s+(\S+)", re.MULTILINE)
_NATIVE_SOURCE_EXTENSIONS = frozenset({".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp"})

_DEFAULT_MODELS_ROOT = "/storage/models"
_DEFAULT_JIT_CACHE_ROOT = "/data/miles_ci/jit"
_DEFAULT_SERVICE_ACCOUNT = "miles-workbench"
_DEFAULT_ACTIVE_DEADLINE_SECONDS = 7200
_DEFAULT_ADMISSION_TIMEOUT_SECONDS = 3600
_DEFAULT_EXECUTION_TIMEOUT_SECONDS = 5400
_DEFAULT_POD_START_POLL_SECONDS = 5.0


@dataclass(frozen=True)
class DriverJobConfig:
    namespace: str
    gpfs_root: str
    image: str
    ci_job_id: str
    command: tuple[str, ...]
    service_account: str = _DEFAULT_SERVICE_ACCOUNT
    models_root: str = _DEFAULT_MODELS_ROOT
    jit_cache_root: str = _DEFAULT_JIT_CACHE_ROOT
    rdma_resource: str | None = None
    colocate_shm_host_path: str | None = None
    active_deadline_seconds: int = _DEFAULT_ACTIVE_DEADLINE_SECONDS
    admission_timeout_seconds: int = _DEFAULT_ADMISSION_TIMEOUT_SECONDS
    execution_timeout_seconds: int = _DEFAULT_EXECUTION_TIMEOUT_SECONDS
    extra_env: dict[str, str] = field(default_factory=dict)
    tolerations: tuple[dict[str, Any], ...] = ()
    node_selector: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class StagedSources:
    staging_root: Path
    miles_dir: Path
    megatron_dir: Path | None = None
    sglang_dir: Path | None = None


def require_config_value(value: str | None, *, env_var: str, flag: str) -> str:
    resolved = (value if value is not None else os.environ.get(env_var, "")).strip()
    if not resolved:
        raise ValueError(f"Required configuration {flag} (or env {env_var}) is missing or empty")
    return resolved


def sanitize_ci_job_id(raw: str) -> str:
    cleaned = _K8S_NAME_SANITIZER.sub("-", raw.strip().lower()).strip("-")
    if not cleaned:
        raise ValueError(f"Cannot derive a valid Kubernetes ci-job-id from {raw!r}")
    return cleaned[:48].rstrip("-")


def driver_job_name(ci_job_id: str) -> str:
    return f"miles-ci-driver-{ci_job_id}"


def check_k8s_dependency_support(
    *,
    megatron_bridge_ref: str = "",
    pr_body: str = "",
    miles_changed_files: Sequence[str] = (),
    sglang_changed_files: Sequence[str] = (),
) -> None:
    """Fail fast on dependency overrides that require container image pip reinstalls."""
    reasons: list[str] = []

    bridge_ref = megatron_bridge_ref.strip()
    if not bridge_ref and pr_body:
        match = _BRIDGE_PR_DIRECTIVE.search(pr_body)
        if match:
            bridge_ref = match.group(1).strip()
    if bridge_ref:
        reasons.append(f"ci-megatron-bridge-pr ({bridge_ref}) requires pip reinstall of megatron-bridge")

    for path in miles_changed_files:
        normalized = path.strip().lstrip("./")
        if normalized == "examples/multi_lora/requirements.txt":
            reasons.append("examples/multi_lora/requirements.txt modification requires pip install inside the image")

    for path in sglang_changed_files:
        normalized = path.strip().lstrip("./")
        if normalized == "radix_transfer_plugins/pyproject.toml" or normalized in (
            "radix_transfer_plugins/setup.py",
            "radix_transfer_plugins/setup.cfg",
        ):
            reasons.append(f"{normalized} modification requires pip reinstall of radix_transfer_plugins")
        elif normalized.startswith("radix_transfer_plugins/") and Path(normalized).suffix in _NATIVE_SOURCE_EXTENSIONS:
            reasons.append(f"{normalized} modifies native extension sources requiring rebuild")

    if reasons:
        bullet_list = "\n".join(f"  - {reason}" for reason in reasons)
        raise RuntimeError(
            "Kubernetes CI does not support runtime pip overrides for this PR:\n"
            f"{bullet_list}\n"
            "Please route this PR to the Ray CI pool (add the 'run-on-ray' label) or rebuild the PR CI image."
        )


def resolve_image_digest(image: str) -> str:
    """Pin a mutable repository:tag reference to an immutable @sha256:... digest."""
    image = image.strip()
    if not image:
        raise ValueError("Container image reference must be non-empty")
    if _DIGEST_PATTERN.search(image):
        return image

    digest = _query_registry_digest(image)
    if not _SHA256_VALUE_PATTERN.fullmatch(digest):
        raise RuntimeError(f"Registry digest lookup for {image!r} returned invalid digest {digest!r}")
    return f"{image}@{digest}"


def _query_registry_digest(image: str) -> str:
    if shutil.which("crane"):
        result = subprocess.run(["crane", "digest", image], check=True, capture_output=True, text=True)
        return result.stdout.strip()
    if shutil.which("skopeo"):
        result = subprocess.run(
            ["skopeo", "inspect", "--format", "{{.Digest}}", f"docker://{image}"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    raise RuntimeError(
        f"Cannot pin mutable image {image!r} to an immutable digest: neither 'crane' nor 'skopeo' is installed "
        "(pass an already digest-pinned image@sha256:... or install crane/skopeo)"
    )


def split_image_for_helm(pinned_image: str) -> tuple[str, str]:
    """Split a digest-pinned image into (repository, tag) compatible with miles-common `repository:tag`."""
    if not _DIGEST_PATTERN.search(pinned_image):
        raise ValueError(f"Expected digest-pinned image ending in @sha256:..., got {pinned_image!r}")
    name_part, digest = pinned_image.rsplit("@", 1)
    last_slash = name_part.rfind("/")
    last_colon = name_part.rfind(":")
    if last_colon > last_slash:
        repository = name_part[:last_colon]
        tag_prefix = name_part[last_colon + 1 :]
    else:
        repository = name_part
        tag_prefix = "pinned"
    return repository, f"{tag_prefix}@{digest}"


def stage_sources(
    *,
    gpfs_root: str | Path,
    ci_job_id: str,
    miles_source_dir: str | Path,
    megatron_source_dir: str | Path | None = None,
    sglang_source_dir: str | Path | None = None,
) -> StagedSources:
    """Copy checked-out source trees into the shared GPFS staging directory for this CI job."""
    staging_root = Path(gpfs_root) / "staging" / ci_job_id
    staging_root.mkdir(parents=True, exist_ok=True)

    miles_dst = staging_root / "miles"
    _copy_source_tree(Path(miles_source_dir), miles_dst)

    megatron_dst: Path | None = None
    if megatron_source_dir is not None:
        megatron_dst = staging_root / "Megatron-LM"
        _copy_source_tree(Path(megatron_source_dir), megatron_dst)

    sglang_dst: Path | None = None
    if sglang_source_dir is not None:
        sglang_dst = staging_root / "sglang"
        _copy_source_tree(Path(sglang_source_dir), sglang_dst)

    return StagedSources(
        staging_root=staging_root,
        miles_dir=miles_dst,
        megatron_dir=megatron_dst,
        sglang_dir=sglang_dst,
    )


def _copy_source_tree(src: Path, dst: Path) -> None:
    if not src.is_dir():
        raise FileNotFoundError(f"Source directory does not exist: {src}")
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, symlinks=True, ignore=shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache"))


def build_infra_values(*, config: DriverJobConfig, staged: StagedSources, pinned_image: str) -> dict[str, Any]:
    """Build per-job Helm infra values shared by all miles-run and command-job pods."""
    repository, tag = split_image_for_helm(pinned_image)
    gpfs_root = Path(config.gpfs_root)

    gpfs_mounts: list[dict[str, Any]] = [
        {"mountPath": str(gpfs_root)},
        {
            "mountPath": "/root/miles",
            "subPath": str(staged.miles_dir.relative_to(gpfs_root)),
            "readOnly": True,
        },
    ]
    if staged.megatron_dir is not None:
        gpfs_mounts.append(
            {
                "mountPath": "/root/Megatron-LM",
                "subPath": str(staged.megatron_dir.relative_to(gpfs_root)),
                "readOnly": True,
            }
        )
    if staged.sglang_dir is not None:
        gpfs_mounts.append(
            {
                "mountPath": "/sgl-workspace/sglang",
                "subPath": str(staged.sglang_dir.relative_to(gpfs_root)),
                "readOnly": True,
            }
        )

    volumes: list[dict[str, Any]] = [
        {
            "name": "gpfs-ci-root",
            "hostPath": {"path": str(gpfs_root), "type": "DirectoryOrCreate"},
            "mounts": gpfs_mounts,
        },
        {
            "name": "models",
            "hostPath": {"path": config.models_root, "type": "Directory"},
            "mounts": [{"mountPath": "/root/models", "readOnly": True}],
        },
        {
            "name": "jit-cache",
            "hostPath": {"path": config.jit_cache_root, "type": "DirectoryOrCreate"},
            "mounts": [{"mountPath": config.jit_cache_root}],
        },
    ]

    scheduling: dict[str, Any] = {}
    if config.node_selector:
        scheduling["nodeSelector"] = dict(config.node_selector)
    if config.tolerations:
        scheduling["tolerations"] = list(config.tolerations)

    dev_shm: dict[str, Any] = {
        "mountPath": "/dev/shm",
        "emptyDir": {"medium": "Memory", "sizeLimit": "32Gi"},
    }
    if config.colocate_shm_host_path:
        dev_shm["colocateHostPath"] = {
            "path": config.colocate_shm_host_path,
            "type": "DirectoryOrCreate",
        }

    infra: dict[str, Any] = {
        "image": {
            "repository": repository,
            "tag": tag,
            "pullPolicy": "IfNotPresent",
        },
        "volumes": volumes,
        "paths": {"runsRoot": str(gpfs_root / "runs" / config.ci_job_id)},
        "devShm": dev_shm,
        "scheduling": scheduling,
        "env": {
            "TRITON_CACHE_DIR": f"{config.jit_cache_root}/triton",
            "TORCHINDUCTOR_CACHE_DIR": f"{config.jit_cache_root}/inductor",
        },
    }
    if config.rdma_resource:
        infra["gpuResources"] = {config.rdma_resource: 1}
    return {"infra": infra}


def build_driver_job_manifest(
    *,
    config: DriverJobConfig,
    staged: StagedSources,
    pinned_image: str,
    infra_values_path: Path,
) -> dict[str, Any]:
    """Build the 0-GPU Driver batch/v1 Job manifest executed under miles-workbench."""
    job_name = driver_job_name(config.ci_job_id)
    gpfs_root = Path(config.gpfs_root)
    gate_record_dir = gpfs_root / "runs" / config.ci_job_id / "ci-gate"

    env_map: dict[str, str] = {
        "MILES_CI_JOB_ID": config.ci_job_id,
        "MILES_SCRIPT_CLUSTER_BACKEND": "kubernetes",
        "MILES_SCRIPT_NAMESPACE": config.namespace,
        "MILES_SCRIPT_HELM_VALUES": str(infra_values_path),
        "MILES_CI_GATE_RECORD_DIR": str(gate_record_dir),
        "MILES_K8S_ADMISSION_TIMEOUT_SECONDS": str(config.admission_timeout_seconds),
        "MILES_K8S_EXECUTION_TIMEOUT_SECONDS": str(config.execution_timeout_seconds),
        "GITHUB_WORKSPACE": "/root/miles",
        "MEGATRON_SOURCE_ROOT": "/root/Megatron-LM",
        "SGLANG_SOURCE_ROOT": "/sgl-workspace/sglang/python",
        "PYTHONPATH": "/root/miles:/root/Megatron-LM:/sgl-workspace/sglang/python",
        **config.extra_env,
    }
    # Never trigger namespace-wide CI_LABEL sweeps on shared K8s CI namespaces.
    env_map.pop("MILES_SCRIPT_CI_RUN", None)

    volume_mounts: list[dict[str, Any]] = [
        {"name": "gpfs-ci-root", "mountPath": str(gpfs_root)},
        {
            "name": "gpfs-ci-root",
            "mountPath": "/root/miles",
            "subPath": str(staged.miles_dir.relative_to(gpfs_root)),
            "readOnly": True,
        },
        {"name": "models", "mountPath": "/root/models", "readOnly": True},
        {"name": "jit-cache", "mountPath": config.jit_cache_root},
    ]
    if staged.megatron_dir is not None:
        volume_mounts.append(
            {
                "name": "gpfs-ci-root",
                "mountPath": "/root/Megatron-LM",
                "subPath": str(staged.megatron_dir.relative_to(gpfs_root)),
                "readOnly": True,
            }
        )
    if staged.sglang_dir is not None:
        volume_mounts.append(
            {
                "name": "gpfs-ci-root",
                "mountPath": "/sgl-workspace/sglang",
                "subPath": str(staged.sglang_dir.relative_to(gpfs_root)),
                "readOnly": True,
            }
        )

    labels = {
        CI_ROLE_LABEL: CI_DRIVER_ROLE,
        CI_JOB_ID_LABEL: config.ci_job_id,
    }
    pod_spec: dict[str, Any] = {
        "serviceAccountName": config.service_account,
        "restartPolicy": "Never",
        "enableServiceLinks": False,
        "containers": [
            {
                "name": "driver",
                "image": pinned_image,
                "imagePullPolicy": "IfNotPresent",
                "workingDir": "/root/miles",
                "command": list(config.command),
                "env": [{"name": k, "value": v} for k, v in sorted(env_map.items())],
                "securityContext": {
                    "allowPrivilegeEscalation": False,
                    "capabilities": {"drop": ["ALL"]},
                },
                "resources": {
                    "requests": {"cpu": "4", "memory": "16Gi"},
                    "limits": {"cpu": "8", "memory": "32Gi"},
                },
                "volumeMounts": volume_mounts,
            }
        ],
        "volumes": [
            {
                "name": "gpfs-ci-root",
                "hostPath": {"path": str(gpfs_root), "type": "DirectoryOrCreate"},
            },
            {
                "name": "models",
                "hostPath": {"path": config.models_root, "type": "Directory"},
            },
            {
                "name": "jit-cache",
                "hostPath": {"path": config.jit_cache_root, "type": "DirectoryOrCreate"},
            },
        ],
    }
    if config.node_selector:
        pod_spec["nodeSelector"] = dict(config.node_selector)
    if config.tolerations:
        pod_spec["tolerations"] = list(config.tolerations)

    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": job_name,
            "namespace": config.namespace,
            "labels": labels,
        },
        "spec": {
            "backoffLimit": 0,
            "activeDeadlineSeconds": config.active_deadline_seconds,
            "ttlSecondsAfterFinished": 600,
            "template": {
                "metadata": {"labels": labels},
                "spec": pod_spec,
            },
        },
    }


def cleanup_ci_job(*, namespace: str, ci_job_id: str, delete_driver_job: bool = True) -> None:
    """Sweep the Driver Job and any leaked Helm releases or command-jobs for one ci_job_id."""
    if delete_driver_job:
        subprocess.run(
            [
                "kubectl",
                "delete",
                "job",
                driver_job_name(ci_job_id),
                "--namespace",
                namespace,
                "--ignore-not-found",
                "--wait=false",
            ],
            check=False,
        )

    selector = f"{CI_JOB_ID_LABEL}={ci_job_id}"
    if shutil.which("helm"):
        listed = subprocess.run(
            ["helm", "list", "--namespace", namespace, "--selector", selector, "--output", "json"],
            check=False,
            capture_output=True,
            text=True,
        )
        if listed.returncode == 0 and listed.stdout.strip():
            for release in json.loads(listed.stdout):
                name = release.get("name")
                if name:
                    subprocess.run(
                        ["helm", "uninstall", name, "--namespace", namespace, "--ignore-not-found"],
                        check=False,
                    )

    subprocess.run(
        [
            "kubectl",
            "delete",
            "job,service",
            "--namespace",
            namespace,
            "--selector",
            selector,
            "--ignore-not-found",
        ],
        check=False,
    )


def run_driver_job(
    *,
    config: DriverJobConfig,
    staged: StagedSources,
    poll_interval_seconds: float = _DEFAULT_POD_START_POLL_SECONDS,
) -> int:
    """Pin image, write infra values, submit Driver Job, stream logs, and return exact exit code."""
    pinned_image = resolve_image_digest(config.image)
    infra_values_path = staged.staging_root / "infra.yaml"
    infra_values_path.write_text(yaml.safe_dump(build_infra_values(config=config, staged=staged, pinned_image=pinned_image)))

    manifest = build_driver_job_manifest(
        config=config,
        staged=staged,
        pinned_image=pinned_image,
        infra_values_path=infra_values_path,
    )
    job_name = manifest["metadata"]["name"]

    def _handle_signal(signum: int, _frame: Any) -> None:
        print(f"Received signal {signum}; cleaning up K8s CI job {config.ci_job_id}", file=sys.stderr, flush=True)
        cleanup_ci_job(namespace=config.namespace, ci_job_id=config.ci_job_id)
        raise SystemExit(128 + signum)

    prev_sigterm = signal.signal(signal.SIGTERM, _handle_signal)
    prev_sigint = signal.signal(signal.SIGINT, _handle_signal)
    try:
        subprocess.run(
            ["kubectl", "apply", "--namespace", config.namespace, "-f", "-"],
            input=yaml.safe_dump(manifest),
            text=True,
            check=True,
        )
        pod_name = _wait_for_driver_pod_ready_for_logs(
            namespace=config.namespace,
            job_name=job_name,
            timeout_seconds=config.admission_timeout_seconds,
            poll_interval_seconds=poll_interval_seconds,
        )
        subprocess.run(
            ["kubectl", "logs", "--namespace", config.namespace, "-c", "driver", "--follow", f"pod/{pod_name}"],
            check=False,
        )
        return _read_driver_exit_code(
            namespace=config.namespace,
            job_name=job_name,
            pod_name=pod_name,
            poll_interval_seconds=poll_interval_seconds,
        )
    finally:
        signal.signal(signal.SIGTERM, prev_sigterm)
        signal.signal(signal.SIGINT, prev_sigint)
        cleanup_ci_job(namespace=config.namespace, ci_job_id=config.ci_job_id)


def _wait_for_driver_pod_ready_for_logs(
    *,
    namespace: str,
    job_name: str,
    timeout_seconds: float,
    poll_interval_seconds: float,
) -> str:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        result = subprocess.run(
            [
                "kubectl",
                "get",
                "pods",
                "--namespace",
                namespace,
                "--selector",
                f"batch.kubernetes.io/job-name={job_name}",
                "--output",
                "json",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        items = json.loads(result.stdout or "{}").get("items", [])
        if items:
            pod = items[0]
            phase = (pod.get("status") or {}).get("phase")
            if phase in ("Running", "Succeeded", "Failed"):
                return pod["metadata"]["name"]
        time.sleep(poll_interval_seconds)
    raise TimeoutError(f"Driver Job {job_name} in {namespace} did not start within {timeout_seconds}s")


def _read_driver_exit_code(
    *,
    namespace: str,
    job_name: str,
    pod_name: str,
    poll_interval_seconds: float,
    max_polls: int = 36,
) -> int:
    for _ in range(max_polls):
        pod_result = subprocess.run(
            ["kubectl", "get", "pod", pod_name, "--namespace", namespace, "--output", "json"],
            check=False,
            capture_output=True,
            text=True,
        )
        if pod_result.returncode == 0 and pod_result.stdout.strip():
            pod = json.loads(pod_result.stdout)
            for status in (pod.get("status") or {}).get("containerStatuses", []):
                terminated = (status.get("state") or {}).get("terminated")
                if terminated is not None and "exitCode" in terminated:
                    return int(terminated["exitCode"])

        job_result = subprocess.run(
            ["kubectl", "get", "job", job_name, "--namespace", namespace, "--output", "json"],
            check=False,
            capture_output=True,
            text=True,
        )
        if job_result.returncode == 0 and job_result.stdout.strip():
            conditions = (json.loads(job_result.stdout).get("status") or {}).get("conditions", [])
            for cond in conditions:
                if cond.get("status") != "True":
                    continue
                if cond.get("type") == "Complete":
                    return 0
                if cond.get("type") == "Failed":
                    return 1
        time.sleep(poll_interval_seconds)
    return 1


def reap_orphaned_jobs(*, namespace: str, gpfs_root: str | Path, max_age_seconds: int) -> dict[str, list[str]]:
    """Sweep leaked Helm releases/command-jobs without an active Driver Job and expired GPFS dirs."""
    active_job_ids = _list_active_driver_job_ids(namespace=namespace)
    reaped_releases: list[str] = []
    reaped_job_ids: set[str] = set()

    if shutil.which("helm"):
        listed = subprocess.run(
            ["helm", "list", "--namespace", namespace, "--selector", CI_JOB_ID_LABEL, "--output", "json"],
            check=False,
            capture_output=True,
            text=True,
        )
        if listed.returncode == 0 and listed.stdout.strip():
            for release in json.loads(listed.stdout):
                name = release.get("name", "")
                ci_job_id = _helm_release_ci_job_id(namespace=namespace, release=name)
                if ci_job_id and ci_job_id not in active_job_ids:
                    subprocess.run(
                        ["helm", "uninstall", name, "--namespace", namespace, "--ignore-not-found"],
                        check=False,
                    )
                    reaped_releases.append(name)
                    reaped_job_ids.add(ci_job_id)

    for ci_job_id in _list_labelled_command_job_ids(namespace=namespace):
        if ci_job_id not in active_job_ids:
            cleanup_ci_job(namespace=namespace, ci_job_id=ci_job_id, delete_driver_job=False)
            reaped_job_ids.add(ci_job_id)

    reaped_dirs = _reap_expired_gpfs_dirs(
        gpfs_root=Path(gpfs_root),
        active_job_ids=active_job_ids,
        max_age_seconds=max_age_seconds,
    )
    return {
        "reaped_releases": sorted(reaped_releases),
        "reaped_job_ids": sorted(reaped_job_ids),
        "reaped_dirs": sorted(reaped_dirs),
    }


def _list_active_driver_job_ids(*, namespace: str) -> set[str]:
    result = subprocess.run(
        [
            "kubectl",
            "get",
            "jobs",
            "--namespace",
            namespace,
            "--selector",
            f"{CI_ROLE_LABEL}={CI_DRIVER_ROLE}",
            "--output",
            "json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    active: set[str] = set()
    for item in json.loads(result.stdout or "{}").get("items", []):
        conditions = (item.get("status") or {}).get("conditions", [])
        is_terminal = any(
            cond.get("status") == "True" and cond.get("type") in ("Complete", "Failed") for cond in conditions
        )
        if not is_terminal:
            ci_job_id = ((item.get("metadata") or {}).get("labels") or {}).get(CI_JOB_ID_LABEL)
            if ci_job_id:
                active.add(ci_job_id)
    return active


def _helm_release_ci_job_id(*, namespace: str, release: str) -> str | None:
    result = subprocess.run(
        [
            "kubectl",
            "get",
            "secrets",
            "--namespace",
            namespace,
            "--selector",
            f"owner=helm,name={release}",
            "--output",
            "json",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return None
    for item in json.loads(result.stdout).get("items", []):
        ci_job_id = ((item.get("metadata") or {}).get("labels") or {}).get(CI_JOB_ID_LABEL)
        if ci_job_id:
            return ci_job_id
    return None


def _list_labelled_command_job_ids(*, namespace: str) -> set[str]:
    result = subprocess.run(
        [
            "kubectl",
            "get",
            "jobs",
            "--namespace",
            namespace,
            "--selector",
            f"{CI_JOB_ID_LABEL},{CI_ROLE_LABEL}!={CI_DRIVER_ROLE}",
            "--output",
            "json",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return set()
    found: set[str] = set()
    for item in json.loads(result.stdout).get("items", []):
        ci_job_id = ((item.get("metadata") or {}).get("labels") or {}).get(CI_JOB_ID_LABEL)
        if ci_job_id:
            found.add(ci_job_id)
    return found


def _reap_expired_gpfs_dirs(*, gpfs_root: Path, active_job_ids: set[str], max_age_seconds: int) -> list[str]:
    now = time.time()
    removed: list[str] = []
    for bucket in ("staging", "runs"):
        bucket_dir = gpfs_root / bucket
        if not bucket_dir.is_dir():
            continue
        for child in bucket_dir.iterdir():
            if not child.is_dir() or child.name in active_job_ids:
                continue
            age = now - child.stat().st_mtime
            if age >= max_age_seconds:
                shutil.rmtree(child, ignore_errors=True)
                removed.append(str(child))
    return removed


def _parse_kv_pairs(items: Sequence[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        parsed[key] = value
    return parsed


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    run_p = subparsers.add_parser("run", help="Stage sources, submit Driver Job, stream logs, and clean up")
    run_p.add_argument("--namespace", default=None)
    run_p.add_argument("--gpfs-root", default=None)
    run_p.add_argument("--image", required=True)
    run_p.add_argument("--ci-job-id", required=True)
    run_p.add_argument("--miles-source-dir", required=True)
    run_p.add_argument("--megatron-source-dir", default=None)
    run_p.add_argument("--sglang-source-dir", default=None)
    run_p.add_argument("--megatron-bridge-ref", default="")
    run_p.add_argument("--pr-body", default="")
    run_p.add_argument("--miles-changed-file", action="append", default=[])
    run_p.add_argument("--sglang-changed-file", action="append", default=[])
    run_p.add_argument("--service-account", default=_DEFAULT_SERVICE_ACCOUNT)
    run_p.add_argument("--models-root", default=_DEFAULT_MODELS_ROOT)
    run_p.add_argument("--jit-cache-root", default=_DEFAULT_JIT_CACHE_ROOT)
    run_p.add_argument("--rdma-resource", default=None)
    run_p.add_argument("--colocate-shm-host-path", default=None)
    run_p.add_argument("--active-deadline-seconds", type=int, default=_DEFAULT_ACTIVE_DEADLINE_SECONDS)
    run_p.add_argument("--admission-timeout-seconds", type=int, default=_DEFAULT_ADMISSION_TIMEOUT_SECONDS)
    run_p.add_argument("--execution-timeout-seconds", type=int, default=_DEFAULT_EXECUTION_TIMEOUT_SECONDS)
    run_p.add_argument("--env", action="append", default=[], help="Extra KEY=VALUE env vars for the Driver Job")
    run_p.add_argument("command", nargs=argparse.REMAINDER)

    clean_p = subparsers.add_parser("cleanup", help="Clean up one CI job's Driver Job, Helm releases, and command-jobs")
    clean_p.add_argument("--namespace", default=None)
    clean_p.add_argument("--ci-job-id", required=True)

    reap_p = subparsers.add_parser("reap", help="Sweep orphaned releases/command-jobs and expired GPFS staging dirs")
    reap_p.add_argument("--namespace", default=None)
    reap_p.add_argument("--gpfs-root", default=None)
    reap_p.add_argument("--max-age-seconds", type=int, default=10800)

    args = parser.parse_args(argv)

    if args.subcommand == "cleanup":
        namespace = require_config_value(args.namespace, env_var="K8S_NAMESPACE", flag="--namespace")
        cleanup_ci_job(namespace=namespace, ci_job_id=sanitize_ci_job_id(args.ci_job_id))
        return 0

    if args.subcommand == "reap":
        namespace = require_config_value(args.namespace, env_var="K8S_NAMESPACE", flag="--namespace")
        gpfs_root = require_config_value(args.gpfs_root, env_var="K8S_GPFS_ROOT", flag="--gpfs-root")
        summary = reap_orphaned_jobs(namespace=namespace, gpfs_root=gpfs_root, max_age_seconds=args.max_age_seconds)
        print(json.dumps(summary, indent=2), flush=True)
        return 0

    namespace = require_config_value(args.namespace, env_var="K8S_NAMESPACE", flag="--namespace")
    gpfs_root = require_config_value(args.gpfs_root, env_var="K8S_GPFS_ROOT", flag="--gpfs-root")
    ci_job_id = sanitize_ci_job_id(args.ci_job_id)

    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise ValueError("Missing driver command after '--'")

    check_k8s_dependency_support(
        megatron_bridge_ref=args.megatron_bridge_ref,
        pr_body=args.pr_body,
        miles_changed_files=args.miles_changed_file,
        sglang_changed_files=args.sglang_changed_file,
    )
    staged = stage_sources(
        gpfs_root=gpfs_root,
        ci_job_id=ci_job_id,
        miles_source_dir=args.miles_source_dir,
        megatron_source_dir=args.megatron_source_dir,
        sglang_source_dir=args.sglang_source_dir,
    )
    config = DriverJobConfig(
        namespace=namespace,
        gpfs_root=gpfs_root,
        image=args.image,
        ci_job_id=ci_job_id,
        command=tuple(command),
        service_account=args.service_account,
        models_root=args.models_root,
        jit_cache_root=args.jit_cache_root,
        rdma_resource=args.rdma_resource,
        colocate_shm_host_path=args.colocate_shm_host_path,
        active_deadline_seconds=args.active_deadline_seconds,
        admission_timeout_seconds=args.admission_timeout_seconds,
        execution_timeout_seconds=args.execution_timeout_seconds,
        extra_env=_parse_kv_pairs(args.env),
    )
    return run_driver_job(config=config, staged=staged)


if __name__ == "__main__":
    raise SystemExit(main())
