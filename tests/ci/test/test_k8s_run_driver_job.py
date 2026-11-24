from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import pytest
import yaml

from tests.ci import k8s_run_driver_job as launcher
from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-cpu", labels=[])

_MANIFEST_DIR = Path(__file__).resolve().parents[1] / "k8s_runner"
_FAKE_DIGEST = "sha256:" + "a" * 64


class TestK8sRunnerManifests:
    def test_all_yaml_manifests_parse_and_enforce_security_boundaries(self):
        docs_by_file: dict[str, list[dict[str, Any]]] = {}
        for path in sorted(_MANIFEST_DIR.glob("*.yaml")):
            docs = [doc for doc in yaml.safe_load_all(path.read_text()) if doc is not None]
            assert docs, f"{path.name} is empty"
            docs_by_file[path.name] = docs

        assert set(docs_by_file) == {
            "admission-policy.yaml",
            "network-policy.yaml",
            "reaper-cronjob.yaml",
            "runner-deployment.yaml",
            "runner-launcher-rbac.yaml",
        }

        rbac_roles = {
            doc["metadata"]["name"]: doc["rules"]
            for doc in docs_by_file["runner-launcher-rbac.yaml"]
            if doc["kind"] == "Role"
        }
        launcher_resources = {res for rule in rbac_roles["runner-launcher"] for res in rule["resources"]}
        assert launcher_resources == {"jobs", "pods", "pods/log", "events"}
        assert not ({"secrets", "pods/exec", "leaderworkersets", "services"} & launcher_resources)

        [deployment] = docs_by_file["runner-deployment.yaml"]
        pod_spec = deployment["spec"]["template"]["spec"]
        assert pod_spec["serviceAccountName"] == "runner-launcher"
        assert pod_spec["securityContext"]["runAsNonRoot"] is True
        assert all(
            (vol.get("hostPath") or {}).get("path") != "/var/run/docker.sock" for vol in pod_spec.get("volumes", [])
        )
        [main_container] = pod_spec["containers"]
        assert "nvidia.com/gpu" not in (main_container.get("resources", {}).get("limits", {}))
        assert main_container["securityContext"]["capabilities"]["drop"] == ["ALL"]

        vap = next(doc for doc in docs_by_file["admission-policy.yaml"] if doc["kind"] == "ValidatingAdmissionPolicy")
        expressions = "\n".join(rule["expression"] for rule in vap["spec"]["validations"])
        assert "IPC_LOCK" in expressions
        assert "NVIDIA_VISIBLE_DEVICES" in expressions
        assert "/storage/shared/miles-ci-k8s/ext" in expressions


class TestConfigAndDependencyBoundaries:
    def test_missing_namespace_or_gpfs_root_fails_fast(self, monkeypatch):
        monkeypatch.delenv("K8S_NAMESPACE", raising=False)
        monkeypatch.delenv("K8S_GPFS_ROOT", raising=False)

        with pytest.raises(ValueError, match="K8S_NAMESPACE"):
            launcher.require_config_value(None, env_var="K8S_NAMESPACE", flag="--namespace")
        with pytest.raises(ValueError, match="K8S_GPFS_ROOT"):
            launcher.require_config_value(None, env_var="K8S_GPFS_ROOT", flag="--gpfs-root")

    def test_allows_pure_python_miles_megatron_and_sglang_changes(self):
        launcher.check_k8s_dependency_support(
            miles_changed_files=["miles/ray/actor.py", "tests/e2e/test_foo.py"],
            sglang_changed_files=["python/sglang/srt/server.py", "radix_transfer_plugins/python/plugin.py"],
        )

    def test_rejects_megatron_bridge_override_with_ray_hint(self):
        with pytest.raises(RuntimeError, match="run-on-ray") as exc_info:
            launcher.check_k8s_dependency_support(pr_body="fixes x\nci-megatron-bridge-pr: #42\n")
        assert "ci-megatron-bridge-pr" in str(exc_info.value)

    def test_rejects_radix_transfer_plugins_pyproject_and_multi_lora_requirements(self):
        with pytest.raises(RuntimeError, match="run-on-ray") as exc_info:
            launcher.check_k8s_dependency_support(
                miles_changed_files=["examples/multi_lora/requirements.txt"],
                sglang_changed_files=["radix_transfer_plugins/pyproject.toml"],
            )
        msg = str(exc_info.value)
        assert "examples/multi_lora/requirements.txt" in msg
        assert "radix_transfer_plugins/pyproject.toml" in msg


class TestDigestPinningAndManifestRendering:
    def test_resolves_mutable_tag_to_immutable_digest_and_splits_for_helm(self, monkeypatch):
        monkeypatch.setattr(launcher.shutil, "which", lambda tool: "/usr/bin/crane" if tool == "crane" else None)
        monkeypatch.setattr(
            launcher.subprocess,
            "run",
            lambda cmd, **kwargs: subprocess.CompletedProcess(args=cmd, returncode=0, stdout=f"{_FAKE_DIGEST}\n"),
        )

        pinned = launcher.resolve_image_digest("radixark/miles:pr-123")
        assert pinned == f"radixark/miles:pr-123@{_FAKE_DIGEST}"
        assert launcher.split_image_for_helm(pinned) == ("radixark/miles", f"pr-123@{_FAKE_DIGEST}")

    def test_stages_sources_and_mounts_readonly_in_infra_and_driver_job(self, tmp_path: Path):
        gpfs_root = tmp_path / "gpfs"
        miles_src = tmp_path / "src-miles"
        megatron_src = tmp_path / "src-megatron"
        sglang_src = tmp_path / "src-sglang"
        for directory, marker in ((miles_src, "miles.py"), (megatron_src, "megatron.py"), (sglang_src, "sglang.py")):
            directory.mkdir()
            (directory / marker).write_text("# ok\n")

        staged = launcher.stage_sources(
            gpfs_root=gpfs_root,
            ci_job_id="gh-101-1",
            miles_source_dir=miles_src,
            megatron_source_dir=megatron_src,
            sglang_source_dir=sglang_src,
        )
        config = launcher.DriverJobConfig(
            namespace="miles-ci-k8s",
            gpfs_root=str(gpfs_root),
            image=f"radixark/miles:dev@{_FAKE_DIGEST}",
            ci_job_id="gh-101-1",
            command=("python3", "tests/ci/run_suite.py"),
            rdma_resource="rdma/hca_shared_devices_all",
            colocate_shm_host_path="/dev/shm/miles-${RELEASE}",
            extra_env={"MILES_SCRIPT_CI_RUN": "true", "CUSTOM_FLAG": "1"},
        )
        infra = launcher.build_infra_values(config=config, staged=staged, pinned_image=config.image)["infra"]
        gpfs_mounts = {m["mountPath"]: m for m in infra["volumes"][0]["mounts"]}
        assert gpfs_mounts["/root/miles"]["readOnly"] is True
        assert gpfs_mounts["/root/Megatron-LM"]["readOnly"] is True
        assert gpfs_mounts["/sgl-workspace/sglang"]["readOnly"] is True
        assert infra["gpuResources"] == {"rdma/hca_shared_devices_all": 1}
        assert infra["devShm"]["colocateHostPath"]["path"] == "/dev/shm/miles-${RELEASE}"

        manifest = launcher.build_driver_job_manifest(
            config=config,
            staged=staged,
            pinned_image=config.image,
            infra_values_path=staged.staging_root / "infra.yaml",
        )
        container = manifest["spec"]["template"]["spec"]["containers"][0]
        env = {entry["name"]: entry["value"] for entry in container["env"]}
        assert manifest["spec"]["template"]["spec"]["serviceAccountName"] == "miles-workbench"
        assert "nvidia.com/gpu" not in container["resources"]["limits"]
        assert env["MILES_CI_JOB_ID"] == "gh-101-1"
        assert env["MILES_SCRIPT_CLUSTER_BACKEND"] == "kubernetes"
        assert "MILES_SCRIPT_CI_RUN" not in env
        assert env["CUSTOM_FLAG"] == "1"

    def test_generated_default_infra_values_satisfy_shared_infra_schema(self, tmp_path: Path):
        jsonschema = pytest.importorskip("jsonschema")
        gpfs_root = tmp_path / "gpfs"
        miles_src = tmp_path / "src-miles"
        miles_src.mkdir()
        (miles_src / "m.py").write_text("")
        staged = launcher.stage_sources(gpfs_root=gpfs_root, ci_job_id="gh-202-1", miles_source_dir=miles_src)
        config = launcher.DriverJobConfig(
            namespace="miles-ci-k8s",
            gpfs_root=str(gpfs_root),
            image=f"radixark/miles:dev@{_FAKE_DIGEST}",
            ci_job_id="gh-202-1",
            command=("python3", "-V"),
        )
        values = launcher.build_infra_values(config=config, staged=staged, pinned_image=config.image)
        schema_path = Path(__file__).resolve().parents[3] / "charts" / "shared-infra.schema.json"
        schema = json.loads(schema_path.read_text())

        jsonschema.Draft7Validator(schema).validate(values)


class TestDriverLifecycleAndReaper:
    def test_propagates_exact_driver_container_exit_code_and_cleans_up(self, monkeypatch, tmp_path: Path):
        gpfs_root = tmp_path / "gpfs"
        miles_src = tmp_path / "src-miles"
        miles_src.mkdir()
        (miles_src / "dummy.py").write_text("")
        staged = launcher.stage_sources(gpfs_root=gpfs_root, ci_job_id="gh-42-1", miles_source_dir=miles_src)

        commands: list[list[str]] = []

        def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
            commands.append(cmd)
            if cmd[:3] == ["kubectl", "get", "pods"]:
                payload = {"items": [{"metadata": {"name": "miles-ci-driver-gh-42-1-xyz"}, "status": {"phase": "Running"}}]}
                return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=json.dumps(payload), stderr="")
            if cmd[:3] == ["kubectl", "get", "pod"]:
                payload = {
                    "status": {
                        "containerStatuses": [{"name": "driver", "state": {"terminated": {"exitCode": 17}}}]
                    }
                }
                return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=json.dumps(payload), stderr="")
            if cmd[:2] == ["helm", "list"]:
                return subprocess.CompletedProcess(
                    args=cmd, returncode=0, stdout=json.dumps([{"name": "miles-run-leaked-all"}]), stderr=""
                )
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

        monkeypatch.setattr(launcher.subprocess, "run", fake_run)
        monkeypatch.setattr(launcher.shutil, "which", lambda tool: f"/usr/bin/{tool}")

        config = launcher.DriverJobConfig(
            namespace="miles-ci-k8s",
            gpfs_root=str(gpfs_root),
            image=f"radixark/miles:dev@{_FAKE_DIGEST}",
            ci_job_id="gh-42-1",
            command=("pytest", "tests/e2e/foo.py"),
        )
        exit_code = launcher.run_driver_job(config=config, staged=staged, poll_interval_seconds=0.0)

        assert exit_code == 17
        assert any(cmd[:3] == ["kubectl", "logs", "--namespace"] and "--follow" in cmd for cmd in commands)
        assert any(cmd[:3] == ["helm", "uninstall", "miles-run-leaked-all"] for cmd in commands)
        assert any(cmd[:3] == ["kubectl", "delete", "job,service"] for cmd in commands)

    def test_reaper_sweeps_only_orphaned_releases_and_expired_gpfs_dirs(self, monkeypatch, tmp_path: Path):
        gpfs_root = tmp_path / "gpfs"
        active_dir = gpfs_root / "staging" / "job-active"
        orphan_old = gpfs_root / "staging" / "job-orphan"
        orphan_fresh = gpfs_root / "runs" / "job-fresh"
        for directory in (active_dir, orphan_old, orphan_fresh):
            directory.mkdir(parents=True)
        old_mtime = time.time() - 7200
        os.utime(orphan_old, (old_mtime, old_mtime))
        os.utime(active_dir, (old_mtime, old_mtime))

        uninstalled: list[str] = []

        def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
            if cmd[:3] == ["kubectl", "get", "jobs"] and f"{launcher.CI_ROLE_LABEL}={launcher.CI_DRIVER_ROLE}" in cmd:
                payload = {
                    "items": [
                        {"metadata": {"labels": {launcher.CI_JOB_ID_LABEL: "job-active"}}, "status": {"conditions": []}},
                        {
                            "metadata": {"labels": {launcher.CI_JOB_ID_LABEL: "job-orphan"}},
                            "status": {"conditions": [{"type": "Failed", "status": "True"}]},
                        },
                    ]
                }
                return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=json.dumps(payload), stderr="")
            if cmd[:2] == ["helm", "list"]:
                payload = [{"name": "rel-active"}, {"name": "rel-orphan"}]
                return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=json.dumps(payload), stderr="")
            if cmd[:3] == ["kubectl", "get", "secrets"]:
                sel = cmd[cmd.index("--selector") + 1]
                job_id = "job-active" if "rel-active" in sel else "job-orphan"
                payload = {"items": [{"metadata": {"labels": {launcher.CI_JOB_ID_LABEL: job_id}}}]}
                return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=json.dumps(payload), stderr="")
            if cmd[:2] == ["helm", "uninstall"]:
                uninstalled.append(cmd[2])
                return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout='{"items": []}', stderr="")

        monkeypatch.setattr(launcher.subprocess, "run", fake_run)
        monkeypatch.setattr(launcher.shutil, "which", lambda tool: f"/usr/bin/{tool}")

        summary = launcher.reap_orphaned_jobs(namespace="miles-ci-k8s", gpfs_root=gpfs_root, max_age_seconds=3600)

        assert uninstalled == ["rel-orphan"]
        assert summary["reaped_releases"] == ["rel-orphan"]
        assert summary["reaped_dirs"] == [str(orphan_old)]
        assert active_dir.exists()
        assert orphan_fresh.exists()
        assert not orphan_old.exists()
