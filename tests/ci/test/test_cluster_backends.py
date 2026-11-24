from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from tests.ci.ci_register import (
    DEFAULT_CLUSTER_BACKENDS,
    HWBackend,
    KNOWN_CLUSTER_BACKENDS,
    collect_tests,
    discover_ci_files,
    register_cpu_ci,
    ut_parse_one_file,
)

register_cpu_ci(est_time=5, suite="stage-a-cpu", labels=[])

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _resolve_local_import(module: str, cur_file: Path, level: int = 0) -> Path | None:
    if level > 0:
        base = cur_file.parent
        for _ in range(level - 1):
            base = base.parent
        rel = base.joinpath(*(module.split(".") if module else []))
    else:
        if not module or not module.startswith(("tests.", "scripts.", "examples.")):
            return None
        rel = _REPO_ROOT.joinpath(*module.split("."))
    for candidate in (rel.with_suffix(".py"), rel / "__init__.py"):
        if candidate.is_file():
            return candidate
    return None


def _drives_cluster_backend(path: Path, cache: dict[Path, bool], seen: set[Path] | None = None) -> bool:
    if path in cache:
        return cache[path]
    if seen is None:
        seen = set()
    if path in seen:
        return False
    seen.add(path)

    text = path.read_text()
    if any(
        token in text
        for token in (
            "execute_train",
            "helm_backend",
            "MILES_SCRIPT_CLUSTER_BACKEND",
            "create_backend",
            "BaseCommandBackend",
            "ExecuteTrainConfig",
        )
    ):
        cache[path] = True
        return True

    if "command_utils" in text and "muse_spark" in text:
        muse_common = _REPO_ROOT / "scripts" / "muse_spark" / "common.py"
        if muse_common.is_file() and _drives_cluster_backend(muse_common, cache, seen):
            cache[path] = True
            return True

    for match in re.findall(r'["\']((?:scripts|examples|tests)/[A-Za-z0-9_./-]+\.py)["\']', text):
        target = _REPO_ROOT / match
        if target.is_file() and _drives_cluster_backend(target, cache, seen):
            cache[path] = True
            return True

    try:
        tree = ast.parse(text)
    except SyntaxError:
        cache[path] = False
        return False

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            names = [node.module or ""] + [
                f"{node.module}.{alias.name}" if node.module else alias.name for alias in node.names
            ]
            for mod_name in names:
                target = _resolve_local_import(mod_name, path, node.level)
                if target and _drives_cluster_backend(target, cache, seen):
                    cache[path] = True
                    return True
        elif isinstance(node, ast.Import):
            for alias in node.names:
                target = _resolve_local_import(alias.name, path, 0)
                if target and _drives_cluster_backend(target, cache, seen):
                    cache[path] = True
                    return True

    cache[path] = False
    return False


class TestClusterBackendsParsing:
    def test_defaults_to_ray_when_omitted_on_cuda(self, tmp_path: Path):
        f = tmp_path / "test_legacy.py"
        f.write_text(
            "from tests.ci.ci_register import register_cuda_ci\n"
            'register_cuda_ci(est_time=60, suite="stage-c-4-gpu-h200", labels=["short"], hardware=["hopper"])\n'
        )
        [reg] = ut_parse_one_file(str(f))
        assert reg.cluster_backends == list(DEFAULT_CLUSTER_BACKENDS)

    @pytest.mark.parametrize(
        "declared",
        [
            ["ray"],
            ["kubernetes"],
            ["ray", "kubernetes"],
        ],
    )
    def test_accepts_valid_cluster_backend_subsets(self, tmp_path: Path, declared: list[str]):
        f = tmp_path / "test_ok.py"
        f.write_text(
            "from tests.ci.ci_register import register_cuda_ci\n"
            f'register_cuda_ci(est_time=60, suite="stage-c-4-gpu-h200", labels=["short"], '
            f'hardware=["hopper"], cluster_backends={declared!r})\n'
        )
        [reg] = ut_parse_one_file(str(f))
        assert reg.cluster_backends == declared

    @pytest.mark.parametrize(
        ("declared", "error_match"),
        [
            ("[]", "at least one backend"),
            ('["slurm"]', "unknown cluster_backends"),
            ('["ray", "ray"]', "duplicated cluster_backend"),
        ],
    )
    def test_rejects_invalid_cluster_backends(self, tmp_path: Path, declared: str, error_match: str):
        f = tmp_path / "test_bad.py"
        f.write_text(
            "from tests.ci.ci_register import register_cuda_ci\n"
            f'register_cuda_ci(est_time=60, suite="stage-c-4-gpu-h200", labels=["short"], '
            f"hardware=['hopper'], cluster_backends={declared})\n"
        )
        with pytest.raises(ValueError, match=error_match):
            ut_parse_one_file(str(f))

    def test_rejects_cluster_backends_on_cpu_and_rocm(self, tmp_path: Path):
        cpu_file = tmp_path / "test_cpu.py"
        cpu_file.write_text(
            "from tests.ci.ci_register import register_cpu_ci\n"
            'register_cpu_ci(est_time=10, suite="stage-a-cpu", cluster_backends=["ray"])\n'
        )
        with pytest.raises(ValueError, match="CUDA-only"):
            ut_parse_one_file(str(cpu_file))

        rocm_file = tmp_path / "test_rocm.py"
        rocm_file.write_text(
            "from tests.ci.ci_register import register_rocm_ci\n"
            'register_rocm_ci(est_time=60, suite="stage-c-4-gpu-mi350", labels=["short"], cluster_backends=["ray"])\n'
        )
        with pytest.raises(ValueError, match="CUDA-only"):
            ut_parse_one_file(str(rocm_file))


class TestRepoClusterBackendsLedgerGuard:
    def test_every_cuda_registration_declares_cluster_backends_explicitly(self, monkeypatch):
        monkeypatch.chdir(_REPO_ROOT)
        missing: list[str] = []
        for path_str in discover_ci_files():
            tree = ast.parse((_REPO_ROOT / path_str).read_text(), filename=path_str)
            for stmt in tree.body:
                if (
                    isinstance(stmt, ast.Expr)
                    and isinstance(stmt.value, ast.Call)
                    and isinstance(stmt.value.func, ast.Name)
                    and stmt.value.func.id == "register_cuda_ci"
                ):
                    if not any(kw.arg == "cluster_backends" for kw in stmt.value.keywords):
                        missing.append(path_str)
        assert missing == [], f"register_cuda_ci calls missing explicit cluster_backends: {missing}"

    def test_every_kubernetes_declared_test_drives_cluster_backend_and_uses_h200_stage(self, monkeypatch):
        """Rejects accidental kubernetes tagging on direct-GPU tests or non-H200 stages."""
        monkeypatch.chdir(_REPO_ROOT)
        regs = [r for r in collect_tests(discover_ci_files(), sanity_check=True) if r.backend == HWBackend.CUDA]
        cache: dict[Path, bool] = {}

        non_driver_k8s: list[str] = []
        non_h200_k8s: list[str] = []
        for reg in regs:
            assert set(reg.cluster_backends) <= set(KNOWN_CLUSTER_BACKENDS)
            if "kubernetes" not in reg.cluster_backends:
                continue
            if not _drives_cluster_backend(_REPO_ROOT / reg.filename, cache):
                non_driver_k8s.append(reg.filename)
            if "h100" in reg.suite or "b200" in reg.suite:
                non_h200_k8s.append(f"{reg.filename} ({reg.suite})")

        assert non_driver_k8s == [], f"Tests claiming kubernetes without command_utils/helm_backend: {non_driver_k8s}"
        assert non_h200_k8s == [], f"Non-H200 stage tests claiming kubernetes: {non_h200_k8s}"

    def test_deploy_and_lws_scaling_tests_are_kubernetes_only(self, monkeypatch):
        monkeypatch.chdir(_REPO_ROOT)
        regs = {
            r.filename: r
            for r in collect_tests(discover_ci_files(), sanity_check=True)
            if r.backend == HWBackend.CUDA
        }
        k8s_only_expected = {
            "tests/e2e/deploy/test_hot_restart_checkpointed.py",
            "tests/e2e/deploy/test_hot_restart_no_checkpoint.py",
            "tests/e2e/deploy/test_hot_restart_realistic_gsm8k.py",
            "tests/e2e/deploy/test_split_deterministic.py",
            "tests/e2e/deploy/test_split_multi_policy.py",
            "tests/e2e/ft/test_inference_scaling__kill_rollout.py",
            "tests/e2e/ft/test_trainer_scaling__kill_train.py",
        }
        for path_str in sorted(k8s_only_expected):
            assert regs[path_str].cluster_backends == ["kubernetes"], path_str
