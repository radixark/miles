import ast
import importlib
import os
import pkgutil
from importlib.util import find_spec
from pathlib import Path


MODULE_ROOT_ENV = {
    "miles": "GITHUB_WORKSPACE",
    "miles_plugins": "GITHUB_WORKSPACE",
    "sglang": "SGLANG_SOURCE_ROOT",
    "megatron.core": "MEGATRON_SOURCE_ROOT",
    "megatron.training": "MEGATRON_SOURCE_ROOT",
}

# Namespaces that only exist once the multi-lora stack lands; walked in full when present.
OPTIONAL_PACKAGES = (
    "miles.backends.megatron_utils.api_backends",
    "miles.ray.multi_lora",
    "miles.rollout.multi_lora",
    "miles.ray.tinker_frontend",
)


def _miles_roots() -> tuple[Path, Path]:
    spec = find_spec("miles")
    if spec is None or spec.origin is None:
        raise RuntimeError("cannot resolve miles")
    miles_root = Path(spec.origin).resolve().parent
    return miles_root, miles_root.parent


def _module_file_exists(repo_root: Path, dotted: str) -> bool:
    path = repo_root.joinpath(*dotted.split("."))
    return path.with_suffix(".py").is_file() or (path / "__init__.py").is_file()


def _resolve_relative(miles_root: Path, repo_root: Path, py_file: Path, node: ast.ImportFrom) -> str | None:
    if not py_file.is_relative_to(miles_root):
        return None
    parts = list(py_file.relative_to(repo_root).parts)
    package = parts[:-1]
    if node.level > 1:
        package = package[: -(node.level - 1)]
    return ".".join(package + node.module.split(".")) if node.module else ".".join(package)


def _iter_miles_import_targets(miles_root: Path, repo_root: Path, py_file: Path):
    tree = ast.parse(py_file.read_text(), filename=str(py_file))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.partition(".")[0] == "miles":
                    yield node.lineno, alias.name
        elif isinstance(node, ast.ImportFrom):
            target = _resolve_relative(miles_root, repo_root, py_file, node) if node.level else node.module
            if target and target.partition(".")[0] == "miles":
                yield node.lineno, target


def _python_files(root: Path):
    return (p for p in sorted(root.rglob("*.py")) if "__pycache__" not in p.parts)


def verify_import_sites_resolve() -> None:
    miles_root, repo_root = _miles_roots()
    stale = []
    roots = [miles_root] + ([repo_root / "examples"] if (repo_root / "examples").is_dir() else [])
    for root in roots:
        for py_file in _python_files(root):
            for lineno, target in _iter_miles_import_targets(miles_root, repo_root, py_file):
                if not _module_file_exists(repo_root, target):
                    stale.append(f"{py_file.relative_to(repo_root)}:{lineno}: {target}")
    if stale:
        raise RuntimeError("stale miles-internal imports:\n" + "\n".join(stale))
    print("import-integrity: every miles import site resolves")


def verify_optional_namespaces_import() -> None:
    for package_name in OPTIONAL_PACKAGES:
        if find_spec(package_name) is None:
            continue
        package = importlib.import_module(package_name)
        for info in pkgutil.walk_packages(package.__path__, prefix=package_name + "."):
            importlib.import_module(info.name)
        print(f"import-integrity: {package_name} imports in full")


def verify_update_weight_lazy_imports() -> None:
    miles_root, repo_root = _miles_roots()
    update_weight_dir = miles_root / "backends" / "megatron_utils" / "update_weight"
    targets = sorted(
        {
            target
            for py_file in _python_files(update_weight_dir)
            for _, target in _iter_miles_import_targets(miles_root, repo_root, py_file)
        }
    )
    if not targets:
        raise RuntimeError("expected function-local miles imports under update_weight/")
    for target in targets:
        try:
            importlib.import_module(target)
        except ModuleNotFoundError as exc:
            if (exc.name or "").partition(".")[0] == "miles":
                raise RuntimeError(f"update_weight lazy import target does not import: {target}") from exc
    print("import-integrity: update_weight lazy imports resolve")


def main() -> None:
    for module_name, root_env in MODULE_ROOT_ENV.items():
        expected_root = Path(os.environ[root_env]).resolve()
        spec = find_spec(module_name)
        if spec is None or spec.origin is None:
            raise RuntimeError(f"cannot resolve {module_name}")
        origin = Path(spec.origin).resolve()
        try:
            origin.relative_to(expected_root)
        except ValueError as exc:
            raise RuntimeError(f"{module_name} resolved to {origin}, expected {expected_root}") from exc
        print(f"{module_name}: {origin}")
    verify_import_sites_resolve()
    verify_optional_namespaces_import()
    verify_update_weight_lazy_imports()


if __name__ == "__main__":
    main()
