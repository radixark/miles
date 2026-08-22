"""Guards the api_backends/multi_lora restructure: moved namespaces import, and every miles-internal import site (incl. function-local) resolves."""

import ast
import importlib
import importlib.util
import pkgutil
from pathlib import Path

import miles

MILES_ROOT = Path(miles.__file__).resolve().parent
REPO_ROOT = MILES_ROOT.parent

# Frontend package exists only on stack heads that carry it; find_spec-gated below.
MOVED_PACKAGES = (
    "miles.backends.megatron_utils.api_backends",
    "miles.ray.multi_lora",
    "miles.rollout.multi_lora",
    "miles.ray.tinker_frontend",
)

# The publish path whose function-local import broke silently under CPU gates.
PUBLISH_PATH_DIR = MILES_ROOT / "backends" / "megatron_utils" / "update_weight"


def _module_file_exists(dotted: str) -> bool:
    path = REPO_ROOT.joinpath(*dotted.split("."))
    return path.with_suffix(".py").is_file() or (path / "__init__.py").is_file()


def _resolve_relative(py_file: Path, node: ast.ImportFrom) -> str | None:
    if not py_file.is_relative_to(MILES_ROOT):
        return None
    parts = list(py_file.relative_to(REPO_ROOT).parts)
    package = parts[:-1]
    if node.level > 1:
        package = package[: -(node.level - 1)]
    return ".".join(package + node.module.split(".")) if node.module else ".".join(package)


def _iter_miles_import_targets(py_file: Path):
    tree = ast.parse(py_file.read_text(), filename=str(py_file))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.partition(".")[0] == "miles":
                    yield node.lineno, alias.name
        elif isinstance(node, ast.ImportFrom):
            target = _resolve_relative(py_file, node) if node.level else node.module
            if target and target.partition(".")[0] == "miles":
                yield node.lineno, target


def _python_files(root: Path):
    return (p for p in sorted(root.rglob("*.py")) if "__pycache__" not in p.parts)


def test_moved_namespace_modules_all_import():
    """Every module under the restructured packages must import cleanly."""
    for package_name in MOVED_PACKAGES:
        if importlib.util.find_spec(package_name) is None:
            continue
        package = importlib.import_module(package_name)
        for info in pkgutil.walk_packages(package.__path__, prefix=package_name + "."):
            importlib.import_module(info.name)


def test_every_miles_import_site_resolves_statically():
    """Every miles.* import statement anywhere in miles/ and examples/ must name a real module."""
    stale = []
    roots = [MILES_ROOT] + ([REPO_ROOT / "examples"] if (REPO_ROOT / "examples").is_dir() else [])
    for root in roots:
        for py_file in _python_files(root):
            for lineno, target in _iter_miles_import_targets(py_file):
                if not _module_file_exists(target):
                    stale.append(f"{py_file.relative_to(REPO_ROOT)}:{lineno}: {target}")
    assert not stale, "stale miles-internal imports:\n" + "\n".join(stale)


def test_publish_path_function_local_imports_importable():
    """importlib-resolve the miles.* targets used inside update_weight function bodies."""
    targets = sorted(
        {target for py_file in _python_files(PUBLISH_PATH_DIR) for _, target in _iter_miles_import_targets(py_file)}
    )
    assert targets, "expected function-local miles imports under update_weight/"
    for target in targets:
        try:
            importlib.import_module(target)
        except ModuleNotFoundError as exc:
            # Optional third-party deps (mooncake, ...) may be absent on CPU CI; a missing miles module is the bug.
            if (exc.name or "").partition(".")[0] == "miles":
                raise
