from __future__ import annotations

import ast
from collections.abc import Iterable
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FRAMEWORK_ROOT = REPO_ROOT / "miles"

SHIPPED_PACKAGE_ROOTS = ("miles", "miles_plugins", "scripts", "examples", "tools")


def framework_modules(*, exclude_dirs: Iterable[Path] = ()) -> list[Path]:
    return python_modules(roots=[FRAMEWORK_ROOT], exclude_dirs=exclude_dirs)


def shipped_modules(*, exclude_dirs: Iterable[Path] = ()) -> list[Path]:
    packaged = python_modules(roots=[REPO_ROOT / name for name in SHIPPED_PACKAGE_ROOTS], exclude_dirs=exclude_dirs)
    excluded = tuple(exclude_dirs)
    scripts = [
        path for path in REPO_ROOT.glob("*.py") if not any(path.is_relative_to(directory) for directory in excluded)
    ]
    return sorted([*packaged, *scripts])


def python_modules(*, roots: Iterable[Path], exclude_dirs: Iterable[Path] = ()) -> list[Path]:
    excluded = tuple(exclude_dirs)
    return sorted(
        path
        for root in roots
        if root.is_dir()
        for path in root.rglob("*.py")
        if "__pycache__" not in path.parts and not any(path.is_relative_to(directory) for directory in excluded)
    )


def relative_paths(paths: Iterable[Path]) -> list[str]:
    return sorted(str(path.relative_to(REPO_ROOT)) for path in paths)


def parse_module(path: Path) -> ast.Module:
    return ast.parse(path.read_text(), filename=str(path))


def imported_modules(path: Path) -> set[str]:
    return imported_modules_of_source(path.read_text(), filename=str(path))


def imported_modules_of_source(source: str, *, filename: str = "<source>") -> set[str]:
    imported: set[str] = set()
    for node in ast.walk(ast.parse(source, filename=filename)):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            imported.add(node.module)
        elif isinstance(node, ast.Call) and _is_dynamic_import(node.func):
            arguments = [*node.args, *(keyword.value for keyword in node.keywords)]
            imported.update(
                argument.value
                for argument in arguments
                if isinstance(argument, ast.Constant) and isinstance(argument.value, str)
            )
    return imported


def imports_package(modules: Iterable[str], package: str) -> bool:
    return any(module == package or module.startswith(f"{package}.") for module in modules)


_DYNAMIC_IMPORT_NAMES = ("import_module", "__import__")


def _is_dynamic_import(func: ast.expr) -> bool:
    if isinstance(func, ast.Name):
        return func.id in _DYNAMIC_IMPORT_NAMES
    return isinstance(func, ast.Attribute) and func.attr in _DYNAMIC_IMPORT_NAMES
