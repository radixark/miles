"""Serve one miles-ext-kernels op from its own miles-ext-kernels commit.

The base checkout ``3rdparty/miles-ext-kernels`` provides the package ``miles_ext_kernels``. Its kernels live in the
namespace ``miles_ext_kernels.cake``: one module per op (``cake/<op>.py``), the op's implementation
(``cake/_impl/<op>/``), the vendored kernel packages (``cake/_vendor/<pkg>/``, each listed under its op in
``cake/_vendor/VENDORED.json``) and the runtime every op shares (``cake/_runtime``: arguments, variants, loader,
prebuild, workspace pool, release).

An op is pinned by a second checkout of the same repository at ``3rdparty/miles-ext-kernels-pins/<op>``, a git
submodule at another commit that ``.gitmodules`` registers. The finder that :func:`install` puts on ``sys.meta_path``
loads the modules the op owns from the pin: ``miles_ext_kernels.cake.<op>``, ``miles_ext_kernels.cake._impl.<op>``
and ``miles_ext_kernels.cake._vendor.<pkg>`` for every ``<pkg>`` the pin's ``VENDORED.json`` lists under the op, each
with all of its submodules. Every other module, the runtime included, still comes from the base, so a pinned op runs
against the base's runtime, which registers the op's arguments from the pinned ``cake/<op>.py``. The base runtime
resolves the alias ``miles_ext_kernels.cake._vendor.cake.<pkg>`` that vendored code imports to the module
``miles_ext_kernels.cake._vendor.<pkg>`` itself, so the alias lands in the pin too.

``MILES_EXT_KERNELS_PINS_DIR`` replaces the pins directory, for a miles that does not run from its source tree: every
checkout in it is a pin. Without pins the finder declines every module. ``miles/__init__.py`` installs it, so the hook
applies in every process that imports miles before an op of ``miles_ext_kernels`` (the driver, the Ray workers,
pytest through ``tests/conftest.py``); importing miles after a module of a pinned op has loaded from the base raises
``ImportError``.

Importing this module imports only the standard library.
"""

import ast
import functools
import importlib.abc
import importlib.util
import json
import logging
import os
import re
import socket
import sys
from pathlib import Path, PurePosixPath

logger = logging.getLogger(__name__)

PINS_DIR_ENV = "MILES_EXT_KERNELS_PINS_DIR"

_PACKAGE = "miles_ext_kernels"
_SOURCE = f"{_PACKAGE}.cake"
_IMPL = f"{_SOURCE}._impl"
_VENDOR = f"{_SOURCE}._vendor"
_MANIFEST = "cake/_vendor/VENDORED.json"  # relative to the package directory
_REGISTRY = "cake/_runtime/registry.py"  # its OPS lists the ops of the commit
_PINS_PATH = PurePosixPath("3rdparty/miles-ext-kernels-pins")
_SOURCE_TREE = Path(__file__).resolve().parents[2]
_GITMODULES_PATH = re.compile(r"^\s*path\s*=\s*(.+?)\s*$", re.MULTILINE)
_IGNORED: set[Path] = set()
_ANNOUNCED: set[tuple[str, Path]] = set()


def pins_dir() -> Path:
    """``$MILES_EXT_KERNELS_PINS_DIR``, else ``3rdparty/miles-ext-kernels-pins`` of the miles source tree.

    Raises ``ImportError`` when the variable is set to anything but the absolute path of a directory: a relative path
    resolves against the working directory of each process, and a directory missing on one node would silently serve
    the base there.
    """
    override = os.environ.get(PINS_DIR_ENV)
    if not override:
        return _SOURCE_TREE / _PINS_PATH
    path = Path(override)
    if not (path.is_absolute() and path.is_dir()):
        raise ImportError(
            f"{PINS_DIR_ENV}={override!r} is not the absolute path of a directory on {socket.gethostname()} "
            f"(working directory {os.getcwd()}): set it to a pins directory every node reaches at that path, or "
            "unset it"
        )
    return path


def find_pins(base_package_dir: Path) -> dict[str, Path]:
    """The pinned ops and their checkouts, checked against the base package directory ``base_package_dir``.

    In the miles source tree the pins are the submodules its ``.gitmodules`` registers under
    ``3rdparty/miles-ext-kernels-pins``, each named after its op; a directory there that ``.gitmodules`` does not
    register, as git leaves behind when a commit removes a submodule, is ignored with a warning. Under
    ``$MILES_EXT_KERNELS_PINS_DIR`` every entry but hidden ones is a pin.

    Raises ``ImportError`` for a pin named after no op of the base, for a pin without the op's module
    ``miles_ext_kernels/cake/<op>.py`` (a submodule not checked out, or a commit from before that layout) and for a
    ``$MILES_EXT_KERNELS_PINS_DIR`` that :func:`pins_dir` rejects.
    """
    root = pins_dir()
    if os.environ.get(PINS_DIR_ENV):
        names = _entries(root)
    else:
        names = _registered_pins(_SOURCE_TREE)
        for name in sorted(_entries(root) - names):
            _warn_ignored(root / name)
    ops = _ops(base_package_dir) if names else set()
    pins = {}
    for op in sorted(names):
        checkout = root / op
        if op not in ops:
            raise ImportError(
                f"{checkout}: {op!r} is not an op of {base_package_dir / 'cake'} "
                f"(ops: {', '.join(sorted(ops)) or 'none'}); name each pin after the op it serves"
            )
        if not (checkout / _PACKAGE / "cake" / f"{op}.py").is_file():
            raise ImportError(
                f"{checkout} pins the op {op!r} but has no {_PACKAGE}/cake/{op}.py: check out the submodule "
                f"('git submodule update --init {checkout}'), pin a commit with the {_SOURCE} layout or remove the "
                "pin"
            )
        pins[op] = checkout
    return pins


class _PinFinder(importlib.abc.MetaPathFinder):
    """Finds the modules a pinned op owns in its pin; declines every other module.

    Logs, in every process, the pin and commit a pinned op loads from, and warns when the pin's copy of the shared
    modules differs from the base's.
    """

    def find_spec(self, fullname, path, target=None):
        parent, _, name = fullname.rpartition(".")
        if parent not in (_SOURCE, _IMPL, _VENDOR) or not path:
            return None
        parent_dir = Path(next(iter(path)))  # ``path`` is the base parent package's ``__path__``
        base_package_dir = parent_dir.parent if parent == _SOURCE else parent_dir.parent.parent
        pins = find_pins(base_package_dir)
        op = _vendored_packages(pins).get(name) if parent == _VENDOR else name
        if op not in pins:
            return None
        checkout = pins[op]
        _announce(base_package_dir, op=op, checkout=checkout)
        cake_dir = checkout / _PACKAGE / "cake"
        if parent == _SOURCE:
            return importlib.util.spec_from_file_location(fullname, str(cake_dir / f"{op}.py"))
        package_dir = cake_dir / ("_impl" if parent == _IMPL else "_vendor") / name
        if not (package_dir / "__init__.py").is_file():
            missing = package_dir.relative_to(checkout) / "__init__.py"
            raise ImportError(f"{checkout} pins the op {op!r} but has no {missing}")
        return importlib.util.spec_from_file_location(
            fullname, str(package_dir / "__init__.py"), submodule_search_locations=[str(package_dir)]
        )


_FINDER = _PinFinder()


def install() -> None:
    """Put the pin finder first on ``sys.meta_path``, once.

    Raises ``ImportError`` when a module of a pinned op was imported earlier from elsewhere than its pin, which the
    finder can no longer change.
    """
    if _FINDER in sys.meta_path:
        return
    if (base_package_dir := _loaded_package_dir()) is not None:
        _loaded_pins(base_package_dir)
    sys.meta_path.insert(0, _FINDER)


def log_miles_ext_kernels_pins() -> None:
    """Log the base checkout and commit of the loaded miles-ext-kernels and those of every pinned op.

    Raises ``ImportError`` when a module of a pinned op has loaded from elsewhere than its pin.
    """
    base_package_dir = _loaded_package_dir()
    if base_package_dir is None:
        return
    pins = _loaded_pins(base_package_dir)
    sources = [f"base {base_package_dir.parent} at {_commit_label(base_package_dir.parent)}"]
    sources += [f"{op} pinned to {checkout} at {_commit_label(checkout)}" for op, checkout in pins.items()]
    logger.info(f"miles-ext-kernels: {'; '.join(sources)}")


def _loaded_package_dir() -> Path | None:
    """The directory of the loaded ``miles_ext_kernels`` package; None when none is loaded."""
    module = sys.modules.get(_PACKAGE)
    path = getattr(module, "__path__", None)  # config-access-exempt: tests put stubs without __path__ there
    return Path(next(iter(path))) if path else None


def _loaded_pins(base_package_dir: Path) -> dict[str, Path]:
    """The pins of the loaded ``miles_ext_kernels``, after checking that every module of a pinned op loaded so far came
    from its pin."""
    pins = find_pins(base_package_dir)
    owned = {op: [f"{_SOURCE}.{op}", f"{_IMPL}.{op}"] for op in pins}
    for package, op in _vendored_packages(pins).items():
        owned[op].append(f"{_VENDOR}.{package}")
    for op, checkout in pins.items():
        for name in owned[op]:
            module = sys.modules.get(name)
            file = getattr(module, "__file__", None)  # config-access-exempt: stubs and namespaces have no __file__
            if module is not None and not (file and Path(file).resolve().is_relative_to(checkout.resolve())):
                raise ImportError(
                    f"{name} is pinned to {checkout} but loaded from {file}: it was imported before miles installed "
                    f"the hook that serves pins; import miles before the ops of {_PACKAGE}"
                )
    return pins


def _vendored_packages(pins: dict[str, Path]) -> dict[str, str]:
    """``{package: op}`` for the vendored packages the pins serve: those each pin's own ``VENDORED.json`` lists under
    its op.

    Raises ``ImportError`` when a pin has no readable manifest and when two pins list the same package.
    """
    owners: dict[str, str] = {}
    for op, checkout in pins.items():
        for package in _manifest(checkout / _PACKAGE).get(op, ()):
            if package in owners:
                raise ImportError(
                    f"the pins of {owners[package]!r} and {op!r} both list the vendored package {package!r} in their "
                    f"{_MANIFEST}; pin commits where it belongs to one op only"
                )
            owners[package] = op
    return owners


def _manifest(package_dir: Path) -> dict[str, tuple[str, ...]]:
    """``{op: packages}`` from the ``VENDORED.json`` of a ``miles_ext_kernels`` directory.

    Raises ``ImportError`` when the file is missing or unreadable.
    """
    path = package_dir / _MANIFEST
    try:
        stat = path.stat()
        return _read_manifest(path, stat.st_mtime_ns, stat.st_size)
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        raise ImportError(f"{path} is not a readable manifest of vendored packages: {error!r}") from error


@functools.cache
def _read_manifest(path: Path, mtime_ns: int, size: int) -> dict[str, tuple[str, ...]]:
    """The packages ``VENDORED.json`` lists under each op, parsed once per version of the file."""
    return {op: tuple(packages) for op, packages in json.loads(path.read_text())["ops"].items()}


def _announce(base_package_dir: Path, *, op: str, checkout: Path) -> None:
    """Log the pin of ``op`` and warn about differing shared modules, once per process."""
    if (op, checkout) in _ANNOUNCED:
        return
    _ANNOUNCED.add((op, checkout))
    logger.info(f"miles-ext-kernels: {op} loads from its pin {checkout} at {_commit_label(checkout)}")
    differences = _shared_module_differences(base_package_dir, checkout / _PACKAGE)
    if differences:
        logger.warning(
            f"miles-ext-kernels: the {op} pin {checkout} has its own copy of the shared modules, which differs "
            f"from the base's in {', '.join(differences)}; the pinned {op} runs against the base's copy, so "
            "an import or argument error that follows may come from this difference"
        )


def _warn_ignored(checkout: Path) -> None:
    if checkout in _IGNORED:
        return
    _IGNORED.add(checkout)
    logger.warning(
        f"miles-ext-kernels: ignoring {checkout}: .gitmodules registers no submodule there, so it pins nothing at "
        f"this commit (git leaves the directory of a removed submodule behind); delete it with 'rm -rf {checkout}'"
    )


def _entries(root: Path) -> set[str]:
    """The names of the entries of ``root`` but hidden ones; none when ``root`` is no directory."""
    return {entry.name for entry in root.iterdir() if not entry.name.startswith(".")} if root.is_dir() else set()


def _registered_pins(source_tree: Path) -> set[str]:
    """The names of the submodules the ``.gitmodules`` of ``source_tree`` registers under the pins directory."""
    try:
        text = (source_tree / ".gitmodules").read_text()
    except FileNotFoundError:
        return set()
    paths = (PurePosixPath(match.strip('"')) for match in _GITMODULES_PATH.findall(text))
    return {path.name for path in paths if path.parent == _PINS_PATH}


def _ops(package_dir: Path) -> set[str]:
    """The ops of a ``miles_ext_kernels`` directory: its modules ``cake/<op>.py`` whose names do not start with _."""
    return {path.stem for path in (package_dir / "cake").glob("*.py") if not path.name.startswith("_")}


def _shared_module_differences(base_package_dir: Path, pin_package_dir: Path) -> list[str]:
    """The shared files of the base, relative to the package, whose copy in the pin differs or is missing."""
    return sorted(
        name
        for name in _shared_files(base_package_dir)
        if _comparable(name, base_package_dir / name) != _comparable(name, pin_package_dir / name)
    )


def _shared_files(package_dir: Path) -> set[str]:
    """The files of ``package_dir`` that every op shares: all but those of the modules the ops own (``cake/<op>.py``,
    ``cake/_impl/<op>/``, ``cake/_vendor/<pkg>/``), the manifest ``VENDORED.json`` and ``__pycache__``."""
    ops = _ops(package_dir)
    packages = {package for listed in _manifest(package_dir).values() for package in listed}
    owned_dirs = {f"cake/_impl/{op}" for op in ops} | {f"cake/_vendor/{package}" for package in packages}
    owned_files = {f"cake/{op}.py" for op in ops} | {_MANIFEST}
    names = set()
    for directory, subdirectories, files in os.walk(package_dir):
        relative = PurePosixPath(Path(directory).relative_to(package_dir).as_posix())
        subdirectories[:] = [
            name for name in subdirectories if name != "__pycache__" and str(relative / name) not in owned_dirs
        ]
        names.update(str(relative / name) for name in files if str(relative / name) not in owned_files)
    return names


def _comparable(name: str, path: Path) -> bytes | None:
    """The bytes of a shared file as compared with the pin's copy; for the registry, without its ``OPS`` assignment:
    each layer of the stack appends its op there, and a pinned op does not depend on the list."""
    data = path.read_bytes() if path.is_file() else None
    if data is None or name != _REGISTRY:
        return data
    try:
        tree = ast.parse(data)
    except (SyntaxError, ValueError):
        return data
    lines = data.splitlines(keepends=True)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        else:
            continue
        if any(isinstance(target, ast.Name) and target.id == "OPS" for target in targets):
            del lines[node.lineno - 1 : node.end_lineno]
            break
    return b"".join(lines)


def _commit_label(checkout: Path) -> str:
    return _checkout_commit(checkout) or "an unknown commit"


def _checkout_commit(checkout: Path) -> str | None:
    """The commit checked out at ``checkout``, read from its git metadata without running git; None when unknown."""
    git_dir = checkout / ".git"
    try:
        if git_dir.is_file():  # a submodule or worktree checkout: "gitdir: <path>", relative or absolute
            git_dir = checkout / git_dir.read_text().removeprefix("gitdir:").strip()
        head = (git_dir / "HEAD").read_text().strip()
        if not head.startswith("ref: "):
            return head  # detached, as a submodule checkout is
        ref = head.removeprefix("ref: ")
        # a linked worktree keeps HEAD in its own git directory and the branches in the common one
        common_dir = git_dir
        if (git_dir / "commondir").is_file():
            common_dir = git_dir / (git_dir / "commondir").read_text().strip()
        for directory in (git_dir, common_dir):
            if (directory / ref).is_file():
                return (directory / ref).read_text().strip()
        for line in (common_dir / "packed-refs").read_text().splitlines():
            commit, _, name = line.partition(" ")
            if name == ref:
                return commit
    except OSError:
        return None
    return None
