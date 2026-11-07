import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from miles.utils import ext_kernel_pins
from miles.utils.ext_kernel_pins import PINS_DIR_ENV, find_pins, pins_dir

REPO_ROOT = Path(__file__).resolve().parents[3]
PIN_COMMIT = "0123456789abcdef0123456789abcdef01234567"
# the vendored packages of each op; cake_kda_train_kcp imports its sibling through the alias ..cake.cake_kda_train
PACKAGES = {"gdn": ("cake_gdn_train",), "kda": ("cake_kda_train", "cake_kda_train_kcp")}

# The base runtime's alias finder, as in miles_ext_kernels.cake._runtime.loader: _vendor.cake.<pkg> is the module
# _vendor.<pkg> itself, found through whatever finder serves it. First on sys.meta_path, ahead of the pin finder.
_ALIAS_FINDER = """\
import sys
from importlib import import_module
from importlib.machinery import ModuleSpec

VENDOR = "miles_ext_kernels.cake._vendor"
ALIAS = VENDOR + ".cake"


class VendorAlias:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == ALIAS or fullname.startswith(ALIAS + "."):
            return ModuleSpec(fullname, self, is_package=fullname == ALIAS)
        return None

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        if module.__name__ != ALIAS:
            sys.modules[module.__name__] = import_module(VENDOR + module.__name__[len(ALIAS) :])


sys.meta_path.insert(0, VendorAlias())
"""

_REGISTRY = """\
# the registered ops
from importlib import import_module

{ops}
VERSION = {shared}


def op_spec(op):
    return import_module(f"miles_ext_kernels.cake.{{op}}").OP
"""

_IMPL_INIT = """\
from miles_ext_kernels.cake._runtime import loader, registry

OP = registry.op_spec("{op}")
"""

# Runs in a fresh interpreter, so the finder that ``import miles`` installs and the modules it loads stay out of the
# test process. argv[1] is a JSON config: the miles source tree and the base checkout to put on sys.path, the modules
# to import before miles, the source tree the finder reads ``.gitmodules`` from (None keeps miles' own) and the modules
# to import after miles. Prints the file of every loaded miles_ext_kernels module and logs to stderr.
_PROBE = """
import importlib, json, logging, sys
from pathlib import Path
config = json.loads(sys.argv[1])
sys.path[:0] = [config["repo"], config["base"]]
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
for name in config["before_miles"]:
    importlib.import_module(name)
from miles.utils import ext_kernel_pins
if config["tree"]:
    ext_kernel_pins._SOURCE_TREE = Path(config["tree"])
for name in config["modules"]:
    importlib.import_module(name)
ext_kernel_pins.log_miles_ext_kernels_pins()
print(json.dumps({n: getattr(m, "__file__", None) for n, m in sys.modules.items() if n.startswith("miles_ext")}))
"""


def _write(path: Path, text: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _ops_tuple(ops: tuple[str, ...]) -> str:
    """``OPS`` as black writes it: one line for a single op, one op per line otherwise."""
    if len(ops) == 1:
        return f'OPS: tuple[str, ...] = ("{ops[0]}",)\n'
    return "OPS: tuple[str, ...] = (\n" + "".join(f'    "{op}",\n' for op in ops) + ")\n"


def _checkout(root: Path, *, shared: str = "1", ops: tuple[str, ...] = ("gdn", "kda"), packages=None) -> Path:
    """A miles-ext-kernels checkout laid out like the real one (``miles_ext_kernels.cake``) with ``ops``.

    ``shared`` versions the runtime every op shares; ``packages`` replaces the vendored packages of some ops.
    """
    packages = {**PACKAGES, **(packages or {})}
    cake = root / "miles_ext_kernels" / "cake"
    _write(cake.parent / "__init__.py", 'SOURCES = ("cake",)\n')
    _write(cake / "__init__.py", "from ._runtime import OPS\n")
    _write(cake / "_runtime" / "__init__.py", f"from .registry import OPS\n\nVERSION = {shared}\n")
    _write(cake / "_runtime" / "registry.py", _REGISTRY.format(ops=_ops_tuple(ops), shared=shared))
    _write(cake / "_runtime" / "loader.py", f"{_ALIAS_FINDER}\nVERSION = {shared}\n")
    _write(cake / "_impl" / "__init__.py")
    _write(cake / "_vendor" / "__init__.py")
    manifest = {op: {package: {} for package in packages[op]} for op in ops}
    _write(cake / "_vendor" / "VENDORED.json", json.dumps({"ops": manifest, "schema": "1"}, indent=2) + "\n")
    for op in ops:
        _write(cake / f"{op}.py", f'from miles_ext_kernels.cake._runtime import registry\n\nOP = "{op}"\n')
        # as the real _impl: the runtime imports the op module by name (loader.install_op -> registry.op_spec)
        _write(cake / "_impl" / op / "__init__.py", _IMPL_INIT.format(op=op))
        imports = "".join(f"from miles_ext_kernels.cake._vendor import {package}\n" for package in packages[op])
        _write(cake / "_impl" / op / "ops.py", imports)
        for package in packages[op]:
            sibling = package.removesuffix("_kcp")
            if sibling != package:
                _write(cake / "_vendor" / package / "__init__.py", f"from ..cake.{sibling} import kernel\n")
            else:
                _write(cake / "_vendor" / package / "__init__.py")
                _write(cake / "_vendor" / package / "kernel.py")
    return root


def _source_tree(root: Path, *registered: str) -> Path:
    """A miles source tree whose ``.gitmodules`` registers the base submodule and a pin per ``registered`` op."""
    paths = ["3rdparty/miles-ext-kernels", *(f"3rdparty/miles-ext-kernels-pins/{op}" for op in registered)]
    _write(root / ".gitmodules", "".join(f'[submodule "{path}"]\n\tpath = {path}\n\turl = <url>\n' for path in paths))
    return root


def _submodule(checkout: Path, *, commit: str) -> None:
    """Give ``checkout`` the git metadata of a submodule checked out at ``commit``."""
    git_dir = checkout.parent / ".modules" / checkout.name
    _write(git_dir / "HEAD", f"{commit}\n")
    _write(checkout / ".git", f"gitdir: ../.modules/{checkout.name}\n")


def _probe(
    base: Path,
    *modules: str,
    pins: Path | str | None = None,
    tree: Path | None = None,
    before_miles: tuple[str, ...] = (),
) -> subprocess.CompletedProcess:
    """Import ``modules`` with ``pins`` as ``$MILES_EXT_KERNELS_PINS_DIR``, or unset to read the pins of ``tree``."""
    config = {
        "repo": str(REPO_ROOT),
        "base": str(base),
        "before_miles": before_miles,
        "tree": tree and str(tree),
        "modules": modules,
    }
    env = {name: value for name, value in os.environ.items() if name != PINS_DIR_ENV}
    if pins is not None:
        env[PINS_DIR_ENV] = str(pins)
    return subprocess.run(
        [sys.executable, "-c", _PROBE, json.dumps(config)],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def _loaded_from(base: Path, *modules: str, **kwargs) -> tuple[dict[str, str | None], str]:
    completed = _probe(base, *modules, **kwargs)
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout.strip().splitlines()[-1]), completed.stderr


def _from(loaded: dict[str, str | None], checkout: Path) -> set[str]:
    """The loaded modules whose file is in ``checkout``."""
    return {name for name, file in loaded.items() if file and Path(file).is_relative_to(checkout)}


def _warnings(stderr: str) -> list[str]:
    return [line for line in stderr.splitlines() if line.startswith("WARNING")]


KDA_MODULES = {
    "miles_ext_kernels.cake.kda",
    "miles_ext_kernels.cake._impl.kda",
    "miles_ext_kernels.cake._impl.kda.ops",
    "miles_ext_kernels.cake._vendor.cake_kda_train",
    "miles_ext_kernels.cake._vendor.cake_kda_train.kernel",
    "miles_ext_kernels.cake._vendor.cake_kda_train_kcp",
    # the alias the kcp package imports its sibling through is the sibling itself; ``from <alias> import kernel``
    # then imports the submodule under the sibling's own name
    "miles_ext_kernels.cake._vendor.cake.cake_kda_train",
}


class TestPinnedOpImports:
    def test_a_pinned_op_and_its_implementation_and_vendored_packages_load_from_the_pin(self, tmp_path: Path) -> None:
        """Only the modules the op owns move: the top-level package, the runtime and the other ops stay on the base,
        so the pin's own copy of the runtime is never imported."""
        base = _checkout(tmp_path / "base")
        pin = _checkout(tmp_path / "pins" / "kda")

        loaded, stderr = _loaded_from(
            base, "miles_ext_kernels.cake.gdn", "miles_ext_kernels.cake._impl.kda.ops", pins=tmp_path / "pins"
        )

        assert _from(loaded, pin) == KDA_MODULES
        assert _from(loaded, base) == {name for name, file in loaded.items() if file} - KDA_MODULES
        on_base = {
            "miles_ext_kernels",
            "miles_ext_kernels.cake",
            "miles_ext_kernels.cake._runtime",
            "miles_ext_kernels.cake._runtime.loader",
            "miles_ext_kernels.cake._runtime.registry",
            "miles_ext_kernels.cake._impl",
            "miles_ext_kernels.cake._vendor",
            "miles_ext_kernels.cake.gdn",
        }
        assert on_base <= _from(loaded, base)
        # every process that loads the pin says so, the Ray workers included
        assert f"INFO miles-ext-kernels: kda loads from its pin {pin} at an unknown commit" in stderr.splitlines()
        assert not _warnings(stderr)

    def test_without_pins_everything_loads_from_the_base(self, tmp_path: Path) -> None:
        base = _checkout(tmp_path / "base")

        loaded, stderr = _loaded_from(
            base, "miles_ext_kernels.cake._impl.kda.ops", tree=_source_tree(tmp_path / "miles")
        )

        assert KDA_MODULES <= loaded.keys()
        assert all(Path(file).is_relative_to(base) for file in loaded.values() if file)
        assert not _warnings(stderr)

    def test_the_pin_serves_the_vendored_packages_its_own_manifest_lists(self, tmp_path: Path) -> None:
        """A newer commit renamed a package of kda: the pin serves its own names, aliases included, and the base's
        names of the op load from the base, as nothing in the pinned op imports them."""
        renamed = ("cake_kda_v2_train", "cake_kda_v2_train_kcp")
        base = _checkout(tmp_path / "base")
        pin = _checkout(tmp_path / "pins" / "kda", packages={"kda": renamed})

        loaded, _ = _loaded_from(
            base,
            "miles_ext_kernels.cake._impl.kda.ops",
            "miles_ext_kernels.cake._vendor.cake_kda_train",
            pins=tmp_path / "pins",
        )

        assert "miles_ext_kernels.cake._vendor.cake.cake_kda_v2_train" in _from(loaded, pin)
        assert "miles_ext_kernels.cake._vendor.cake_kda_v2_train_kcp" in _from(loaded, pin)
        assert "miles_ext_kernels.cake._vendor.cake_kda_train" in _from(loaded, base)

    @pytest.mark.parametrize("checkout", ["empty", "before-the-cake-layout"])
    def test_a_pin_without_the_op_module_fails_the_import(self, tmp_path: Path, checkout: str) -> None:
        """A submodule registered but not checked out leaves an empty directory, and a commit from before the
        ``miles_ext_kernels.cake`` layout has no op module; the base copy must not stand in."""
        base = _checkout(tmp_path / "base")
        _write(tmp_path / "pins" / "kda" / ("README.md" if checkout == "empty" else "miles_ext_kernels/kda/ops.py"))

        completed = _probe(base, "miles_ext_kernels.cake.gdn", pins=tmp_path / "pins")

        assert completed.returncode != 0
        assert "has no miles_ext_kernels/cake/kda.py" in completed.stderr
        assert "git submodule update --init" in completed.stderr

    def test_a_pin_without_a_readable_manifest_fails_the_import_of_its_vendored_packages(self, tmp_path: Path) -> None:
        base = _checkout(tmp_path / "base")
        pin = _checkout(tmp_path / "pins" / "kda")
        (pin / "miles_ext_kernels" / "cake" / "_vendor" / "VENDORED.json").write_text("{")

        completed = _probe(base, "miles_ext_kernels.cake._impl.kda.ops", pins=tmp_path / "pins")

        assert completed.returncode != 0
        assert "VENDORED.json is not a readable manifest of vendored packages" in completed.stderr

    def test_two_pins_listing_one_vendored_package_fail_the_import(self, tmp_path: Path) -> None:
        base = _checkout(tmp_path / "base")
        _checkout(tmp_path / "pins" / "gdn", packages={"gdn": ("cake_gdn_train", "cake_kda_train")})
        _checkout(tmp_path / "pins" / "kda")

        completed = _probe(base, "miles_ext_kernels.cake._impl.kda.ops", pins=tmp_path / "pins")

        assert completed.returncode != 0
        assert "the pins of 'gdn' and 'kda' both list the vendored package 'cake_kda_train'" in completed.stderr

    def test_an_unknown_op_fails_the_import(self, tmp_path: Path) -> None:
        base = _checkout(tmp_path / "base")
        _checkout(tmp_path / "pins" / "kdb")

        completed = _probe(base, "miles_ext_kernels.cake.gdn", pins=tmp_path / "pins")

        assert completed.returncode != 0
        assert "'kdb' is not an op of" in completed.stderr
        assert "ops: gdn, kda" in completed.stderr

    def test_importing_miles_after_a_pinned_op_loaded_from_the_base_fails(self, tmp_path: Path) -> None:
        """A script that imports an op before miles has it from the base before the hook exists; it must fail rather
        than run the base where the pin was meant to run. The top-level package and the runtime may come first, and
        without pins any order is fine."""
        base = _checkout(tmp_path / "base")
        (tmp_path / "pins").mkdir()

        _loaded_from(base, pins=tmp_path / "pins", before_miles=("miles_ext_kernels.cake.kda",))

        pin = _checkout(tmp_path / "pins" / "kda")
        early = ("miles_ext_kernels", "miles_ext_kernels.cake", "miles_ext_kernels.cake._runtime.loader")
        loaded, _ = _loaded_from(
            base, "miles_ext_kernels.cake._impl.kda.ops", pins=tmp_path / "pins", before_miles=early
        )
        assert _from(loaded, pin) == KDA_MODULES

        completed = _probe(base, pins=tmp_path / "pins", before_miles=("miles_ext_kernels.cake.kda",))

        assert completed.returncode != 0
        assert f"miles_ext_kernels.cake.kda is pinned to {pin} but loaded from {base}" in completed.stderr
        assert "import miles before the ops of miles_ext_kernels" in completed.stderr


class TestSourceTreePins:
    def test_gitmodules_decides_which_checkouts_are_pins(self, tmp_path: Path) -> None:
        """A checkout git left behind after a commit removed its submodule (kda) is ignored, with a warning."""
        base = _checkout(tmp_path / "base")
        tree = _source_tree(tmp_path / "miles", "gdn")
        pins = tree / "3rdparty" / "miles-ext-kernels-pins"
        _checkout(pins / "gdn")
        _checkout(pins / "kda")

        loaded, stderr = _loaded_from(base, "miles_ext_kernels.cake.gdn", "miles_ext_kernels.cake.kda", tree=tree)

        assert Path(loaded["miles_ext_kernels.cake.gdn"]).is_relative_to(pins / "gdn")
        assert Path(loaded["miles_ext_kernels.cake.kda"]).is_relative_to(base)
        warnings = _warnings(stderr)
        assert len(warnings) == 1
        assert f"ignoring {pins / 'kda'}: .gitmodules registers no submodule there" in warnings[0]
        assert f"rm -rf {pins / 'kda'}" in warnings[0]

    def test_a_registered_pin_that_is_not_checked_out_fails_the_import(self, tmp_path: Path) -> None:
        base = _checkout(tmp_path / "base")

        completed = _probe(base, "miles_ext_kernels.cake.kda", tree=_source_tree(tmp_path / "miles", "gdn"))

        assert completed.returncode != 0
        assert "has no miles_ext_kernels/cake/gdn.py" in completed.stderr


class TestPinsDirectoryOverride:
    @pytest.mark.parametrize("override", ["pins", "absent"])
    def test_a_relative_or_missing_override_fails_the_import(self, tmp_path: Path, override: str) -> None:
        """A relative path resolves against each process's working directory, and a directory missing on one node
        would quietly serve the base there."""
        base = _checkout(tmp_path / "base")
        _checkout(tmp_path / "pins" / "gdn")
        value = override if override == "pins" else str(tmp_path / override)

        completed = _probe(base, "miles_ext_kernels.cake.gdn", pins=value)

        assert completed.returncode != 0
        assert f"{PINS_DIR_ENV}={value!r} is not the absolute path of a directory" in completed.stderr

    def test_the_pins_directory_defaults_to_the_source_tree_and_follows_the_override(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(PINS_DIR_ENV, raising=False)
        assert pins_dir() == REPO_ROOT / "3rdparty" / "miles-ext-kernels-pins"

        monkeypatch.setenv(PINS_DIR_ENV, str(tmp_path))
        assert pins_dir() == tmp_path

    def test_hidden_entries_of_the_override_are_not_pins(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        base = _checkout(tmp_path / "base")
        _checkout(tmp_path / "pins" / "kda")
        _write(tmp_path / "pins" / ".gitkeep")
        monkeypatch.setenv(PINS_DIR_ENV, str(tmp_path / "pins"))

        assert find_pins(base / "miles_ext_kernels") == {"kda": tmp_path / "pins" / "kda"}


class TestPinDiagnostics:
    def test_the_report_names_each_checkout_and_its_commit(self, tmp_path: Path) -> None:
        base = _checkout(tmp_path / "base")
        pin = _checkout(tmp_path / "pins" / "kda")
        _submodule(pin, commit=PIN_COMMIT)

        _, stderr = _loaded_from(base, "miles_ext_kernels.cake.kda", pins=tmp_path / "pins")

        assert f"kda loads from its pin {pin} at {PIN_COMMIT}" in stderr
        assert f"base {base} at an unknown commit; kda pinned to {pin} at {PIN_COMMIT}" in stderr
        assert not _warnings(stderr)

    def test_differing_shared_modules_are_named_in_a_warning(self, tmp_path: Path) -> None:
        """The pinned op imports the base's runtime; the warning comes when the op's first module loads, once."""
        base = _checkout(tmp_path / "base")
        _write(base / "miles_ext_kernels" / "cake" / "_runtime" / "extra.py")
        pin = _checkout(tmp_path / "pins" / "kda", shared="2")

        _, stderr = _loaded_from(base, "miles_ext_kernels.cake._impl.kda.ops", pins=tmp_path / "pins")

        lines = stderr.splitlines()
        warnings = _warnings(stderr)
        assert len(warnings) == 1
        shared = ", ".join(f"cake/_runtime/{name}.py" for name in ("__init__", "extra", "loader", "registry"))
        assert f"the kda pin {pin} has its own copy of the shared modules" in warnings[0]
        assert f"which differs from the base's in {shared};" in warnings[0]
        assert "the pinned kda runs against the base's copy" in warnings[0]
        report = next(index for index, line in enumerate(lines) if line.startswith("INFO miles-ext-kernels: base"))
        assert lines.index(warnings[0]) < report  # at import, before the startup report

    def test_what_each_layer_of_the_stack_adds_is_not_a_difference(self, tmp_path: Path) -> None:
        """A pin at an earlier commit of the stack lacks the ops added after it: their modules and packages, their
        entries in the manifest and in OPS (written on one line for a single op) are not shared modules. Ops the pin
        has and the base lacks are not either."""
        base = _checkout(tmp_path / "base", ops=("gdn", "kda", "conv1d"), packages={"conv1d": ("cake_conv1d",)})
        pin = _checkout(tmp_path / "pins" / "kda", ops=("kda", "mok"), packages={"mok": ("cake_mok",)})

        loaded, stderr = _loaded_from(base, "miles_ext_kernels.cake._impl.kda.ops", pins=tmp_path / "pins")

        assert _from(loaded, pin) == KDA_MODULES
        assert not _warnings(stderr)


class TestInstall:
    def test_install_is_idempotent(self) -> None:
        """``import miles`` installed the finder already; installing it again must not stack a second one."""
        ext_kernel_pins.install()
        ext_kernel_pins.install()

        assert [type(finder).__module__ for finder in sys.meta_path].count(ext_kernel_pins.__name__) == 1


class TestCheckoutCommit:
    def test_a_branch_checkout_reads_its_loose_or_packed_ref(self, tmp_path: Path) -> None:
        git_dir = tmp_path / "checkout" / ".git"
        _write(git_dir / "HEAD", "ref: refs/heads/main\n")
        _write(git_dir / "packed-refs", f"# pack-refs with: peeled\n{PIN_COMMIT} refs/heads/main\n")
        assert ext_kernel_pins._checkout_commit(tmp_path / "checkout") == PIN_COMMIT

        _write(git_dir / "refs" / "heads" / "main", "0" * 40 + "\n")
        assert ext_kernel_pins._checkout_commit(tmp_path / "checkout") == "0" * 40

    def test_a_linked_worktree_reads_its_branch_from_the_common_git_directory(self, tmp_path: Path) -> None:
        """``git worktree add`` keeps HEAD in ``<common>/worktrees/<name>`` and the branches in ``<common>``."""
        common = tmp_path / "clone" / ".git"
        worktree_git_dir = common / "worktrees" / "gdn"
        _write(worktree_git_dir / "HEAD", "ref: refs/heads/pin\n")
        _write(worktree_git_dir / "commondir", "../..\n")
        _write(common / "packed-refs", f"{PIN_COMMIT} refs/heads/pin\n")
        _write(tmp_path / "gdn" / ".git", f"gitdir: {worktree_git_dir}\n")
        assert ext_kernel_pins._checkout_commit(tmp_path / "gdn") == PIN_COMMIT

        _write(common / "refs" / "heads" / "pin", "0" * 40 + "\n")
        assert ext_kernel_pins._checkout_commit(tmp_path / "gdn") == "0" * 40

    def test_a_checkout_without_git_metadata_has_no_commit(self, tmp_path: Path) -> None:
        assert ext_kernel_pins._checkout_commit(tmp_path) is None
