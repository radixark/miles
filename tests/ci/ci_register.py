import ast
import glob
import warnings
from dataclasses import dataclass, field
from enum import Enum, auto

from tests.ci.labels import KNOWN_LABELS

__all__ = [
    "HWBackend",
    "CIRegistry",
    "collect_tests",
    "discover_ci_files",
    "register_cpu_ci",
    "register_cuda_ci",
    "ut_parse_one_file",
]

# Only these two parameters may be passed positionally; everything else
# (labels, always_on, nightly, disabled) is keyword-only.
_POSITIONAL_PARAMS = ("est_time", "suite")

# All accepted keyword arguments (in addition to the positional pair above).
_VALID_KWARGS = frozenset({"est_time", "suite", "labels", "nightly", "disabled"})

_REGISTER_NAMES = frozenset({"register_cpu_ci", "register_cuda_ci"})

_UNSET = object()


class HWBackend(Enum):
    CPU = auto()
    CUDA = auto()


@dataclass
class CIRegistry:
    backend: HWBackend
    filename: str
    est_time: float
    suite: str
    labels: list[str] = field(default_factory=list)
    nightly: bool = False
    disabled: str | None = None  # None = enabled, string = disabled reason
    # True only when collect_tests synthesized this entry by directory
    # convention (currently for tests/fast/ files that declare no register
    # call); False for every entry parsed from a register_*_ci() call.
    implicit: bool = False


def register_cpu_ci(
    est_time: float,
    suite: str,
    *,
    labels: list[str] | None = None,
    nightly: bool = False,
    disabled: str | None = None,
):
    """Marker for CPU CI registration (parsed via AST; runtime no-op).

    `labels=None` and `labels=[]` are equivalent: the test runs on every PR
    regardless of `run-ci-*` labels. A non-empty `labels` list gates the test
    on PR labels — the test runs when the PR carries `run-ci-<x>` for any
    `<x>` in `labels`.
    """
    return None


def register_cuda_ci(
    est_time: float,
    suite: str,
    *,
    labels: list[str] | None = None,
    nightly: bool = False,
    disabled: str | None = None,
):
    """Marker for CUDA CI registration (parsed via AST; runtime no-op).

    See `register_cpu_ci` for label semantics.
    """
    return None


_REGISTER_BACKEND_MAP = {
    "register_cpu_ci": HWBackend.CPU,
    "register_cuda_ci": HWBackend.CUDA,
}


def _extract_constant(node: ast.AST) -> object:
    """Return the literal value of an ast.Constant; otherwise return _UNSET.

    Sentinel return (instead of raising) lets callers compose richer error
    messages with parameter names and file paths.
    """
    if isinstance(node, ast.Constant):
        return node.value
    return _UNSET


def _extract_list_constant(node: ast.AST, *, context: str = "value") -> list:
    """Return a list of literal string constants from `ast.List`.

    Accepts `None` (as `ast.Constant(None)`) and treats it as an empty list,
    so callers may write `labels=None` interchangeably with `labels=[]`.

    Raises ValueError when the node is neither a list literal of string
    constants nor a literal `None`.
    """
    if isinstance(node, ast.Constant) and node.value is None:
        return []
    if not isinstance(node, ast.List):
        raise ValueError(f"{context} must be a list of string literals or None (got {type(node).__name__})")
    out: list = []
    for elt in node.elts:
        v = _extract_constant(elt)
        if v is _UNSET:
            raise ValueError(f"{context} must be a list of string literals (non-literal element)")
        if not isinstance(v, str):
            raise ValueError(f"{context} must be a list of string literals (got {type(v).__name__} element)")
        out.append(v)
    return out


class RegistryVisitor(ast.NodeVisitor):
    def __init__(self, filename: str):
        self.filename = filename
        self.registries: list[CIRegistry] = []

    def _parse_call_args(self, func_call: ast.Call, func_name: str) -> CIRegistry:
        if any(isinstance(arg, ast.Starred) for arg in func_call.args):
            raise ValueError(f"{self.filename}: starred arguments are not supported in {func_name}()")

        if len(func_call.args) > len(_POSITIONAL_PARAMS):
            raise ValueError(
                f"{self.filename}: too many positional arguments in {func_name}(); "
                f"only {list(_POSITIONAL_PARAMS)} may be positional "
                f"(labels and later are keyword-only)"
            )

        parsed: dict[str, object] = {}

        for name, arg in zip(_POSITIONAL_PARAMS, func_call.args, strict=False):
            v = _extract_constant(arg)
            if v is _UNSET:
                raise ValueError(f"{self.filename}: {name} in {func_name}() must be a literal constant")
            parsed[name] = v

        for kw in func_call.keywords:
            if kw.arg is None:
                raise ValueError(f"{self.filename}: **kwargs are not supported in {func_name}()")
            if kw.arg in parsed:
                raise ValueError(f"{self.filename}: duplicated argument '{kw.arg}' in {func_name}()")
            if kw.arg not in _VALID_KWARGS:
                raise ValueError(f"{self.filename}: unknown argument '{kw.arg}' in {func_name}()")
            if kw.arg == "labels":
                parsed["labels"] = _extract_list_constant(
                    kw.value, context=f"{self.filename}: labels in {func_name}()"
                )
            else:
                v = _extract_constant(kw.value)
                if v is _UNSET:
                    raise ValueError(f"{self.filename}: {kw.arg} in {func_name}() must be a literal constant")
                parsed[kw.arg] = v

        if "est_time" not in parsed:
            raise ValueError(f"{self.filename}: est_time is required in {func_name}()")
        if "suite" not in parsed:
            raise ValueError(f"{self.filename}: suite is required in {func_name}()")

        if not isinstance(parsed["est_time"], (int, float)):
            raise ValueError(f"{self.filename}: est_time must be a number in {func_name}()")
        if not isinstance(parsed["suite"], str):
            raise ValueError(f"{self.filename}: suite must be a string in {func_name}()")

        # `labels` is optional. Missing / None / [] all mean "always run on
        # every PR"; only a non-empty list gates the test on PR labels.
        labels = parsed.get("labels", [])
        if not isinstance(labels, list):
            raise ValueError(f"{self.filename}: labels must be a list or None in {func_name}()")

        nightly = parsed.get("nightly", False)
        if not isinstance(nightly, bool):
            raise ValueError(f"{self.filename}: nightly must be a boolean in {func_name}()")

        disabled = parsed.get("disabled", None)
        if disabled is not None and not isinstance(disabled, str):
            raise ValueError(f"{self.filename}: disabled must be a string or None in {func_name}()")

        unknown = [label for label in labels if label not in KNOWN_LABELS]
        if unknown:
            valid_list = ", ".join(sorted(KNOWN_LABELS))
            raise ValueError(
                f"{self.filename}: unknown labels {unknown} in {func_name}(); "
                f"valid labels: [{valid_list}]. "
                f"To add a new label: edit tests/ci/labels.py + create matching "
                f"`run-ci-<label>` in GitHub repo Settings -> Labels."
            )

        return CIRegistry(
            backend=_REGISTER_BACKEND_MAP[func_name],
            filename=self.filename,
            est_time=float(parsed["est_time"]),
            suite=parsed["suite"],
            labels=list(labels),
            nightly=nightly,
            disabled=disabled,
            implicit=False,
        )

    def _collect_ci_registry(self, func_call: ast.Call):
        if not isinstance(func_call.func, ast.Name):
            return None
        if func_call.func.id not in _REGISTER_NAMES:
            return None
        return self._parse_call_args(func_call, func_call.func.id)

    def visit_Module(self, node):
        for stmt in node.body:
            if not isinstance(stmt, ast.Expr) or not isinstance(stmt.value, ast.Call):
                continue
            cr = self._collect_ci_registry(stmt.value)
            if cr is not None:
                self.registries.append(cr)


def ut_parse_one_file(filename: str) -> list[CIRegistry]:
    with open(filename) as f:
        file_content = f.read()
    tree = ast.parse(file_content, filename=filename)
    visitor = RegistryVisitor(filename=filename)
    visitor.visit(tree)
    return visitor.registries


def _is_implicit_fast_cpu_path(filename: str) -> bool:
    return filename.startswith("tests/fast/")


# Directories the CI runner scans.
# 1. tests/fast/ is CPU-only and auto-registers
# 2. the rest require an explicit register_*_ci on each discovered file.
# 3. Only test_*.py are collected.
# 4. Patterns are repo-relative, so the runner must run from the repo root (the same cwd ut_parse_one_file's open() assumes).
_DISCOVERY_ROOTS = ("tests/fast", "tests/fast-gpu", "tests/e2e", "tests/ci")


def discover_ci_files() -> list[str]:
    """Return every CI test file (sorted, repo-relative) across the roots."""
    files: list[str] = []
    for root in _DISCOVERY_ROOTS:
        files.extend(glob.glob(f"{root}/**/test_*.py", recursive=True))
    return sorted(files)


def _file_text_mentions_register(filename: str) -> bool:
    """True when the file's text contains `register_cpu_ci` or
    `register_cuda_ci` as a substring anywhere.

    Used as a defense-in-depth check before synthesizing an implicit CPU
    registry for a tests/fast/ file with zero parsed registries: if the
    file mentions either symbol but the AST visitor found no top-level
    Expr(Call), the file probably has an aliased import, a non-toplevel
    call, or an attribute-style call (`ci_register.register_cpu_ci(...)`)
    -- silently treating it as unregistered would mask the intent.
    """
    try:
        with open(filename) as f:
            content = f.read()
    except OSError:
        return False
    return "register_cpu_ci" in content or "register_cuda_ci" in content


def _make_implicit_cpu_registry(filename: str) -> CIRegistry:
    return CIRegistry(
        backend=HWBackend.CPU,
        filename=filename,
        est_time=1.0,
        suite="stage-a-cpu",
        labels=[],
        nightly=False,
        disabled=None,
        implicit=True,
    )


def collect_tests(files: list[str], sanity_check: bool = True) -> list[CIRegistry]:
    ci_tests: list[CIRegistry] = []
    for file in files:
        registries = ut_parse_one_file(file)
        if _is_implicit_fast_cpu_path(file):
            # tests/fast/ is CPU-only by location;
            for r in registries:
                if r.backend != HWBackend.CPU:
                    raise ValueError(
                        f"{file}: register_cuda_ci is forbidden in tests/fast/; "
                        f"move the file to tests/fast-gpu/ instead"
                    )
            if len(registries) == 0:
                if _file_text_mentions_register(file):
                    raise ValueError(
                        f"{file}: file mentions register_cpu_ci or register_cuda_ci "
                        f"textually but no top-level call was parsed; check for "
                        f"aliased import, non-toplevel call, or attribute access"
                    )
                ci_tests.append(_make_implicit_cpu_registry(file))
                continue
        if len(registries) == 0:
            msg = f"No CI registry found in {file}"
            if sanity_check:
                raise ValueError(msg)
            warnings.warn(msg, stacklevel=2)
            continue
        ci_tests.extend(registries)
    return ci_tests
