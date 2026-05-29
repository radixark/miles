"""AST-time validation tests for register_cuda_ci / register_cpu_ci.

Covers the AST-collection behavior of the suite-as-runner-class refactor:
`labels` is optional (default None ≡ []); a non-empty list of canonical
domain labels gates the test on PR labels, while None / [] / omitted means
the test runs on every PR; `num_gpus` is gone; `labels` must be passed
by keyword (not as a positional third argument).
"""

import ast
import textwrap
from pathlib import Path

import pytest
from tests.ci.ci_register import (
    _UNSET,
    HWBackend,
    _extract_constant,
    _extract_list_constant,
    _file_text_mentions_register,
    _is_implicit_fast_cpu_path,
    collect_tests,
    register_cpu_ci,
    ut_parse_one_file,
)

register_cpu_ci(est_time=1, suite="stage-a-cpu", labels=[])


@pytest.fixture(autouse=True)
def _run_from_tmp_repo_root(tmp_path, monkeypatch):
    """Run every test with cwd at a throwaway tmp_path "repo root".

    The collect_tests helpers below write fixture files under
    tmp_path/tests/fast/... and hand collect_tests the repo-relative path
    (tests/fast/...), exactly as run_suite.py does from the repo root in CI.
    Anchoring cwd here lets both ut_parse_one_file's open() and the
    tests/fast/ prefix check resolve against the same root. Tests that read
    files by absolute path or by __file__ (the AC-9 tree scan) are unaffected.
    """
    monkeypatch.chdir(tmp_path)


def _make_fixture(body: str, tmp_path: Path, name: str = "fixture.py") -> str:
    p = tmp_path / name
    p.write_text(textwrap.dedent(body).lstrip("\n"))
    return str(p)


# --- Positive: accepted register_cuda_ci / register_cpu_ci shapes -----------


class TestRegisterPositive:
    def test_cuda_basic_with_one_label(self, tmp_path):
        path = _make_fixture(
            """
            from tests.ci.ci_register import register_cuda_ci
            register_cuda_ci(est_time=600, suite="stage-c-8-gpu-h100", labels=["megatron"])
            """,
            tmp_path,
        )
        registries = ut_parse_one_file(path)
        assert len(registries) == 1
        r = registries[0]
        assert r.backend == HWBackend.CUDA
        assert r.suite == "stage-c-8-gpu-h100"
        assert r.labels == ["megatron"]
        assert not hasattr(r, "num_gpus")
        assert not hasattr(r, "always_on")

    def test_labels_omitted_is_always_run(self, tmp_path):
        # No `labels=` keyword at all: defaults to [] (always-run semantics).
        path = _make_fixture(
            """
            from tests.ci.ci_register import register_cpu_ci
            register_cpu_ci(est_time=30, suite="stage-a-cpu")
            """,
            tmp_path,
        )
        r = ut_parse_one_file(path)[0]
        assert r.backend == HWBackend.CPU
        assert r.suite == "stage-a-cpu"
        assert r.labels == []

    def test_labels_none_is_always_run(self, tmp_path):
        # Explicit `labels=None` is equivalent to omitting / `labels=[]`.
        path = _make_fixture(
            """
            from tests.ci.ci_register import register_cuda_ci
            register_cuda_ci(est_time=60, suite="stage-b-2-gpu-h200", labels=None)
            """,
            tmp_path,
        )
        r = ut_parse_one_file(path)[0]
        assert r.labels == []

    def test_labels_empty_list_is_always_run(self, tmp_path):
        # `labels=[]` is also legal and means always-run; no never-run rule.
        path = _make_fixture(
            """
            from tests.ci.ci_register import register_cuda_ci
            register_cuda_ci(est_time=60, suite="stage-b-2-gpu-h200", labels=[])
            """,
            tmp_path,
        )
        r = ut_parse_one_file(path)[0]
        assert r.labels == []

    def test_cuda_multiple_labels(self, tmp_path):
        path = _make_fixture(
            """
            from tests.ci.ci_register import register_cuda_ci
            register_cuda_ci(
                est_time=600,
                suite="stage-c-8-gpu-h100",
                labels=["megatron", "sglang"],
            )
            """,
            tmp_path,
        )
        assert ut_parse_one_file(path)[0].labels == ["megatron", "sglang"]

    def test_disabled_string_passthrough(self, tmp_path):
        path = _make_fixture(
            """
            from tests.ci.ci_register import register_cuda_ci
            register_cuda_ci(
                est_time=600,
                suite="stage-c-8-gpu-h100",
                labels=["megatron"],
                disabled="known regression in megatron 0.12",
            )
            """,
            tmp_path,
        )
        r = ut_parse_one_file(path)[0]
        assert r.disabled == "known regression in megatron 0.12"


# --- Negative: rejected shapes (each error message is part of the contract) -


class TestRegisterNegative:
    def test_unknown_label_rejected(self, tmp_path):
        path = _make_fixture(
            """
            from tests.ci.ci_register import register_cuda_ci
            register_cuda_ci(est_time=600, suite="stage-c-8-gpu-h100", labels=["megatorn"])
            """,
            tmp_path,
        )
        with pytest.raises(ValueError, match=r"unknown labels.*megatorn"):
            ut_parse_one_file(path)

    def test_num_gpus_kwarg_rejected(self, tmp_path):
        path = _make_fixture(
            """
            from tests.ci.ci_register import register_cuda_ci
            register_cuda_ci(
                est_time=600, suite="stage-c-8-gpu-h100", labels=["megatron"], num_gpus=8
            )
            """,
            tmp_path,
        )
        with pytest.raises(ValueError, match=r"unknown argument 'num_gpus'"):
            ut_parse_one_file(path)

    def test_always_on_kwarg_rejected(self, tmp_path):
        # `always_on` is gone in the new design; passing it must error.
        path = _make_fixture(
            """
            from tests.ci.ci_register import register_cuda_ci
            register_cuda_ci(
                est_time=600, suite="stage-c-8-gpu-h100", labels=[], always_on=True
            )
            """,
            tmp_path,
        )
        with pytest.raises(ValueError, match=r"unknown argument 'always_on'"):
            ut_parse_one_file(path)

    def test_labels_string_rejected(self, tmp_path):
        path = _make_fixture(
            """
            from tests.ci.ci_register import register_cuda_ci
            register_cuda_ci(est_time=600, suite="stage-c-8-gpu-h100", labels="megatron")
            """,
            tmp_path,
        )
        with pytest.raises(ValueError, match=r"must be a list of string literals or None"):
            ut_parse_one_file(path)

    def test_positional_third_arg_rejected(self, tmp_path):
        # labels is keyword-only; a third positional argument must be rejected.
        path = _make_fixture(
            """
            from tests.ci.ci_register import register_cuda_ci
            register_cuda_ci(600, "stage-c-8-gpu-h100", ["megatron"])
            """,
            tmp_path,
        )
        with pytest.raises(ValueError, match=r"too many positional arguments"):
            ut_parse_one_file(path)

    def test_duplicate_kwarg_rejected(self, tmp_path):
        path = _make_fixture(
            """
            from tests.ci.ci_register import register_cuda_ci
            register_cuda_ci(
                est_time=600, suite="stage-c-8-gpu-h100", suite="oops", labels=["megatron"]
            )
            """,
            tmp_path,
        )
        with pytest.raises(ValueError, match=r"duplicated argument 'suite'"):
            ut_parse_one_file(path)


# --- AST helpers (extraction-only, isolated from KNOWN_LABELS validation) ---


class TestExtractionHelpers:
    def test_extract_constant_int(self):
        node = ast.parse("42", mode="eval").body
        assert _extract_constant(node) == 42

    def test_extract_constant_str(self):
        node = ast.parse("'hello'", mode="eval").body
        assert _extract_constant(node) == "hello"

    def test_extract_constant_bool(self):
        assert _extract_constant(ast.parse("True", mode="eval").body) is True
        assert _extract_constant(ast.parse("False", mode="eval").body) is False

    def test_extract_constant_non_constant_returns_unset(self):
        node = ast.parse("some_var", mode="eval").body
        assert _extract_constant(node) is _UNSET

    def test_extract_list_constant_strings(self):
        node = ast.parse('["a", "b"]', mode="eval").body
        assert _extract_list_constant(node) == ["a", "b"]

    def test_extract_list_constant_empty(self):
        node = ast.parse("[]", mode="eval").body
        assert _extract_list_constant(node) == []

    def test_extract_list_constant_none_is_empty(self):
        # Treat literal `None` as equivalent to `[]` (always-run intent).
        node = ast.parse("None", mode="eval").body
        assert _extract_list_constant(node) == []

    def test_extract_list_constant_non_list(self):
        node = ast.parse("some_var", mode="eval").body
        with pytest.raises(ValueError, match=r"must be a list of string literals or None"):
            _extract_list_constant(node)

    def test_extract_list_constant_non_literal_element(self):
        node = ast.parse("[get_label()]", mode="eval").body
        with pytest.raises(ValueError, match=r"must be a list of string literals"):
            _extract_list_constant(node)

    def test_extract_list_constant_non_string_element(self):
        node = ast.parse("[1, 2]", mode="eval").body
        with pytest.raises(ValueError, match=r"must be a list of string literals"):
            _extract_list_constant(node)


# --- _is_implicit_fast_cpu_path: prefix-anchored path discrimination ---------


class TestIsImplicitFastCpuPath:
    def test_relative_tests_fast_path(self):
        assert _is_implicit_fast_cpu_path("tests/fast/test_x.py")

    def test_dot_prefixed_path_not_recognized(self):
        # Inputs are the bare repo-relative paths glob.glob yields
        # (tests/fast/...); a ./-prefixed form is out of contract.
        assert not _is_implicit_fast_cpu_path("./tests/fast/test_x.py")

    def test_absolute_path_not_recognized(self):
        # Classification is on the repo-relative path; absolute paths are not
        # recognized (open(filename) would already require cwd == repo root).
        assert not _is_implicit_fast_cpu_path("/abs/repo/tests/fast/test_x.py")

    def test_nested_subdir(self):
        assert _is_implicit_fast_cpu_path("tests/fast/sub/dir/test_x.py")

    def test_fast_gpu_sibling_rejected(self):
        assert not _is_implicit_fast_cpu_path("tests/fast-gpu/test_x.py")

    def test_fast_gpu_absolute_rejected(self):
        assert not _is_implicit_fast_cpu_path("/abs/repo/tests/fast-gpu/test_x.py")

    def test_fast_gpu_nested_rejected(self):
        assert not _is_implicit_fast_cpu_path("tests/fast-gpu/sub/test_x.py")

    def test_same_prefix_siblings_rejected(self):
        assert not _is_implicit_fast_cpu_path("tests/fastish/test_x.py")
        assert not _is_implicit_fast_cpu_path("tests/fastfoo/test_x.py")

    def test_bare_directory_without_file_rejected(self):
        # "tests/fast" alone is a directory path; helper is for file paths.
        assert not _is_implicit_fast_cpu_path("tests/fast")

    def test_other_subtrees_rejected(self):
        assert not _is_implicit_fast_cpu_path("tests/e2e/short/test_x.py")
        assert not _is_implicit_fast_cpu_path("tests/utils/test_x.py")
        assert not _is_implicit_fast_cpu_path("tests/ci/test_x.py")


# --- _file_text_mentions_register: substring-based suspicious detection ------


class TestFileTextMentionsRegister:
    def test_file_with_cpu_call_matches(self, tmp_path):
        p = tmp_path / "f.py"
        p.write_text(
            "from tests.ci.ci_register import register_cpu_ci\nregister_cpu_ci(est_time=10, suite='stage-a-cpu', labels=[])\n"
        )
        assert _file_text_mentions_register(str(p))

    def test_file_with_cuda_call_matches(self, tmp_path):
        p = tmp_path / "f.py"
        p.write_text(
            "from tests.ci.ci_register import register_cuda_ci\nregister_cuda_ci(est_time=60, suite='stage-b-2-gpu-h200', labels=[])\n"
        )
        assert _file_text_mentions_register(str(p))

    def test_clean_file_does_not_match(self, tmp_path):
        p = tmp_path / "f.py"
        p.write_text("def test_x():\n    assert True\n")
        assert not _file_text_mentions_register(str(p))

    def test_missing_file_does_not_raise(self, tmp_path):
        assert not _file_text_mentions_register(str(tmp_path / "nonexistent.py"))


# --- collect_tests: implicit synthesis, suspicious guard, cuda ban -----------


def _make_under(tmp_path: Path, subtree: str, rel_path: str, body: str) -> str:
    """Create a file under tmp_path/<subtree>/<rel_path> and return the
    repo-relative path (<subtree>/<rel_path>).

    The file lives under tmp_path (the autouse-fixture cwd), but collect_tests
    is handed the repo-relative path so both its open() and its tests/fast/
    prefix check resolve the same way they do in CI.
    """
    target = tmp_path / subtree / rel_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(textwrap.dedent(body).lstrip("\n"))
    return f"{subtree}/{rel_path}"


def _make_under_tests_fast(tmp_path: Path, rel_path: str, body: str) -> str:
    return _make_under(tmp_path, "tests/fast", rel_path, body)


def _make_under_tests_fast_gpu(tmp_path: Path, rel_path: str, body: str) -> str:
    return _make_under(tmp_path, "tests/fast-gpu", rel_path, body)


class TestCollectTestsImplicitFallback:
    def test_fast_cpu_no_register_call_synthesizes_default(self, tmp_path):
        # AC-1 positive: a tests/fast/ file with no register call yields
        # one implicit CPU CIRegistry with the documented default values.
        path = _make_under_tests_fast(
            tmp_path,
            "test_foo.py",
            "def test_x():\n    assert True\n",
        )
        registries = collect_tests([path], sanity_check=True)
        assert len(registries) == 1
        r = registries[0]
        assert r.backend == HWBackend.CPU
        assert r.suite == "stage-a-cpu"
        assert r.est_time == 1.0
        assert r.labels == []
        assert r.nightly is False
        assert r.disabled is None
        assert r.implicit is True
        assert r.filename == path

    def test_fast_cpu_nested_subdir_also_synthesizes(self, tmp_path):
        # AC-1 positive: deep subdirectories under tests/fast/ behave the same.
        path = _make_under_tests_fast(
            tmp_path,
            "sub/deeper/test_x.py",
            "def test_x():\n    assert True\n",
        )
        registries = collect_tests([path], sanity_check=True)
        assert len(registries) == 1 and registries[0].implicit is True

    def test_outside_fast_raises_with_sanity_check(self, tmp_path):
        # AC-1 negative: non-tests/fast/ subtrees still hit the existing
        # ValueError path under sanity_check=True.
        for subtree in ("tests/utils", "tests/e2e/short", "tests/ci"):
            path = _make_under(tmp_path, subtree, "test_no_reg.py", "def test_x():\n    assert True\n")
            with pytest.raises(ValueError, match=r"No CI registry found"):
                collect_tests([path], sanity_check=True)


class TestCollectTestsExplicitPrecedence:
    def test_explicit_cpu_call_wins_over_implicit(self, tmp_path):
        # AC-2 positive: when registries is non-empty, no implicit
        # synthesis fires; explicit values take effect verbatim.
        path = _make_under_tests_fast(
            tmp_path,
            "test_bar.py",
            """
            from tests.ci.ci_register import register_cpu_ci
            register_cpu_ci(est_time=20, suite="stage-a-cpu", labels=["megatron"])
            """,
        )
        registries = collect_tests([path], sanity_check=True)
        assert len(registries) == 1
        r = registries[0]
        assert r.est_time == 20.0
        assert r.labels == ["megatron"]
        assert r.implicit is False

    def test_explicit_overrides_can_use_alternate_suite(self, tmp_path):
        # AC-2 positive: register_cpu_ci is allowed to override any field,
        # including switching to a non-default CPU suite.
        path = _make_under_tests_fast(
            tmp_path,
            "test_baz.py",
            """
            from tests.ci.ci_register import register_cpu_ci
            register_cpu_ci(est_time=15, suite="stage-b-cpu", labels=[])
            """,
        )
        r = collect_tests([path], sanity_check=True)[0]
        assert r.suite == "stage-b-cpu"
        assert r.est_time == 15.0
        assert r.implicit is False


class TestCollectTestsFastGpuStrict:
    def test_fast_gpu_with_cuda_call_accepted(self, tmp_path):
        # AC-3 positive: tests/fast-gpu/ files with register_cuda_ci pass
        # through unchanged, just like any other CUDA registry.
        path = _make_under_tests_fast_gpu(
            tmp_path,
            "test_gpu_thing.py",
            """
            from tests.ci.ci_register import register_cuda_ci
            register_cuda_ci(est_time=60, suite="stage-b-2-gpu-h200", labels=[])
            """,
        )
        registries = collect_tests([path], sanity_check=True)
        assert len(registries) == 1
        assert registries[0].backend == HWBackend.CUDA
        assert registries[0].implicit is False

    def test_fast_gpu_with_cuda_alternate_suite_accepted(self, tmp_path):
        # AC-3 positive: other CUDA suites and label sets work too.
        path = _make_under_tests_fast_gpu(
            tmp_path,
            "test_h100.py",
            """
            from tests.ci.ci_register import register_cuda_ci
            register_cuda_ci(
                est_time=120, suite="stage-c-8-gpu-h100", labels=["megatron"]
            )
            """,
        )
        registries = collect_tests([path], sanity_check=True)
        assert registries[0].suite == "stage-c-8-gpu-h100"

    def test_fast_gpu_with_no_register_raises(self, tmp_path):
        # AC-3 negative: tests/fast-gpu/ does NOT enjoy implicit fallback;
        # zero register calls still go through the ValueError path.
        path = _make_under_tests_fast_gpu(
            tmp_path,
            "test_y.py",
            "def test_x():\n    assert True\n",
        )
        with pytest.raises(ValueError, match=r"No CI registry found"):
            collect_tests([path], sanity_check=True)


class TestCollectTestsCudaBanInFast:
    def test_register_cuda_ci_in_fast_raises(self, tmp_path):
        # AC-4 positive: register_cuda_ci anywhere in tests/fast/ is a
        # hard error directing the developer to tests/fast-gpu/.
        path = _make_under_tests_fast(
            tmp_path,
            "test_misplaced.py",
            """
            from tests.ci.ci_register import register_cuda_ci
            register_cuda_ci(est_time=60, suite="stage-b-2-gpu-h200", labels=[])
            """,
        )
        with pytest.raises(ValueError, match=r"register_cuda_ci is forbidden in tests/fast/"):
            collect_tests([path], sanity_check=True)

    def test_same_call_accepted_in_fast_gpu(self, tmp_path):
        # AC-4 negative: identical call inside tests/fast-gpu/ passes.
        path = _make_under_tests_fast_gpu(
            tmp_path,
            "test_ok.py",
            """
            from tests.ci.ci_register import register_cuda_ci
            register_cuda_ci(est_time=60, suite="stage-b-2-gpu-h200", labels=[])
            """,
        )
        registries = collect_tests([path], sanity_check=True)
        assert registries[0].backend == HWBackend.CUDA


class TestCollectTestsSuspiciousPattern:
    def test_aliased_import_no_call_raises(self, tmp_path):
        # AC-6 positive: file mentions register_cpu_ci textually (via
        # aliased import) but visitor finds zero registries; suspicious
        # guard must raise instead of silently synthesizing default.
        path = _make_under_tests_fast(
            tmp_path,
            "test_aliased.py",
            """
            from tests.ci.ci_register import register_cpu_ci as reg
            reg(est_time=10, suite="stage-a-cpu", labels=[])
            """,
        )
        with pytest.raises(ValueError, match=r"mentions register_cpu_ci or register_cuda_ci"):
            collect_tests([path], sanity_check=True)

    def test_non_toplevel_call_raises(self, tmp_path):
        # AC-6 positive: register_cpu_ci buried inside an `if` block is
        # not parsed by RegistryVisitor (top-level Expr(Call) only), but
        # the textual substring trips the suspicious guard.
        path = _make_under_tests_fast(
            tmp_path,
            "test_nontoplevel.py",
            """
            from tests.ci.ci_register import register_cpu_ci
            if True:
                register_cpu_ci(est_time=10, suite="stage-a-cpu", labels=[])
            """,
        )
        with pytest.raises(ValueError, match=r"mentions register_cpu_ci or register_cuda_ci"):
            collect_tests([path], sanity_check=True)

    def test_attribute_access_call_raises(self, tmp_path):
        # AC-6 positive: ci_register.register_cpu_ci(...) is parsed as
        # ast.Attribute, not ast.Name, so the visitor produces zero
        # registries; suspicious guard catches it.
        path = _make_under_tests_fast(
            tmp_path,
            "test_attr.py",
            """
            from tests.ci import ci_register
            ci_register.register_cpu_ci(est_time=10, suite="stage-a-cpu", labels=[])
            """,
        )
        with pytest.raises(ValueError, match=r"mentions register_cpu_ci or register_cuda_ci"):
            collect_tests([path], sanity_check=True)

    def test_clean_file_passes_implicit(self, tmp_path):
        # AC-6 negative: a file with neither substring present takes the
        # legitimate implicit-fallback path.
        path = _make_under_tests_fast(
            tmp_path,
            "test_clean.py",
            "def test_x():\n    assert True\n",
        )
        registries = collect_tests([path], sanity_check=True)
        assert registries[0].implicit is True


# --- AC-9 regression guard: no semantic-default register_cpu_ci remains ------


def _is_semantic_default_register_cpu_ci(call: ast.Call) -> bool:
    """Mirror of `strip_default_cpu_register._is_default_form_register_cpu_ci`.

    True when the call matches any of three semantically identical shapes
    of the implicit-fallback default (all collapse to
    `est_time=10, suite="stage-a-cpu", labels=[]` after AST extraction):

    1. Full 3-kwarg form: `register_cpu_ci(est_time=10, suite="stage-a-cpu", labels=[])`.
    2. Omitted-`labels` 2-kwarg form: `register_cpu_ci(est_time=10, suite="stage-a-cpu")`
       (omitted `labels` defaults to `None` ≡ `[]` per
       `_extract_list_constant`'s rule).
    3. Explicit-None 3-kwarg form:
       `register_cpu_ci(est_time=10, suite="stage-a-cpu", labels=None)`.
    """
    if not isinstance(call.func, ast.Name) or call.func.id != "register_cpu_ci":
        return False
    if call.args:
        return False
    kwarg_names = {kw.arg for kw in call.keywords}
    accepted = ({"est_time", "suite", "labels"}, {"est_time", "suite"})
    if kwarg_names not in accepted:
        return False
    for kw in call.keywords:
        if kw.arg == "est_time":
            if not isinstance(kw.value, ast.Constant) or kw.value.value != 10:
                return False
        elif kw.arg == "suite":
            if not isinstance(kw.value, ast.Constant) or kw.value.value != "stage-a-cpu":
                return False
        elif kw.arg == "labels":
            is_empty_list = isinstance(kw.value, ast.List) and not kw.value.elts
            is_literal_none = isinstance(kw.value, ast.Constant) and kw.value.value is None
            if not (is_empty_list or is_literal_none):
                return False
        else:
            return False
    return True


class TestNoSemanticDefaultRegisterCpuCi:
    """AC-9 meta-test: every file under `tests/fast/` that still calls
    `register_cpu_ci` must do so with a *non-default* signature.

    The implicit-CPU-registration fallback in `collect_tests` already
    synthesises the equivalent of `register_cpu_ci(est_time=10,
    suite="stage-a-cpu", labels=[])` for any `tests/fast/**/*.py` file with
    no top-level register call, so a literal default-form caller is
    redundant. This test scans the real tree and fails on any reintroduced
    default to keep the codemod's guarantee enforced.

    The default has three semantically-equivalent spellings (full
    `labels=[]`, omitted-`labels`, and explicit `labels=None`); the
    predicate matches all three so the guarantee is exhaustive.
    """

    def test_no_default_form_register_cpu_ci_in_tests_fast(self):
        repo_root = Path(__file__).resolve().parents[2]
        fast_root = repo_root / "tests" / "fast"
        if not fast_root.is_dir():
            pytest.skip(f"{fast_root} does not exist in this checkout")

        offenders: list[str] = []
        for py in sorted(fast_root.rglob("*.py")):
            try:
                text = py.read_text()
            except OSError:
                continue
            try:
                tree = ast.parse(text, filename=str(py))
            except SyntaxError:
                continue
            for node in tree.body:
                if not isinstance(node, ast.Expr):
                    continue
                if not isinstance(node.value, ast.Call):
                    continue
                if _is_semantic_default_register_cpu_ci(node.value):
                    rel = py.relative_to(repo_root)
                    offenders.append(f"{rel}:{node.lineno}")

        assert not offenders, (
            "found semantic-default register_cpu_ci call(s) under tests/fast/; "
            "the implicit fallback in tests/ci/ci_register.collect_tests already "
            "synthesises the equivalent registry. Remove these calls (and the "
            "orphan import) or run scripts/tools/strip_default_cpu_register.py.\n"
            "Offenders:\n  " + "\n  ".join(offenders)
        )

    # --- inline AST-fabrication unit tests for the predicate itself ---------
    #
    # The full-tree scan above passes vacuously today (no callers under
    # tests/fast/ use the default form anymore). These fabricated AST nodes
    # exercise each accepted shape and a few rejection cases directly, so a
    # regression in the matching logic surfaces here without needing a real
    # offender file to be committed.

    @staticmethod
    def _parse_call(src: str) -> ast.Call:
        tree = ast.parse(src, mode="exec")
        expr = tree.body[0]
        assert isinstance(expr, ast.Expr) and isinstance(expr.value, ast.Call)
        return expr.value

    def test_predicate_accepts_full_empty_list_form(self):
        call = self._parse_call('register_cpu_ci(est_time=10, suite="stage-a-cpu", labels=[])')
        assert _is_semantic_default_register_cpu_ci(call)

    def test_predicate_accepts_omitted_labels_form(self):
        call = self._parse_call('register_cpu_ci(est_time=10, suite="stage-a-cpu")')
        assert _is_semantic_default_register_cpu_ci(call)

    def test_predicate_accepts_labels_none_literal_form(self):
        # The third equivalence class Codex flagged: `labels=None` literal
        # is identical to omitted / `labels=[]` per `_extract_list_constant`.
        call = self._parse_call('register_cpu_ci(est_time=10, suite="stage-a-cpu", labels=None)')
        assert _is_semantic_default_register_cpu_ci(call)

    def test_predicate_rejects_nonempty_labels(self):
        call = self._parse_call('register_cpu_ci(est_time=10, suite="stage-a-cpu", labels=["megatron"])')
        assert not _is_semantic_default_register_cpu_ci(call)

    def test_predicate_rejects_nondefault_est_time(self):
        call = self._parse_call('register_cpu_ci(est_time=20, suite="stage-a-cpu", labels=[])')
        assert not _is_semantic_default_register_cpu_ci(call)

    def test_predicate_rejects_nondefault_suite(self):
        call = self._parse_call('register_cpu_ci(est_time=10, suite="stage-b-cpu", labels=[])')
        assert not _is_semantic_default_register_cpu_ci(call)

    def test_predicate_rejects_extra_kwarg(self):
        call = self._parse_call('register_cpu_ci(est_time=10, suite="stage-a-cpu", labels=[], nightly=True)')
        assert not _is_semantic_default_register_cpu_ci(call)

    def test_predicate_rejects_positional_args(self):
        call = self._parse_call('register_cpu_ci(10, "stage-a-cpu")')
        assert not _is_semantic_default_register_cpu_ci(call)

    def test_predicate_rejects_labels_non_literal_name(self):
        # `labels=SOME_NAME` is neither an empty-list literal nor a literal
        # `None`; the predicate must keep its hands off the call.
        call = self._parse_call('register_cpu_ci(est_time=10, suite="stage-a-cpu", labels=SOME_NAME)')
        assert not _is_semantic_default_register_cpu_ci(call)
