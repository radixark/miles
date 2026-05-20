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
from tests.ci.ci_register import _UNSET, HWBackend, _extract_constant, _extract_list_constant, ut_parse_one_file


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
