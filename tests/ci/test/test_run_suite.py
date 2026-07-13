"""Unit tests for `run_suite.py`.

These cover the Python-side label pipeline:

* `strip_run_ci_prefix`: empty input, prefix stripping, silent skip of
  workflow-only labels, warning on other non-prefixed inputs.
* `resolve_scope`: the broad-scope policy (event + raw labels -> one
  effective include-label set), pinned case-by-case to the selection the
  per-stage pr-test.yml expressions produced before the policy moved here.
* The pr-test.yml seam: every stage passes `--event-name` and no scope flags.
* `filter_tests`: include-set selection with the "empty labels means always
  run" semantic; a scope subtraction is not a per-test veto.
* `PER_COMMIT_SUITES`: locked to the new taxonomy including the
  always-run GPU bucket `stage-b-2-gpu-h200`.

We build `CIRegistry` instances directly via a small factory rather than
parsing fixture files -- the AST-side validation lives in
`test_ci_register.py`; this module exercises the runtime filter.
"""

import warnings
from pathlib import Path

import pytest
from tests.ci.ci_register import CIRegistry, HWBackend, discover_ci_files, register_cpu_ci
from tests.ci.labels import KNOWN_LABELS
from tests.ci.run_suite import (
    PER_COMMIT_SUITES,
    build_cpu_pytest_cmd,
    filter_tests,
    resolve_scope,
    strip_run_ci_prefix,
)

register_cpu_ci(est_time=1, suite="stage-a-cpu", labels=[])


def _make(
    filename: str,
    *,
    backend: HWBackend = HWBackend.CUDA,
    suite: str = "stage-c-8-gpu-h100",
    labels: list[str] | None = None,
    est_time: float = 60.0,
    nightly: bool = False,
    disabled: str | None = None,
) -> CIRegistry:
    """Minimal `CIRegistry` factory for filter tests.

    `labels=None` and `labels=[]` are equivalent (always-run semantics).
    """
    return CIRegistry(
        backend=backend,
        filename=filename,
        est_time=est_time,
        suite=suite,
        labels=list(labels) if labels is not None else [],
        nightly=nightly,
        disabled=disabled,
        implicit=False,
    )


# --- build_cpu_pytest_cmd: -x gated on continue_on_error --------------------


class TestBuildCpuPytestCmd:
    def test_x_present_by_default(self):
        # Default per-commit run stops at the first failure.
        cmd = build_cpu_pytest_cmd(["tests/fast/a.py", "tests/fast/b.py"], continue_on_error=False)
        assert "-x" in cmd

    def test_x_dropped_on_continue_on_error(self):
        # bypass-fastfail passes --continue-on-error -> run every file to the end.
        cmd = build_cpu_pytest_cmd(["tests/fast/a.py", "tests/fast/b.py"], continue_on_error=True)
        assert "-x" not in cmd
        assert cmd[0] == "pytest"
        assert "tests/fast/a.py" in cmd and "tests/fast/b.py" in cmd


# --- PER_COMMIT_SUITES locked to the new taxonomy ---------------------------


class TestPerCommitSuites:
    def test_cpu_suites_exact(self):
        assert PER_COMMIT_SUITES[HWBackend.CPU] == ["stage-a-cpu", "stage-b-cpu"]

    def test_cuda_suites_exact(self):
        assert PER_COMMIT_SUITES[HWBackend.CUDA] == [
            "stage-b-2-gpu-h200",
            "stage-c-8-gpu-h100",
            "stage-c-8-gpu-h200",
            "stage-c-4-gpu-h200",
            "stage-c-2-gpu-h200",
        ]

    def test_no_legacy_suite_names_remain(self):
        legacy = {
            "stage-a-fast",
            "stage-b-fast-1-gpu",
            "stage-b-fast-gpu",
            "stage-b-short-8-gpu",
            "stage-b-sglang-8-gpu",
            "stage-b-8-gpu-h100",
            "stage-c-fsdp-8-gpu",
            "stage-c-megatron-8-gpu",
            "stage-c-precision-8-gpu",
            "stage-c-ckpt-8-gpu",
            "stage-c-long-8-gpu",
            "stage-c-lora-8-gpu",
            "stage-c-all",
        }
        all_suites = {s for suites in PER_COMMIT_SUITES.values() for s in suites}
        assert legacy.isdisjoint(all_suites), f"Legacy suite name(s) still present: {legacy & all_suites}"


# --- `strip_run_ci_prefix` direct tests -------------------------------------


class TestStripRunCiPrefix:
    def test_empty_input_yields_empty_set(self):
        assert strip_run_ci_prefix([]) == set()

    def test_single_prefixed_label_stripped(self):
        assert strip_run_ci_prefix(["run-ci-megatron"]) == {"megatron"}

    def test_multiple_prefixed_labels_stripped(self):
        assert strip_run_ci_prefix(["run-ci-megatron", "run-ci-fsdp"]) == {"megatron", "fsdp"}

    def test_duplicate_inputs_deduplicate(self):
        assert strip_run_ci_prefix(["run-ci-megatron", "run-ci-megatron"]) == {"megatron"}

    def test_non_prefixed_input_warns_and_is_skipped(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = strip_run_ci_prefix(["megatron"])
        assert result == set(), "non-prefixed entries must be dropped, not silently included"
        assert len(caught) == 1
        assert "missing" in str(caught[0].message)
        assert "run-ci-" in str(caught[0].message)

    def test_mixed_inputs_keep_only_prefixed(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = strip_run_ci_prefix(["run-ci-megatron", "fsdp", "run-ci-short"])
        assert result == {"megatron", "short"}
        assert len(caught) == 1  # only the bare `fsdp` warns

    def test_empty_string_entries_skipped_without_warning(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = strip_run_ci_prefix(["", "run-ci-megatron"])
        assert result == {"megatron"}
        assert len(caught) == 0

    def test_workflow_only_labels_skipped_without_warning(self):
        # `nightly` / `bypass-fastfail` are scope/behavior switches consumed
        # by resolve_scope and pr-test.yml, not malformed domain labels.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = strip_run_ci_prefix(["nightly", "bypass-fastfail", "run-ci-megatron"])
        assert result == {"megatron"}
        assert len(caught) == 0


# --- resolve_scope: broad-scope policy, single unit-tested source ------------


_ALL = set(KNOWN_LABELS)


class TestResolveScope:
    @pytest.mark.parametrize(
        ("event", "labels", "expected"),
        [
            ("pull_request", set(), set()),
            ("pull_request", {"run-ci-megatron"}, {"megatron"}),
            ("pull_request", {"bypass-fastfail"}, set()),
            ("pull_request", {"run-ci-image"}, _ALL - {"ft-short", "ft-long"}),
            ("pull_request", {"nightly"}, _ALL - {"ft-long"}),
            ("pull_request", {"run-ci-all"}, _ALL),
            ("pull_request", {"run-ci-image", "nightly"}, _ALL - {"ft-long"}),
            ("pull_request", {"run-ci-image", "run-ci-all"}, _ALL),
            ("pull_request", {"nightly", "run-ci-all"}, _ALL),
            ("pull_request", {"run-ci-image", "nightly", "run-ci-all"}, _ALL),
            ("schedule", set(), _ALL - {"ft-long"}),
            ("workflow_dispatch", set(), _ALL),
            ("", set(), set()),
            ("", {"run-ci-image"}, _ALL - {"ft-short", "ft-long"}),
        ],
    )
    def test_selection_matches_former_yaml_expressions(self, event, labels, expected):
        # Each case selects exactly the tests the pre-refactor per-stage YAML
        # flags selected for that trigger/label combination (single-label
        # registry; multi-label subtraction semantics pinned separately).
        assert resolve_scope(event, labels) == expected

    @pytest.mark.parametrize(
        ("event", "labels", "expected"),
        [
            ("schedule", {"run-ci-all"}, _ALL),
            ("schedule", {"run-ci-image"}, _ALL - {"ft-long"}),
            ("workflow_dispatch", {"nightly"}, _ALL),
        ],
    )
    def test_counterfactual_event_label_combos_pinned(self, event, labels, expected):
        # Unreachable today (schedule/workflow_dispatch runs carry no PR
        # labels) but pinned as decided behavior: an explicit `run-ci-all`
        # outranks the event, and a dispatch outranks a nightly label. If a
        # future trigger ever passes labels, this is not up for re-derivation.
        assert resolve_scope(event, labels) == expected

    @pytest.mark.parametrize(
        ("event", "labels", "expected"),
        [
            ("pull_request", {"run-ci-image", "run-ci-ft-short"}, _ALL - {"ft-long"}),
            ("pull_request", {"nightly", "run-ci-ft-long"}, _ALL),
            ("pull_request", {"run-ci-image", "run-ci-ft-short", "run-ci-ft-long"}, _ALL),
            ("schedule", {"run-ci-ft-long"}, _ALL),
        ],
    )
    def test_explicit_domain_label_wins_over_scope_subtraction(self, event, labels, expected):
        # Asking for FT coverage on an image bump must not be silently
        # dropped: explicit requests are unioned in after the subtraction.
        assert resolve_scope(event, labels) == expected

    @pytest.mark.parametrize(
        ("event", "labels"),
        [
            ("pull_request", set()),
            ("pull_request", {"run-ci-megatron", "run-ci-typo", "bypass-fastfail"}),
            ("pull_request", {"run-ci-image", "run-ci-ft-short"}),
            ("schedule", set()),
            ("workflow_dispatch", set()),
        ],
    )
    def test_include_set_stays_inside_known_labels(self, event, labels):
        # The include set is drawn from the registry only: scope-label
        # stripping artifacts (`image`, `all`) and typo'd requests must not
        # leak in, and scope subtractions must name real registry labels.
        assert resolve_scope(event, labels) <= _ALL


# --- pr-test.yml seam: stages pass facts, never scope policy -----------------


class TestWorkflowScopeSeam:
    def test_every_stage_passes_event_name_and_no_scope_flags(self):
        workflow = (Path(__file__).resolve().parents[3] / ".github" / "workflows" / "pr-test.yml").read_text()
        commands = workflow.split("execute_command:")[1:]
        assert len(commands) == 7, "stage inventory changed; update this lock test"
        for block in commands:
            cmd = block.split("secrets:")[0]
            assert "--event-name ${{ github.event_name }}" in cmd
            assert "--match-all-labels" not in cmd, "scope policy lives in resolve_scope, not YAML"
            assert "--exclude-labels" not in cmd, "scope policy lives in resolve_scope, not YAML"


# --- discover_ci_files: location-based discovery across the CI roots --------


class TestDiscoverCiFiles:
    def test_only_test_prefixed_files_under_known_roots(self, monkeypatch):
        # discover_ci_files globs repo-relative; anchor cwd to the repo root
        # so it scans the real tree regardless of where pytest is invoked.
        repo_root = Path(__file__).resolve().parents[3]
        monkeypatch.chdir(repo_root)
        files = discover_ci_files()

        roots = ("tests/fast/", "tests/fast-gpu/", "tests/e2e/", "tests/ci/")
        for f in files:
            assert f.startswith(roots), f
            assert Path(f).name.startswith("test_"), f
        # helpers / conftest / __init__ / _common excluded by the glob pattern
        assert not any(Path(f).name in ("conftest.py", "__init__.py") for f in files)
        assert not any(Path(f).name.startswith("_") for f in files)
        # representative files across the roots are discovered
        assert "tests/ci/test/test_ci_register.py" in files
        assert "tests/fast-gpu/test_semaphore.py" in files
        assert "tests/e2e/short/test_dumper.py" in files  # re-enabled, no carve-out


# --- `filter_tests` label-selection scenarios --------------------------------


@pytest.fixture
def cuda_h100_tests():
    """A representative `stage-c-8-gpu-h100` registry used across scenarios.

    Composition:
    * 2 always-run tests (`labels=[]`)
    * 1 megatron-only test
    * 1 fsdp-only test
    * 1 megatron+sglang test (multi-label, exercises OR semantics)
    * 1 disabled megatron test (must always be classified as skipped)
    """
    return [
        _make("tests/e2e/fast1.py", labels=[]),
        _make("tests/e2e/fast2.py", labels=[]),
        _make("tests/e2e/megatron/m1.py", labels=["megatron"]),
        _make("tests/e2e/fsdp/f1.py", labels=["fsdp"]),
        _make("tests/e2e/megatron/m_or_s.py", labels=["megatron", "sglang"]),
        _make("tests/e2e/megatron/disabled.py", labels=["megatron"], disabled="known flaky"),
    ]


def _names(tests: list[CIRegistry]) -> set[str]:
    return {t.filename for t in tests}


class TestFilterTestsLabels:
    def test_case1_no_labels_keeps_only_always_run(self, cuda_h100_tests):
        # Empty --labels (after stripping) -> tests with empty `labels`
        # survive (always run); labelled tests are filtered out.
        enabled, skipped = filter_tests(
            cuda_h100_tests,
            HWBackend.CUDA,
            "stage-c-8-gpu-h100",
            labels=set(),
        )
        assert _names(enabled) == {"tests/e2e/fast1.py", "tests/e2e/fast2.py"}
        assert skipped == []

    def test_case2_single_domain_label(self, cuda_h100_tests):
        # `run-ci-megatron` -> always-run + megatron-labelled tests.
        enabled, skipped = filter_tests(
            cuda_h100_tests,
            HWBackend.CUDA,
            "stage-c-8-gpu-h100",
            labels={"megatron"},
        )
        assert _names(enabled) == {
            "tests/e2e/fast1.py",
            "tests/e2e/fast2.py",
            "tests/e2e/megatron/m1.py",
            "tests/e2e/megatron/m_or_s.py",
        }
        # `disabled.py` matches the megatron label but is disabled, so it
        # belongs to the skipped bucket.
        assert _names(skipped) == {"tests/e2e/megatron/disabled.py"}

    def test_case3_multiple_domain_labels_or_semantics(self, cuda_h100_tests):
        # {megatron, fsdp} -> union (OR) of matches plus always-run tests.
        enabled, _ = filter_tests(
            cuda_h100_tests,
            HWBackend.CUDA,
            "stage-c-8-gpu-h100",
            labels={"megatron", "fsdp"},
        )
        assert _names(enabled) == {
            "tests/e2e/fast1.py",
            "tests/e2e/fast2.py",
            "tests/e2e/megatron/m1.py",
            "tests/e2e/fsdp/f1.py",
            "tests/e2e/megatron/m_or_s.py",
        }

    def test_case4_full_include_set_runs_everything_in_suite(self, cuda_h100_tests):
        # The full registry as include set (run-ci-all / workflow_dispatch /
        # --match-all-labels): every enabled hw/suite/nightly match runs.
        enabled, skipped = filter_tests(
            cuda_h100_tests,
            HWBackend.CUDA,
            "stage-c-8-gpu-h100",
            labels=_ALL,
        )
        assert _names(enabled) == {
            "tests/e2e/fast1.py",
            "tests/e2e/fast2.py",
            "tests/e2e/megatron/m1.py",
            "tests/e2e/fsdp/f1.py",
            "tests/e2e/megatron/m_or_s.py",
        }
        assert _names(skipped) == {"tests/e2e/megatron/disabled.py"}

    def test_case5_unknown_pr_side_label_is_silent_noop(self, cuda_h100_tests):
        # Unknown PR-side label (e.g. `run-ci-foo`) -- after stripping,
        # `foo` simply produces an empty intersection. No error; only
        # always-run tests survive.
        enabled, _ = filter_tests(
            cuda_h100_tests,
            HWBackend.CUDA,
            "stage-c-8-gpu-h100",
            labels={"foo"},
        )
        assert _names(enabled) == {"tests/e2e/fast1.py", "tests/e2e/fast2.py"}


# --- filter_tests: broad CI scopes as include sets ---------------------------


@pytest.fixture
def broad_scope_tests():
    return [
        _make("tests/e2e/always.py", labels=[]),
        _make("tests/e2e/megatron.py", labels=["megatron"]),
        _make("tests/e2e/ft/short.py", labels=["ft-short"]),
        _make("tests/e2e/ft/long.py", labels=["ft-long", "long"]),
    ]


class TestFilterTestsBroadScopes:
    def test_image_scope_selects_no_ft_only_tests(self, broad_scope_tests):
        enabled, _ = filter_tests(
            broad_scope_tests,
            HWBackend.CUDA,
            "stage-c-8-gpu-h100",
            labels=resolve_scope("pull_request", {"run-ci-image"}),
        )
        # ft/long.py still runs: its `long` label is in the include set. A
        # subtraction is not a per-test veto (maintainer-decided semantics).
        assert _names(enabled) == {
            "tests/e2e/always.py",
            "tests/e2e/megatron.py",
            "tests/e2e/ft/long.py",
        }

    def test_nightly_scope_selects_ft_short_but_not_ft_only_soak(self, broad_scope_tests):
        enabled, _ = filter_tests(
            broad_scope_tests,
            HWBackend.CUDA,
            "stage-c-8-gpu-h100",
            labels=resolve_scope("schedule", set()),
        )
        # ft/long.py again enters via `long`; a soak test that must never
        # run at nightly must carry only FT labels.
        assert _names(enabled) == {
            "tests/e2e/always.py",
            "tests/e2e/megatron.py",
            "tests/e2e/ft/short.py",
            "tests/e2e/ft/long.py",
        }

    def test_subtracted_only_test_drops_out_entirely(self):
        tests = [
            _make("tests/e2e/always.py", labels=[]),
            _make("tests/e2e/ft/soak.py", labels=["ft-long"]),
            _make("tests/e2e/ft/soak_disabled.py", labels=["ft-long"], disabled="flaky"),
        ]
        enabled, skipped = filter_tests(
            tests,
            HWBackend.CUDA,
            "stage-c-8-gpu-h100",
            labels=resolve_scope("schedule", set()),
        )
        # A test whose only labels were subtracted is out of scope entirely,
        # including from the skip report.
        assert _names(enabled) == {"tests/e2e/always.py"}
        assert skipped == []

    def test_all_scope_includes_every_label(self, broad_scope_tests):
        enabled, _ = filter_tests(
            broad_scope_tests,
            HWBackend.CUDA,
            "stage-c-8-gpu-h100",
            labels=resolve_scope("pull_request", {"run-ci-all"}),
        )
        assert _names(enabled) == _names(broad_scope_tests)


# --- filter_tests: hw/suite/nightly partitioning still works ----------------


class TestFilterTestsBaseDimensions:
    def test_cross_suite_isolation(self):
        # A test registered to stage-c-4-gpu-h200 must not surface in
        # stage-c-8-gpu-h100, even with the full include set.
        tests = [
            _make("tests/e2e/h100/t.py", suite="stage-c-8-gpu-h100", labels=[]),
            _make("tests/e2e/h200/t.py", suite="stage-c-4-gpu-h200", labels=[]),
        ]
        enabled, _ = filter_tests(
            tests,
            HWBackend.CUDA,
            "stage-c-8-gpu-h100",
            labels=_ALL,
        )
        assert _names(enabled) == {"tests/e2e/h100/t.py"}

    def test_cross_backend_isolation(self):
        # CPU suite must not pull in CUDA-registered always-run tests.
        tests = [
            _make("tests/fast/t.py", backend=HWBackend.CPU, suite="stage-a-cpu", labels=[]),
            _make("tests/e2e/h100/t.py", backend=HWBackend.CUDA, suite="stage-c-8-gpu-h100", labels=[]),
        ]
        enabled, _ = filter_tests(
            tests,
            HWBackend.CPU,
            "stage-a-cpu",
            labels=set(),
        )
        assert _names(enabled) == {"tests/fast/t.py"}

    def test_nightly_dimension_respected(self):
        tests = [
            _make("tests/e2e/per_commit.py", labels=["megatron"], nightly=False),
            _make("tests/e2e/nightly.py", labels=["megatron"], nightly=True),
        ]
        enabled, _ = filter_tests(
            tests,
            HWBackend.CUDA,
            "stage-c-8-gpu-h100",
            nightly=False,
            labels={"megatron"},
        )
        assert _names(enabled) == {"tests/e2e/per_commit.py"}

    def test_stage_b_2_gpu_h200_is_addressable(self):
        # The always-run GPU bucket must be a first-class suite that
        # filter_tests can route to without a "unknown suite" warning fail.
        tests = [
            _make("tests/fast/q.py", suite="stage-b-2-gpu-h200", labels=[]),
        ]
        enabled, _ = filter_tests(
            tests,
            HWBackend.CUDA,
            "stage-b-2-gpu-h200",
            labels=set(),
        )
        assert _names(enabled) == {"tests/fast/q.py"}
