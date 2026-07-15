"""Unit tests for `run_suite.py`.

These cover the Python-side policy and label pipeline:

* `strip_run_ci_prefix`: empty input, prefix stripping, silent skip of
  workflow-only labels, warning on other non-prefixed inputs.
* `resolve_policy`: explicit cadence + raw labels -> selection and fast-fail.
* The pr-test.yml seam: one adapter resolves trigger facts and every stage
  consumes its outputs.
* `filter_tests`: include-set selection with the "empty labels means always
  run" semantic; a scope subtraction is not a per-test veto.
* `CI_SUITES`: locked to the new taxonomy including the
  always-run GPU bucket `stage-b-2-gpu-h200`.

We build `CIRegistry` instances directly via a small factory rather than
parsing fixture files -- the AST-side validation lives in
`test_ci_register.py`; this module exercises the runtime filter.
"""

import os
import re
import subprocess
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import pytest
import tests.ci.run_suite as run_suite_module
from tests.ci.ci_policy import NIGHTLY_CADENCE, REGULAR_CADENCE, SCHEDULE_POLICIES, resolve_policy, strip_run_ci_prefix
from tests.ci.ci_register import CIRegistry, HWBackend, discover_ci_files, register_cpu_ci
from tests.ci.labels import KNOWN_LABELS
from tests.ci.run_suite import CI_SUITES, build_cpu_pytest_cmd, filter_tests

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
        # A regular run stops at the first failure by default.
        cmd = build_cpu_pytest_cmd(["tests/fast/a.py", "tests/fast/b.py"], continue_on_error=False)
        assert "-x" in cmd

    def test_x_dropped_on_continue_on_error(self):
        # bypass-fastfail passes --continue-on-error -> run every file to the end.
        cmd = build_cpu_pytest_cmd(["tests/fast/a.py", "tests/fast/b.py"], continue_on_error=True)
        assert "-x" not in cmd
        assert cmd[0] == "pytest"
        assert "tests/fast/a.py" in cmd and "tests/fast/b.py" in cmd


# --- CI_SUITES locked to the stage taxonomy ---------------------------------


class TestCISuites:
    def test_cpu_suites_exact(self):
        assert CI_SUITES[HWBackend.CPU] == ["stage-a-cpu", "stage-b-cpu"]

    def test_cuda_suites_exact(self):
        assert CI_SUITES[HWBackend.CUDA] == [
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
        all_suites = {s for suites in CI_SUITES.values() for s in suites}
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
        # `nightly` / `bypass-fastfail` are cadence/behavior switches consumed
        # by the resolved policy, not malformed domain labels.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = strip_run_ci_prefix(["nightly", "bypass-fastfail", "run-ci-megatron"])
        assert result == {"megatron"}
        assert len(caught) == 0


# --- resolve_policy: explicit cadence, scope, and fast-fail ------------------


_ALL = set(KNOWN_LABELS)


class TestResolvePolicy:
    @pytest.mark.parametrize(
        ("cadence", "labels", "expected", "bypass"),
        [
            (REGULAR_CADENCE, set(), set(), False),
            (REGULAR_CADENCE, {"run-ci-megatron"}, {"megatron"}, False),
            (REGULAR_CADENCE, {"bypass-fastfail"}, set(), True),
            (REGULAR_CADENCE, {"run-ci-image"}, _ALL - {"ft-short", "ft-long"}, False),
            (REGULAR_CADENCE, {"run-ci-all"}, _ALL, False),
            (REGULAR_CADENCE, {"run-ci-image", "run-ci-all"}, _ALL, False),
            (NIGHTLY_CADENCE, set(), _ALL - {"ft-long"}, True),
            (NIGHTLY_CADENCE, {"nightly"}, _ALL - {"ft-long"}, True),
            (NIGHTLY_CADENCE, {"run-ci-image", "nightly"}, _ALL - {"ft-long"}, True),
            (NIGHTLY_CADENCE, {"nightly", "run-ci-all"}, _ALL, True),
        ],
    )
    def test_selection_and_fastfail(self, cadence, labels, expected, bypass):
        policy = resolve_policy(cadence, labels)
        assert policy.cadence == cadence
        assert policy.include_labels == expected
        assert policy.is_nightly is (cadence == NIGHTLY_CADENCE)
        assert policy.bypass_fastfail is bypass

    def test_unknown_cadence_rejected(self):
        with pytest.raises(ValueError, match="Unknown CI cadence 'weekly'"):
            resolve_policy("weekly", set())

    def test_nightly_tag_and_explicit_cadence_converge(self):
        assert resolve_policy(NIGHTLY_CADENCE, {"nightly"}) == resolve_policy(NIGHTLY_CADENCE, set())

    def test_nightly_label_requires_resolved_nightly_cadence(self):
        with pytest.raises(ValueError, match="nightly workflow label"):
            resolve_policy(REGULAR_CADENCE, {"nightly"})

    @pytest.mark.parametrize(
        ("cadence", "labels", "expected"),
        [
            (REGULAR_CADENCE, {"run-ci-image", "run-ci-ft-short"}, _ALL - {"ft-long"}),
            (NIGHTLY_CADENCE, {"nightly", "run-ci-ft-long"}, _ALL),
            (REGULAR_CADENCE, {"run-ci-image", "run-ci-ft-short", "run-ci-ft-long"}, _ALL),
            (NIGHTLY_CADENCE, {"run-ci-ft-long"}, _ALL),
        ],
    )
    def test_explicit_domain_label_wins_over_scope_subtraction(self, cadence, labels, expected):
        # Asking for FT coverage on an image bump must not be silently
        # dropped: explicit requests are unioned in after the subtraction.
        assert resolve_policy(cadence, labels).include_labels == expected

    @pytest.mark.parametrize(
        ("cadence", "labels"),
        [
            (REGULAR_CADENCE, set()),
            (REGULAR_CADENCE, {"run-ci-megatron", "run-ci-typo", "bypass-fastfail"}),
            (REGULAR_CADENCE, {"run-ci-image", "run-ci-ft-short"}),
            (NIGHTLY_CADENCE, set()),
        ],
    )
    def test_include_set_stays_inside_known_labels(self, cadence, labels):
        # The include set is drawn from the registry only: scope-label
        # stripping artifacts (`image`, `all`) and typo'd requests must not
        # leak in, and scope subtractions must name real registry labels.
        assert resolve_policy(cadence, labels).include_labels <= _ALL


# --- pr-test.yml seam: one trigger adapter, shared stage inputs ---------------


class TestWorkflowScopeSeam:
    @staticmethod
    def _workflow() -> str:
        return (Path(__file__).resolve().parents[3] / ".github" / "workflows" / "pr-test.yml").read_text()

    def test_every_stage_consumes_resolved_policy(self):
        workflow = self._workflow()
        commands = workflow.split("execute_command:")[1:]
        assert len(commands) == 7, "stage inventory changed; update this lock test"
        for block in commands:
            cmd = block.split("secrets:")[0]
            assert "--cadence ${{ needs.resolve-ci-policy.outputs.cadence }}" in cmd
            assert "--labels ${{ needs.resolve-ci-policy.outputs.raw_labels }}" in cmd
            assert "--event-name" not in cmd
            assert "--continue-on-error" not in cmd

    def test_both_cpu_stages_require_both_resolvers(self):
        workflow = self._workflow()
        stage_a = workflow.split("  stage-a-cpu:", 1)[1].split("  stage-b-cpu:", 1)[0]
        stage_b = workflow.split("  stage-b-cpu:", 1)[1].split("  stage-b-2-gpu-h200:", 1)[0]

        expected = "needs: [resolve-ci-policy, resolve-ci-image]"
        assert expected in stage_a
        assert expected in stage_b

    def test_policy_job_is_a_thin_python_adapter(self):
        workflow = self._workflow()
        policy_block = workflow.split("resolve-ci-policy:", 1)[1].split("resolve-ci-image:", 1)[0]
        assert "uses: actions/checkout@v4" in policy_block
        assert "persist-credentials: false" in policy_block
        assert "run: python -m tests.ci.ci_policy" in policy_block
        assert "cadence: ${{ steps.resolve.outputs.cadence }}" in policy_block
        assert "raw_labels: ${{ steps.resolve.outputs.raw_labels }}" in policy_block
        assert "bypass_fastfail: ${{ steps.resolve.outputs.bypass_fastfail }}" in policy_block
        assert 'case "$EVENT_NAME"' not in policy_block
        assert "jq " not in policy_block

    def test_policy_job_passes_trigger_facts(self):
        workflow = self._workflow()
        policy_block = workflow.split("resolve-ci-policy:", 1)[1].split("resolve-ci-image:", 1)[0]
        assert "EVENT_NAME: ${{ github.event_name }}" in policy_block
        assert "SCHEDULE: ${{ github.event.schedule || '' }}" in policy_block
        assert "PR_LABELS_JSON: ${{ toJSON(github.event.pull_request.labels.*.name) }}" in policy_block
        assert "join(github.event.pull_request.labels" not in policy_block

    def test_every_configured_cron_has_an_explicit_python_policy(self):
        workflow = self._workflow()
        configured = set(re.findall(r"^\s+- cron: ['\"]([^'\"]+)['\"]\s*$", workflow, flags=re.MULTILINE))
        assert configured == set(SCHEDULE_POLICIES)

    def test_dispatch_has_no_implicit_scope(self):
        workflow = self._workflow()
        dispatch_inputs = workflow.split("workflow_dispatch:", 1)[1].split("permissions:", 1)[0]
        assert "ci_cadence" not in dispatch_inputs
        assert "ci_scope" not in dispatch_inputs

    def test_gpu_gates_consume_shared_bypass_output(self):
        workflow = self._workflow()
        bypass_gate = "needs.resolve-ci-policy.outputs.bypass_fastfail == 'true'"
        assert workflow.count(bypass_gate) == 5
        assert workflow.count("needs.resolve-ci-policy.result == 'success'") == 5
        assert workflow.count("needs.resolve-ci-image.result == 'success'") == 5

    def test_non_pr_concurrency_does_not_collapse_to_ref(self):
        workflow = self._workflow()
        assert "github.event.schedule || github.run_id" in workflow


# --- CLI seam: local nightly alias and invalid-suite exit behavior -----------


class TestRunSuiteCLI:
    @staticmethod
    def _run(*args: str) -> subprocess.CompletedProcess[str]:
        repo_root = Path(__file__).resolve().parents[3]
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(repo_root), env.get("PYTHONPATH"))))
        return subprocess.run(
            [sys.executable, "tests/ci/run_suite.py", *args],
            cwd=repo_root,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_nightly_alias_matches_explicit_cadence(self):
        common = ("--hw", "cpu", "--suite", "stage-b-cpu", "--list-only")
        alias = self._run(*common, "--nightly")
        explicit = self._run(*common, "--cadence", NIGHTLY_CADENCE)

        assert alias.returncode == explicit.returncode == 0
        alias_policy = alias.stdout.splitlines()[0]
        explicit_policy = explicit.stdout.splitlines()[0]
        assert alias_policy == explicit_policy
        assert "cadence='nightly' bypass_fastfail=True" in alias_policy
        assert "'ft-short'" in alias_policy
        assert "'ft-long'" not in alias_policy
        assert "Continue on error: True" in alias.stdout

    def test_nightly_alias_and_explicit_cadence_are_mutually_exclusive(self):
        result = self._run(
            "--hw",
            "cpu",
            "--suite",
            "stage-b-cpu",
            "--nightly",
            "--cadence",
            NIGHTLY_CADENCE,
            "--list-only",
        )
        assert result.returncode == 2
        assert "not allowed with argument" in result.stderr

    def test_unknown_suite_exits_nonzero(self):
        result = self._run(
            "--hw",
            "cuda",
            "--suite",
            "stage-c-unknown",
            "--list-only",
        )
        assert result.returncode != 0
        assert "Unknown suite stage-c-unknown" in result.stderr
        assert "No tests to run. Exiting with success." not in result.stdout


# --- run_a_suite: resolved policy reaches cadence and runner behavior --------


def _run_args(*, hw: str, suite: str, cadence: str, labels: list[str] | None = None):
    return SimpleNamespace(
        hw=hw,
        suite=suite,
        cadence=cadence,
        labels=labels or [],
        match_all_labels=False,
        continue_on_error=False,
        auto_partition_id=None,
        auto_partition_size=None,
        list_only=False,
        timeout_per_file=1800,
        enable_retry=False,
        retry_timeout_increase=600,
        max_attempts=2,
        retry_wait_seconds=60,
    )


class TestRunSuitePolicyIntegration:
    @staticmethod
    def _stub_collection(monkeypatch, tests):
        monkeypatch.setattr(run_suite_module, "discover_ci_files", lambda: [])
        monkeypatch.setattr(run_suite_module, "collect_tests", lambda *_args, **_kwargs: tests)

    def test_regular_all_scope_does_not_unlock_nightly_only(self):
        tests = [
            _make("tests/e2e/regular.py", labels=["megatron"]),
            _make("tests/e2e/nightly.py", labels=["megatron"], nightly=True),
        ]
        policy = resolve_policy(REGULAR_CADENCE, {"run-ci-all"})
        enabled, _ = filter_tests(
            tests,
            HWBackend.CUDA,
            "stage-c-8-gpu-h100",
            nightly=policy.is_nightly,
            labels=set(policy.include_labels),
        )
        assert _names(enabled) == {"tests/e2e/regular.py"}

    def test_nightly_bypass_reaches_cpu_runner(self, monkeypatch):
        tests = [_make("tests/fast/test_regular.py", backend=HWBackend.CPU, suite="stage-a-cpu")]
        self._stub_collection(monkeypatch, tests)
        captured = {}

        def fake_call(cmd):
            captured["cmd"] = cmd
            return 0

        monkeypatch.setattr(run_suite_module.subprocess, "call", fake_call)
        result = run_suite_module.run_a_suite(_run_args(hw="cpu", suite="stage-a-cpu", cadence=NIGHTLY_CADENCE))
        assert result == 0
        assert "-x" not in captured["cmd"]

    def test_nightly_bypass_reaches_cuda_runner(self, monkeypatch):
        tests = [_make("tests/e2e/test_regular.py", suite="stage-c-8-gpu-h100")]
        self._stub_collection(monkeypatch, tests)
        captured = {}

        def fake_run_unittest_files(ci_tests, **kwargs):
            captured.update(kwargs)
            return 0

        monkeypatch.setattr(run_suite_module, "run_unittest_files", fake_run_unittest_files)
        result = run_suite_module.run_a_suite(
            _run_args(hw="cuda", suite="stage-c-8-gpu-h100", cadence=NIGHTLY_CADENCE)
        )
        assert result == 0
        assert captured["continue_on_error"] is True


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
        # The full registry as include set (run-ci-all / --match-all-labels):
        # every enabled hw/suite/cadence match runs.
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
            labels=set(resolve_policy(REGULAR_CADENCE, {"run-ci-image"}).include_labels),
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
            nightly=True,
            labels=set(resolve_policy(NIGHTLY_CADENCE, set()).include_labels),
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
            nightly=True,
            labels=set(resolve_policy(NIGHTLY_CADENCE, set()).include_labels),
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
            labels=set(resolve_policy(REGULAR_CADENCE, {"run-ci-all"}).include_labels),
        )
        assert _names(enabled) == _names(broad_scope_tests)


# --- filter_tests: hw/suite/cadence eligibility ------------------------------


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

    @staticmethod
    def _cadence_tests():
        return [
            _make("tests/e2e/regular.py", labels=["megatron"], nightly=False),
            _make("tests/e2e/nightly.py", labels=["megatron"], nightly=True),
        ]

    def test_regular_run_excludes_nightly_only(self):
        enabled, _ = filter_tests(
            self._cadence_tests(),
            HWBackend.CUDA,
            "stage-c-8-gpu-h100",
            nightly=False,
            labels={"megatron"},
        )
        assert _names(enabled) == {"tests/e2e/regular.py"}

    def test_nightly_run_includes_regular_and_nightly_only(self):
        enabled, _ = filter_tests(
            self._cadence_tests(),
            HWBackend.CUDA,
            "stage-c-8-gpu-h100",
            nightly=True,
            labels={"megatron"},
        )
        assert _names(enabled) == {"tests/e2e/regular.py", "tests/e2e/nightly.py"}

    def test_disabled_nightly_only_is_skipped_only_when_eligible(self):
        tests = [
            _make("tests/e2e/regular.py", labels=["megatron"]),
            _make("tests/e2e/nightly.py", labels=["megatron"], nightly=True, disabled="flaky"),
        ]
        _, regular_skipped = filter_tests(
            tests,
            HWBackend.CUDA,
            "stage-c-8-gpu-h100",
            nightly=False,
            labels={"megatron"},
        )
        _, nightly_skipped = filter_tests(
            tests,
            HWBackend.CUDA,
            "stage-c-8-gpu-h100",
            nightly=True,
            labels={"megatron"},
        )
        assert regular_skipped == []
        assert _names(nightly_skipped) == {"tests/e2e/nightly.py"}

    def test_nightly_only_ft_long_still_obeys_domain_scope(self):
        tests = [_make("tests/e2e/ft/soak.py", labels=["ft-long"], nightly=True)]
        standard_policy = resolve_policy(NIGHTLY_CADENCE, set())
        explicit_policy = resolve_policy(NIGHTLY_CADENCE, {"run-ci-ft-long"})

        standard, _ = filter_tests(
            tests,
            HWBackend.CUDA,
            "stage-c-8-gpu-h100",
            nightly=True,
            labels=set(standard_policy.include_labels),
        )
        explicit, _ = filter_tests(
            tests,
            HWBackend.CUDA,
            "stage-c-8-gpu-h100",
            nightly=True,
            labels=set(explicit_policy.include_labels),
        )
        assert standard == []
        assert _names(explicit) == {"tests/e2e/ft/soak.py"}

    def test_unknown_suite_fails_instead_of_green_empty(self):
        with pytest.raises(ValueError, match="Unknown suite stage-c-unknown"):
            filter_tests([], HWBackend.CUDA, "stage-c-unknown")

    def test_known_empty_suite_is_valid(self):
        enabled, skipped = filter_tests([], HWBackend.CPU, "stage-b-cpu")
        assert enabled == []
        assert skipped == []

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
