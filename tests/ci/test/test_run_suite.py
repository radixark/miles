"""Unit tests for `run_suite.py`.

These cover the Python-side policy and label pipeline:

* `strip_run_ci_prefix`: empty input, prefix stripping, silent skip of
  workflow-only labels, warning on other non-prefixed inputs.
* `resolve_policy`: explicit cadence + raw labels -> selection and fast-fail.
* The PR workflow seams: one adapter resolves trigger facts and every CUDA/ROCm stage consumes its outputs.
* `filter_tests`: include-set selection, including CPU always-on coverage and
  GPU domain labels; a scope subtraction is not a per-test veto.
* `CI_SUITES`: locked to the current hardware taxonomy.

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
from tests.ci.ci_policy import (
    NIGHTLY_CADENCE,
    REGULAR_CADENCE,
    RELEASE_CADENCE,
    SCHEDULE_POLICIES,
    WEEKLY_CADENCE,
    resolve_policy,
    strip_run_ci_prefix,
)
from tests.ci.ci_register import CIRegistry, HWBackend, discover_ci_files, register_cpu_ci
from tests.ci.labels import KNOWN_LABELS
from tests.ci.run_suite import CI_SUITES, build_cpu_pytest_cmd, filter_tests
from tests.ci.stage_selection import PR_GPU_STAGES

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

    CUDA fixtures default to the `megatron` domain; CPU fixtures default to
    the always-on empty label set.
    """
    if labels is None:
        labels = [] if backend == HWBackend.CPU else ["megatron"]
    return CIRegistry(
        backend=backend,
        filename=filename,
        est_time=est_time,
        suite=suite,
        labels=list(labels),
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
            "stage-c-8-gpu-b200",
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
            (REGULAR_CADENCE, {"run-ci-image"}, _ALL - {"long", "ft-short", "ft-long"}, False),
            (REGULAR_CADENCE, {"run-ci-all"}, _ALL, False),
            (REGULAR_CADENCE, {"run-ci-image", "run-ci-all"}, _ALL, False),
            (NIGHTLY_CADENCE, set(), _ALL - {"long", "ft-long"}, True),
            (NIGHTLY_CADENCE, {"nightly"}, _ALL - {"long", "ft-long"}, True),
            (NIGHTLY_CADENCE, {"run-ci-image", "nightly"}, _ALL - {"long", "ft-long"}, True),
            (NIGHTLY_CADENCE, {"nightly", "run-ci-all"}, _ALL, True),
            (WEEKLY_CADENCE, set(), _ALL, True),
            (WEEKLY_CADENCE, {"run-ci-image"}, _ALL, True),
            (RELEASE_CADENCE, set(), _ALL, True),
            (RELEASE_CADENCE, {"run-ci-image"}, _ALL, True),
        ],
    )
    def test_selection_and_fastfail(self, cadence, labels, expected, bypass):
        policy = resolve_policy(cadence, labels)
        assert policy.cadence == cadence
        assert policy.include_labels == expected
        scheduled_cadence = cadence in {NIGHTLY_CADENCE, WEEKLY_CADENCE}
        assert policy.admit_nightly_tests is (scheduled_cadence or cadence == RELEASE_CADENCE)
        assert policy.bypass_fastfail is bypass
        # Release runs weekly's scope, but its frozen dependency SHAs must
        # never write the rolling perf baseline.
        assert policy.write_baseline is scheduled_cadence

    def test_unknown_cadence_rejected(self):
        with pytest.raises(ValueError, match="Unknown CI cadence 'hourly'"):
            resolve_policy("hourly", set())

    def test_nightly_tag_and_explicit_cadence_converge(self):
        assert resolve_policy(NIGHTLY_CADENCE, {"nightly"}) == resolve_policy(NIGHTLY_CADENCE, set())

    def test_nightly_label_requires_resolved_nightly_cadence(self):
        with pytest.raises(ValueError, match="nightly workflow label"):
            resolve_policy(REGULAR_CADENCE, {"nightly"})

    @pytest.mark.parametrize(
        ("cadence", "labels", "expected"),
        [
            (REGULAR_CADENCE, {"run-ci-image", "run-ci-long"}, _ALL - {"ft-short", "ft-long"}),
            (REGULAR_CADENCE, {"run-ci-image", "run-ci-ft-short"}, _ALL - {"long", "ft-long"}),
            (NIGHTLY_CADENCE, {"nightly", "run-ci-long"}, _ALL - {"ft-long"}),
            (NIGHTLY_CADENCE, {"nightly", "run-ci-ft-long"}, _ALL - {"long"}),
            (REGULAR_CADENCE, {"run-ci-image", "run-ci-ft-short", "run-ci-ft-long"}, _ALL - {"long"}),
            (NIGHTLY_CADENCE, {"run-ci-ft-long"}, _ALL - {"long"}),
        ],
    )
    def test_explicit_domain_label_wins_over_scope_subtraction(self, cadence, labels, expected):
        assert resolve_policy(cadence, labels).include_labels == expected

    @pytest.mark.parametrize(
        ("cadence", "labels"),
        [
            (REGULAR_CADENCE, set()),
            (REGULAR_CADENCE, {"run-ci-megatron", "run-ci-typo", "bypass-fastfail"}),
            (REGULAR_CADENCE, {"run-ci-image", "run-ci-ft-short"}),
            (NIGHTLY_CADENCE, set()),
            (WEEKLY_CADENCE, set()),
            (RELEASE_CADENCE, set()),
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

    @staticmethod
    def _reusable_workflow(name: str) -> str:
        return (Path(__file__).resolve().parents[3] / ".github" / "workflows" / name).read_text()

    def test_every_stage_consumes_resolved_policy(self):
        workflow = self._workflow()
        commands = workflow.split("execute_command:")[1:]
        assert len(commands) == 8, "stage inventory changed; update this lock test"
        for block in commands:
            cmd = block.split("secrets:")[0]
            assert "--cadence ${{ needs.resolve-ci-policy.outputs.cadence }}" in cmd
            assert "--labels ${{ needs.resolve-ci-policy.outputs.raw_labels }}" in cmd
            assert "--event-name" not in cmd
            assert "--continue-on-error" not in cmd

    def test_cpu_stages_only_require_policy(self):
        workflow = self._workflow()
        stage_a = workflow.split("  stage-a-cpu:", 1)[1].split("  stage-b-cpu:", 1)[0]
        stage_b = workflow.split("  stage-b-cpu:", 1)[1].split("  stage-b-2-gpu-h200:", 1)[0]

        expected = "needs: [resolve-ci-policy]"
        assert expected in stage_a
        assert expected in stage_b
        assert "resolve-ci-image" not in stage_a
        assert "resolve-ci-image" not in stage_b

    def test_cpu_and_gpu_stages_use_dedicated_reusable_workflows(self):
        workflow = self._workflow()
        assert workflow.count("uses: ./.github/workflows/_run-cpu-ci.yml") == 2
        assert workflow.count("uses: ./.github/workflows/_run-ci.yml") == 6
        assert workflow.count("uses: ./.github/workflows/_build-pr-ci-image.yml") == 1
        assert "cpu_runner" not in workflow

        gpu_workflow = self._reusable_workflow("_run-ci.yml")
        cpu_workflow = self._reusable_workflow("_run-cpu-ci.yml")
        docker_workflow = self._reusable_workflow("_build-pr-ci-image.yml")
        job_id_pattern = r"^  ([A-Za-z_][A-Za-z0-9_-]*):$"
        gpu_jobs = re.findall(job_id_pattern, gpu_workflow.split("\njobs:\n", 1)[1], re.MULTILINE)
        cpu_jobs = re.findall(job_id_pattern, cpu_workflow.split("\njobs:\n", 1)[1], re.MULTILINE)
        docker_jobs = re.findall(job_id_pattern, docker_workflow.split("\njobs:\n", 1)[1], re.MULTILINE)
        assert gpu_jobs == ["run"]
        assert cpu_jobs == ["run-cpu"]
        assert docker_jobs == ["docker-decide", "docker-build"]
        assert "cpu_runner" not in gpu_workflow
        assert "cpu_runner" not in cpu_workflow

    def test_docker_build_waits_for_cpu_gate_and_preserves_bypass(self):
        workflow = self._workflow()
        caller = workflow.split("  docker-build:", 1)[1].split("  resolve-ci-image:", 1)[0]

        assert "needs: [resolve-ci-policy, stage-a-cpu]" in caller
        assert "always() && !cancelled()" in caller
        assert "github.event.action != 'closed'" in caller
        assert "needs.resolve-ci-policy.result == 'success'" in caller
        assert "needs.stage-a-cpu.result == 'success'" in caller
        assert "needs.resolve-ci-policy.outputs.bypass_fastfail == 'true'" in caller
        assert "needs.stage-a-cpu.result == 'failure'" not in caller
        assert "stage-b-cpu" not in caller
        assert "uses: ./.github/workflows/_build-pr-ci-image.yml" in caller
        assert "secrets: inherit" in caller

    def test_docker_build_body_lives_in_reusable_workflow(self):
        workflow = self._workflow()
        reusable = self._reusable_workflow("_build-pr-ci-image.yml")

        assert "  docker-decide:" not in workflow
        assert "value: ${{ jobs.docker-build.outputs.built }}" in reusable
        assert "value: ${{ jobs.docker-build.outputs.tag_available }}" in reusable
        assert "needs: [docker-decide]" in reusable
        # Rebuilds follow the build inputs, not whether the PR diff touched them.
        assert "python3 docker/image_inputs.py --rev HEAD^1" in reusable
        assert "python3 docker/image_inputs.py --read-label" in reusable
        assert "github.event.pull_request.head.repo.full_name == github.repository" in reusable
        assert "python3 docker/build.py --variant cu13 --image-tag custom" in reusable

    def test_docker_decision_logs_in_only_for_tag_inspection(self):
        reusable = self._reusable_workflow("_build-pr-ci-image.yml")
        decide_job = reusable.split("  docker-decide:", 1)[1].split("  docker-build:", 1)[0]

        compare = decide_job.index("python3 docker/image_inputs.py --rev HEAD^1")
        login = decide_job.index("- name: Login to Docker Hub")
        inspect = decide_job.index("docker buildx imagetools inspect")
        assert compare < login < inspect
        assert "if: steps.prepare.outputs.inspect_tag == 'true'" in decide_job

    def test_docker_force_rebuild_uses_live_label_for_decision_and_consumption(self):
        reusable = self._reusable_workflow("_build-pr-ci-image.yml")
        decide_job = reusable.split("  docker-decide:", 1)[1].split("  docker-build:", 1)[0]

        live_labels = '"repos/${GITHUB_REPOSITORY}/issues/${PR_NUMBER}/labels"'
        assert live_labels in decide_job
        assert decide_job.index(live_labels) < decide_job.index("- name: Login to Docker Hub")
        assert "LABELS=$(gh api --paginate" in decide_job
        assert 'grep -Fxq "rebuild-ci-image" <<< "$LABELS"' in decide_job
        assert "github.event.pull_request.labels.*.name" not in reusable
        assert "needs.docker-decide.outputs.force_rebuild == 'true'" in reusable

    def test_policy_job_is_a_thin_python_adapter(self):
        workflow = self._workflow()
        policy_block = workflow.split("resolve-ci-policy:", 1)[1].split("  docker-build:", 1)[0]
        assert "uses: actions/checkout@v4" in policy_block
        assert "persist-credentials: false" in policy_block
        assert "run: python -m tests.ci.ci_policy" in policy_block
        assert "cadence: ${{ steps.resolve.outputs.cadence }}" in policy_block
        assert "raw_labels: ${{ steps.resolve.outputs.raw_labels }}" in policy_block
        assert "bypass_fastfail: ${{ steps.resolve.outputs.bypass_fastfail }}" in policy_block
        assert "skipped_stages: ${{ steps.resolve.outputs.skipped_stages }}" in policy_block
        assert "fetch-depth: 2" in policy_block
        assert "git diff --name-status -z -M HEAD^1 HEAD" in policy_block
        assert "CHANGED_FILES_PATH: ${{ runner.temp }}/changed-files.z" in policy_block
        assert 'case "$EVENT_NAME"' not in policy_block
        assert "jq " not in policy_block

    def test_policy_job_passes_trigger_facts(self):
        workflow = self._workflow()
        policy_block = workflow.split("resolve-ci-policy:", 1)[1].split("  docker-build:", 1)[0]
        assert "EVENT_NAME: ${{ github.event_name }}" in policy_block
        assert "SCHEDULE: ${{ github.event.schedule || '' }}" in policy_block
        assert "PR_LABELS_JSON: ${{ toJSON(github.event.pull_request.labels.*.name) }}" in policy_block
        assert "join(github.event.pull_request.labels" not in policy_block

    def test_every_configured_cron_has_an_explicit_python_policy(self):
        workflow = self._workflow()
        configured = set(re.findall(r"^\s+- cron: ['\"]([^'\"]+)['\"]\s*$", workflow, flags=re.MULTILINE))
        assert configured == set(SCHEDULE_POLICIES)

    def test_scheduled_runs_use_utc_1500(self):
        workflow = self._workflow()
        assert "    - cron: '0 15 * * 0-5'" in workflow
        assert "    - cron: '0 15 * * 6'" in workflow
        assert "timezone:" not in workflow

    def test_weekly_serializes_each_gpu_matrix(self):
        workflow = self._workflow()
        normal_parallelism = {
            "stage-c-8-gpu-h100": 2,
            "stage-c-8-gpu-h200": 2,
            "stage-c-4-gpu-h200": 3,
            "stage-c-2-gpu-h200": 2,
        }
        for job, default in normal_parallelism.items():
            block = workflow.split(f"  {job}:", 1)[1]
            block = re.split(r"^  [A-Za-z_][A-Za-z0-9_-]*:\s*$", block, maxsplit=1, flags=re.MULTILINE)[0]
            expected = (
                "max-parallel: ${{ needs.resolve-ci-policy.outputs.cadence == 'weekly' " f"&& 1 || {default} }}}}"
            )
            assert expected in block

    def test_dispatch_has_no_scope_input_but_runs_all_cuda_domains(self):
        workflow = self._workflow()
        dispatch_inputs = workflow.split("workflow_dispatch:", 1)[1].split("permissions:", 1)[0]
        assert "ci_cadence" not in dispatch_inputs
        assert "ci_scope" not in dispatch_inputs
        manual_scope = "${{ github.event_name == 'workflow_dispatch' && '--match-all-labels' || '' }}"
        cuda_stages = workflow.split("  stage-b-2-gpu-h200:", 1)[1]
        assert cuda_stages.count(manual_scope) == 6

    def test_gpu_gates_consume_shared_bypass_output(self):
        workflow = self._workflow()
        gpu_stages = workflow.split("  stage-b-2-gpu-h200:", 1)[1]
        bypass_gate = "needs.resolve-ci-policy.outputs.bypass_fastfail == 'true'"
        assert gpu_stages.count(bypass_gate) == 6
        assert gpu_stages.count("needs.resolve-ci-policy.result == 'success'") == 6
        assert gpu_stages.count("needs.resolve-ci-image.result == 'success'") == 6
        assert "needs.stage-a-cpu.result == 'failure'" not in gpu_stages

    def test_each_cuda_stage_consumes_the_fail_open_skip_list(self):
        workflow = self._workflow()
        cuda_stages = PR_GPU_STAGES - {"stage-c-4-gpu-mi350"}
        for stage_name in cuda_stages:
            block = workflow.split(f"  {stage_name}:", 1)[1]
            block = re.split(r"^  [A-Za-z_][A-Za-z0-9_-]*:\s*$", block, maxsplit=1, flags=re.MULTILINE)[0]
            expected = f"!contains(fromJSON(needs.resolve-ci-policy.outputs.skipped_stages || '[]'), '{stage_name}')"
            assert expected in block

        assert workflow.count("outputs.skipped_stages || '[]'") == len(cuda_stages)

    def test_non_pr_concurrency_does_not_collapse_to_ref(self):
        workflow = self._workflow()
        # schedule outranks the workflow_call-only inputs.ref segment, so cron
        # runs never share a ref-derived group; plain dispatches (no ref
        # input) fall through to run_id.
        assert "github.event.schedule || inputs.ref || github.run_id" in workflow

    def test_closed_pr_only_cancels_existing_run(self):
        workflow = self._workflow()
        assert "types: [opened, synchronize, reopened, ready_for_review, labeled, closed]" in workflow
        assert (
            "group: pr-test-${{ github.event.number || github.event.schedule || inputs.ref || github.run_id }}"
            in workflow
        )

        policy_header = workflow.split("  resolve-ci-policy:", 1)[1].split("    runs-on:", 1)[0]
        docker_caller = workflow.split("  docker-build:", 1)[1].split("  resolve-ci-image:", 1)[0]
        reusable = self._reusable_workflow("_build-pr-ci-image.yml")

        assert "github.event.action != 'closed'" in policy_header
        assert "github.event.action != 'closed'" in docker_caller
        assert reusable.count("github.event.action != 'closed'") == 2


class TestRocmWorkflowScopeSeam:
    @staticmethod
    def _workflow() -> str:
        return (Path(__file__).resolve().parents[3] / ".github" / "workflows" / "pr-test-rocm.yml").read_text()

    def test_pr_schedules_and_dispatch_share_policy(self):
        workflow = self._workflow()
        assert "pull_request:\n    types: [opened, synchronize, reopened, ready_for_review, labeled]" in workflow
        assert "pull_request_target:" not in workflow
        configured = set(re.findall(r"^\s+- cron: ['\"]([^'\"]+)['\"]\s*$", workflow, flags=re.MULTILINE))
        assert configured == set(SCHEDULE_POLICIES)
        assert "    - cron: '0 15 * * 0-5'" in workflow
        assert "    - cron: '0 15 * * 6'" in workflow
        assert "timezone:" not in workflow

        policy_block = workflow.split("resolve-ci-policy:", 1)[1].split("resolve-ci-image:", 1)[0]
        assert "allow_self_hosted" not in policy_block
        assert "EVENT_NAME: ${{ github.event_name }}" in policy_block
        assert "SCHEDULE: ${{ github.event.schedule || '' }}" in policy_block
        assert "PR_LABELS_JSON: ${{ toJSON(github.event.pull_request.labels.*.name) }}" in policy_block
        assert "CHANGED_FILES_PATH: ${{ runner.temp }}/changed-files.z" in policy_block
        assert "git diff --name-status -z -M HEAD^1 HEAD" in policy_block
        assert "skipped_stages: ${{ steps.resolve.outputs.skipped_stages }}" in policy_block
        assert "run: python -m tests.ci.ci_policy" in policy_block
        assert "github.event.schedule || inputs.ref || github.run_id" in workflow

    def test_stage_consumes_policy_and_preserves_manual_full_scope(self):
        workflow = self._workflow()
        stage = workflow.split("  stage-c-4-gpu-mi350:", 1)[1]
        command = stage.split("execute_command:", 1)[1].split("secrets:", 1)[0]

        assert "needs: [resolve-ci-policy, resolve-ci-image, resolve-ci-deps]" in stage
        assert "allow_self_hosted" not in stage
        assert "partition_id: [0, 1]" in stage
        assert "max-parallel: ${{ needs.resolve-ci-policy.outputs.cadence == 'weekly' && 1 || 2 }}" in stage
        assert "--auto-partition-size 2" in command
        assert "checkout_ref:" not in stage
        assert "--cadence ${{ needs.resolve-ci-policy.outputs.cadence }}" in command
        assert "--labels ${{ needs.resolve-ci-policy.outputs.raw_labels }}" in command
        assert "${{ github.event_name == 'workflow_dispatch' && '--match-all-labels' || '' }}" in command
        assert "WANDB_API_KEY: ${{ secrets.WANDB_API_KEY }}" in stage
        assert "needs.resolve-ci-policy.result == 'success'" in stage
        assert "needs.resolve-ci-image.result == 'success'" in stage
        assert "needs.resolve-ci-deps.result == 'success'" in stage
        assert (
            "skip_dependency_install: ${{ needs.resolve-ci-deps.outputs.skip_dependency_install == 'true' }}" in stage
        )
        assert (
            "!contains(fromJSON(needs.resolve-ci-policy.outputs.skipped_stages || '[]'), 'stage-c-4-gpu-mi350')"
            in stage
        )
        assert "--labels amd" not in command
        assert "if: github.event_name == 'workflow_dispatch'" not in workflow

        reusable = (Path(__file__).resolve().parents[3] / ".github" / "workflows" / "_run-ci-rocm.yml").read_text()
        assert "checkout_ref:" not in reusable
        assert "persist-credentials: false" in reusable
        assert "allow-unsafe-pr-checkout" not in reusable
        assert "MILES_HARDWARE_PLATFORM: rocm" in reusable

    def test_megatron_override_installs_the_checked_out_ref_unpatched(self):
        reusable = (Path(__file__).resolve().parents[3] / ".github" / "workflows" / "_run-ci-rocm.yml").read_text()
        override = reusable.split('if [ -n "$MEGATRON_PR" ]; then', 1)[1].split("          cd $GITHUB_WORKSPACE", 1)[0]

        checkout = override.index("git checkout -f FETCH_HEAD")
        install = override.index("pip install -e . --no-deps --break-system-packages")

        assert checkout < install
        assert "amd_patch" not in override


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
        assert "'long'" not in alias_policy
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
            admit_nightly_tests=policy.admit_nightly_tests,
            labels=set(policy.include_labels),
        )
        assert _names(enabled) == {"tests/e2e/regular.py"}

    def test_weekly_full_scope_admits_nightly_only_tests(self):
        tests = [
            _make("tests/e2e/regular.py", labels=["long"]),
            _make("tests/e2e/nightly.py", labels=["ft-long"], nightly=True),
        ]
        policy = resolve_policy(WEEKLY_CADENCE, set())
        enabled, _ = filter_tests(
            tests,
            HWBackend.CUDA,
            "stage-c-8-gpu-h100",
            admit_nightly_tests=policy.admit_nightly_tests,
            labels=set(policy.include_labels),
        )
        assert _names(enabled) == {"tests/e2e/regular.py", "tests/e2e/nightly.py"}

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
        assert captured["gate_write_baseline"] is True

    def test_weekly_policy_reaches_cuda_runner(self, monkeypatch):
        tests = [_make("tests/e2e/test_weekly.py", suite="stage-c-8-gpu-h100", labels=["long"])]
        self._stub_collection(monkeypatch, tests)
        captured = {}

        def fake_run_unittest_files(ci_tests, **kwargs):
            captured["tests"] = ci_tests
            captured.update(kwargs)
            return 0

        monkeypatch.setattr(run_suite_module, "run_unittest_files", fake_run_unittest_files)
        result = run_suite_module.run_a_suite(_run_args(hw="cuda", suite="stage-c-8-gpu-h100", cadence=WEEKLY_CADENCE))
        assert result == 0
        assert _names(captured["tests"]) == {"tests/e2e/test_weekly.py"}
        assert captured["continue_on_error"] is True
        assert captured["gate_write_baseline"] is True


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
    * 2 precision tests
    * 1 megatron-only test
    * 1 fsdp-only test
    * 1 megatron+sglang test (multi-label, exercises OR semantics)
    * 1 disabled megatron test (must always be classified as skipped)
    """
    return [
        _make("tests/e2e/precision1.py", labels=["precision"]),
        _make("tests/e2e/precision2.py", labels=["precision"]),
        _make("tests/e2e/megatron/m1.py", labels=["megatron"]),
        _make("tests/e2e/fsdp/f1.py", labels=["fsdp"]),
        _make("tests/e2e/megatron/m_or_s.py", labels=["megatron", "sglang"]),
        _make("tests/e2e/megatron/disabled.py", labels=["megatron"], disabled="known flaky"),
    ]


def _names(tests: list[CIRegistry]) -> set[str]:
    return {t.filename for t in tests}


class TestFilterTestsLabels:
    def test_case1_no_labels_selects_no_gpu_tests(self, cuda_h100_tests):
        # Every GPU registration has a domain, so an empty include set selects
        # no GPU tests.
        enabled, skipped = filter_tests(
            cuda_h100_tests,
            HWBackend.CUDA,
            "stage-c-8-gpu-h100",
            labels=set(),
        )
        assert enabled == []
        assert skipped == []

    def test_case2_single_domain_label(self, cuda_h100_tests):
        # `run-ci-megatron` selects only megatron-labelled tests.
        enabled, skipped = filter_tests(
            cuda_h100_tests,
            HWBackend.CUDA,
            "stage-c-8-gpu-h100",
            labels={"megatron"},
        )
        assert _names(enabled) == {
            "tests/e2e/megatron/m1.py",
            "tests/e2e/megatron/m_or_s.py",
        }
        # `disabled.py` matches the megatron label but is disabled, so it
        # belongs to the skipped bucket.
        assert _names(skipped) == {"tests/e2e/megatron/disabled.py"}

    def test_case3_multiple_domain_labels_or_semantics(self, cuda_h100_tests):
        # {megatron, fsdp} -> union (OR) of domain matches.
        enabled, _ = filter_tests(
            cuda_h100_tests,
            HWBackend.CUDA,
            "stage-c-8-gpu-h100",
            labels={"megatron", "fsdp"},
        )
        assert _names(enabled) == {
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
            "tests/e2e/precision1.py",
            "tests/e2e/precision2.py",
            "tests/e2e/megatron/m1.py",
            "tests/e2e/fsdp/f1.py",
            "tests/e2e/megatron/m_or_s.py",
        }
        assert _names(skipped) == {"tests/e2e/megatron/disabled.py"}

    def test_case5_unknown_pr_side_label_is_silent_noop(self, cuda_h100_tests):
        # Unknown PR-side label (e.g. `run-ci-foo`) produces an empty
        # intersection and selects no GPU tests.
        enabled, _ = filter_tests(
            cuda_h100_tests,
            HWBackend.CUDA,
            "stage-c-8-gpu-h100",
            labels={"foo"},
        )
        assert enabled == []


# --- filter_tests: broad CI scopes as include sets ---------------------------


@pytest.fixture
def broad_scope_tests():
    return [
        _make("tests/e2e/precision.py", labels=["precision"]),
        _make("tests/e2e/megatron.py", labels=["megatron"]),
        _make("tests/e2e/long.py", labels=["long"]),
        _make("tests/e2e/ft/short.py", labels=["ft-short"]),
        _make("tests/e2e/ft/long.py", labels=["ft-long", "long"]),
    ]


class TestFilterTestsBroadScopes:
    def test_image_scope_excludes_long_and_ft_tests(self, broad_scope_tests):
        enabled, _ = filter_tests(
            broad_scope_tests,
            HWBackend.CUDA,
            "stage-c-8-gpu-h100",
            labels=set(resolve_policy(REGULAR_CADENCE, {"run-ci-image"}).include_labels),
        )
        assert _names(enabled) == {
            "tests/e2e/precision.py",
            "tests/e2e/megatron.py",
        }

    def test_nightly_scope_excludes_long_and_ft_long_but_selects_ft_short(self, broad_scope_tests):
        enabled, _ = filter_tests(
            broad_scope_tests,
            HWBackend.CUDA,
            "stage-c-8-gpu-h100",
            admit_nightly_tests=True,
            labels=set(resolve_policy(NIGHTLY_CADENCE, set()).include_labels),
        )
        assert _names(enabled) == {
            "tests/e2e/precision.py",
            "tests/e2e/megatron.py",
            "tests/e2e/ft/short.py",
        }

    def test_subtracted_only_test_drops_out_entirely(self):
        tests = [
            _make("tests/e2e/precision.py", labels=["precision"]),
            _make("tests/e2e/ft/soak.py", labels=["ft-long"]),
            _make("tests/e2e/ft/soak_disabled.py", labels=["ft-long"], disabled="flaky"),
        ]
        enabled, skipped = filter_tests(
            tests,
            HWBackend.CUDA,
            "stage-c-8-gpu-h100",
            admit_nightly_tests=True,
            labels=set(resolve_policy(NIGHTLY_CADENCE, set()).include_labels),
        )
        # A test whose only labels were subtracted is out of scope entirely,
        # including from the skip report.
        assert _names(enabled) == {"tests/e2e/precision.py"}
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
            _make("tests/e2e/h100/t.py", suite="stage-c-8-gpu-h100", labels=["precision"]),
            _make("tests/e2e/h200/t.py", suite="stage-c-4-gpu-h200", labels=["precision"]),
        ]
        enabled, _ = filter_tests(
            tests,
            HWBackend.CUDA,
            "stage-c-8-gpu-h100",
            labels=_ALL,
        )
        assert _names(enabled) == {"tests/e2e/h100/t.py"}

    def test_cross_backend_isolation(self):
        # CPU suite must not pull in CUDA registrations.
        tests = [
            _make("tests/fast/t.py", backend=HWBackend.CPU, suite="stage-a-cpu", labels=[]),
            _make(
                "tests/e2e/h100/t.py",
                backend=HWBackend.CUDA,
                suite="stage-c-8-gpu-h100",
                labels=["precision"],
            ),
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
            admit_nightly_tests=False,
            labels={"megatron"},
        )
        assert _names(enabled) == {"tests/e2e/regular.py"}

    def test_nightly_run_includes_regular_and_nightly_only(self):
        enabled, _ = filter_tests(
            self._cadence_tests(),
            HWBackend.CUDA,
            "stage-c-8-gpu-h100",
            admit_nightly_tests=True,
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
            admit_nightly_tests=False,
            labels={"megatron"},
        )
        _, nightly_skipped = filter_tests(
            tests,
            HWBackend.CUDA,
            "stage-c-8-gpu-h100",
            admit_nightly_tests=True,
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
            admit_nightly_tests=True,
            labels=set(standard_policy.include_labels),
        )
        explicit, _ = filter_tests(
            tests,
            HWBackend.CUDA,
            "stage-c-8-gpu-h100",
            admit_nightly_tests=True,
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
        # The fast GPU bucket remains a first-class suite.
        tests = [
            _make("tests/fast/q.py", suite="stage-b-2-gpu-h200", labels=["precision"]),
        ]
        enabled, _ = filter_tests(
            tests,
            HWBackend.CUDA,
            "stage-b-2-gpu-h200",
            labels={"precision"},
        )
        assert _names(enabled) == {"tests/fast/q.py"}
